# utils.py
# Helpers: limpieza, fetch URL, chunking, SimpleDoc, embedder factory, formatting CSV->prompt

import os, re, unicodedata, datetime
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd

# you may need pandas/numpy in app; import there to avoid extra deps here

DATE_TEXT_PATTERNS = [
    re.compile(r"(?P<d>[0-3]?\d)[\-\/\.](?P<m>[01]?\d)[\-\/\.](?P<y>\d{4})"),
    re.compile(r"(?P<y>\d{4})[\-\/\.](?P<m>[01]?\d)[\-\/\.](?P<d>[0-3]?\d)"),
]
MESES = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
         "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
         "noviembre":11,"diciembre":12}

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ").replace("\u00AD", "").replace("\u200B", "")
    s = re.sub(r"[ \t]+", " ", s)
    # escape rarely used markdown triggers for safety
    s = s.replace("#", "＃").replace("*", "＊").replace("_", "﹎")
    return s.strip()

def fetch_url_text(url: str, timeout: int = 8) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        ps = soup.find_all("p")
        txt = "\n\n".join(p.get_text(strip=True) for p in ps if p.get_text(strip=True))
        return clean_text(txt)[:20000]
    except Exception:
        return ""

def fetch_news_for_date(target_date: str, sources: list, keywords: list = None, timeout: int = 8) -> tuple:
    """Fetch text from news sources and filter by market-relevant keywords.

    Returns (combined_text, has_relevant_news) where has_relevant_news is True
    only if at least one source contained a recognized market keyword.
    """
    parts = []
    has_relevant = False
    kw_lower = [k.lower() for k in keywords] if keywords else []

    for url in sources:
        txt = fetch_url_text(url, timeout=timeout)
        if not txt:
            continue
        txt_lower = txt.lower()
        is_relevant = any(kw in txt_lower for kw in kw_lower) if kw_lower else True
        if is_relevant:
            has_relevant = True
        parts.append(f"[Fuente: {url}]\n{txt[:3000]}")

    return "\n\n".join(parts), has_relevant


def extract_date_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    for pat in DATE_TEXT_PATTERNS:
        m = pat.search(t)
        if m:
            y = int(m.group("y")); mth = int(m.group("m")); d = int(m.group("d"))
            return f"{y:04d}-{mth:02d}-{d:02d}"
    m = re.search(r"(?P<d>[0-3]?\d)\s+de\s+(?P<mes>\w+)\s+de\s+(?P<y>\d{4})", t)
    if m:
        d = int(m.group("d")); y = int(m.group("y"))
        mes_name = m.group("mes")
        if mes_name in MESES:
            return f"{y:04d}-{MESES[mes_name]:02d}-{d:02d}"
    return None

class SimpleDoc:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

# small formatting helper for CSV->single-line prompt chunk
def format_variations_for_prompt(df: "pd.DataFrame", source_name: str = "variacion_diaria.csv") -> str:
    """
    Convierte todo el DataFrame en UN SOLO string (una única fuente).
    - Mantiene la precisión numérica (no redondea).
    - Convierte Var_diaria_% a formato humano con coma decimal y '%' si existe.
    - Usa '|' como separador de campos para evitar ambigüedad con la coma decimal.
    - Devuelve: "Fuente:<source_name> Tickers:\nheader\nfila1\nfila2\n..."
    """
    SEP = " | "

    def fmt_pct(v):
        if v is None:
            return ""
        # usa pandas.isna para detectar NaN/NA
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        s = f"{v}"
        s = s.replace(".", ",")
        return s + "%"

    rows = []
    # Header para que el LLM entienda cada columna
    rows.append(f"Ticker{SEP}YahooSymbol{SEP}Close_last{SEP}Close_prev{SEP}Var_diaria_%{SEP}Fecha_last")
    for _, r in df.iterrows():
        ticker = r.get("Ticker", "") or ""
        sym = r.get("YahooSymbol", "") or ""

        cl = r.get("Close_last", "")
        cl_str = "" if (cl is None or (isinstance(cl, float) and pd.isna(cl))) else f"{cl}"

        prev = r.get("Close_prev", "")
        prev_str = "" if (prev is None or (isinstance(prev, float) and pd.isna(prev))) else f"{prev}"

        pct = r.get("Var_diaria_%", "")
        pct_str = fmt_pct(pct) if pct != "" and not (isinstance(pct, float) and pd.isna(pct)) else ""

        date = r.get("Fecha_last", "")
        date_str = "" if (date is None or (isinstance(date, float) and pd.isna(date))) else f"{date}"

        rows.append(f"{ticker}{SEP}{sym}{SEP}{cl_str}{SEP}{prev_str}{SEP}{pct_str}{SEP}{date_str}")

    # Una única fuente para todo el CSV
    body = "\n".join(rows)
    return f"Fuente:{source_name} Tickers:\n{body}"


def df_to_single_doc(df: "pd.DataFrame", source_name: str = "variacion_diaria.csv", extra_meta: dict = None) -> SimpleDoc:
    """
    Devuelve un único SimpleDoc que representa TODO el CSV.
    Úsalo en tu pipeline de RAG para indexar este CSV como una sola fuente/chunk.
    """
    content = format_variations_for_prompt(df, source_name=source_name)
    metadata = {"source": source_name, "type": "csv", "rows": len(df)}
    if extra_meta:
        metadata.update(extra_meta)
    return SimpleDoc(page_content=content, metadata=metadata)