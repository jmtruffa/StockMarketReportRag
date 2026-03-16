# market_config.py
# Multi-market configuration system.
# Each market defines its own ticker map, system prompt, decimal conventions, etc.

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketConfig:
    """Configuration for a specific financial market."""
    market_id: str                          # e.g. "US", "AR"
    market_name: str                        # e.g. "Estados Unidos", "Argentina"
    ticker_map: Dict[str, str]              # label -> Yahoo Finance symbol
    system_prompt_path: str                 # path to systemprompt template
    rag_prompt_path: str                    # path to RAG prompt template
    threshold_dataset_path: str             # path to evaluation dataset
    decimal_separator: str = ","            # "," for AR/ES locale, "." for US
    percentage_suffix: str = "%"
    currency_symbol: str = "USD"
    default_lookback: str = "30d"
    timezone: str = "America/New_York"
    openai_model: str = "gpt-5-mini"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_eval_retries: int = 5
    min_eval_score: float = 0.95
    news_keywords: List[str] = field(default_factory=list)  # keywords to detect relevant news


# ── US market (default) ──────────────────────────────────────────────
US_NEWS_KEYWORDS: List[str] = [
    # Indices & macro
    "S&P 500", "S&P500", "SPX", "Nasdaq", "NDX", "Dow Jones", "VIX", "volatility index",
    "Federal Reserve", "Fed", "Treasury", "Wall Street", "stock market",
    "crude oil", "WTI", "oil prices", "gold prices",
    # ETFs (by name)
    "Energy Select", "Communication Services", "Consumer Staples",
    "Technology Select", "Health Care Select", "semiconductor",
    "iShares Latin America", "iShares MSCI Brazil", "emerging markets bond",
    "Defiance Quantum", "SPDR Gold",
    # Tickers (symbols also appear in news)
    "XLE", "XLC", "XLP", "XLK", "XLV", "SOXX", "GLD", "ILF", "EWZ", "EMB",
    # Companies
    "Tesla", "TSLA", "Apple", "AAPL", "Alphabet", "Google", "GOOG",
    "Nvidia", "NVDA", "Meta", "META", "Microsoft", "MSFT", "Amazon", "AMZN",
    "Rigetti", "RGTI", "D-Wave", "QBTS", "IonQ", "IONQ",
]

US_TICKER_MAP: Dict[str, str] = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "VIX": "^VIX",
    "ILF": "ILF",
    "EWZ": "EWZ",
    "EMB": "EMB",
    "/CL": "CL=F",
    "GLD": "GLD",
    "XLE": "XLE",
    "XLC": "XLC",
    "XLP": "XLP",
    "XLK": "XLK",
    "XLV": "XLV",
    "QTUM": "QTUM",
    "SOXX": "SOXX",
    "TSLA": "TSLA",
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "NVDA": "NVDA",
    "META": "META",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "RGTI": "RGTI",
    "QBTS": "QBTS",
    "IONQ": "IONQ",
}

US_CONFIG = MarketConfig(
    market_id="US",
    market_name="Estados Unidos",
    ticker_map=US_TICKER_MAP,
    system_prompt_path="./prompts/systemprompt_template.txt",
    rag_prompt_path="./prompts/systemprompt_template.txt",
    threshold_dataset_path="./data/threshold_dataset.jsonl",
    decimal_separator=",",
    currency_symbol="USD",
    timezone="America/New_York",
    news_keywords=US_NEWS_KEYWORDS,
)


# ── Argentina market ─────────────────────────────────────────────────
AR_NEWS_KEYWORDS: List[str] = [
    # Index & market
    "Merval", "ADR", "Argentina", "bolsa", "acciones argentinas",
    # Companies (expanded names)
    "YPF", "Grupo Financiero Galicia", "Galicia", "GGAL",
    "Banco Macro", "BMA", "BBVA Argentina", "BBAR",
    "Supervielle", "SUPV",
    "Central Puerto", "CEPU",
    "Edenor", "EDN",
    "Loma Negra", "LOMA",
    "Telecom Argentina", "TEO",
    "Transportadora de Gas del Sur", "TGS",
    "Pampa Energía", "Pampa Energia", "PAM",
    "Cresud", "CRESY",
    "Ternium", "TX",
    "IRSA", "IRS",
    "VISTA", "VIST",
]

AR_TICKER_MAP: Dict[str, str] = {
    "GGAL":   "GGAL",
    "YPF":    "YPF",
    "BMA":    "BMA",
    "BBAR":   "BBAR",
    "SUPV":   "SUPV",
    "CEPU":   "CEPU",
    "EDN":    "EDN",
    "LOMA":   "LOMA",
    "TEO":    "TEO",
    "TGS":    "TGS",
    "PAM":    "PAM",
    "CRESY":  "CRESY",
    "TX":     "TX",
    "IRS":    "IRS",
    "VIST":   "VIST",
}

AR_CONFIG = MarketConfig(
    market_id="AR",
    market_name="Argentina",
    ticker_map=AR_TICKER_MAP,
    system_prompt_path="./prompts/systemprompt_template_ar.txt",
    rag_prompt_path="./prompts/systemprompt_template_ar.txt",
    threshold_dataset_path="./data/threshold_dataset_ar.jsonl",
    decimal_separator=",",
    currency_symbol="ARS",
    timezone="America/Argentina/Buenos_Aires",
    news_keywords=AR_NEWS_KEYWORDS,
    min_eval_score=0.88,
)


# ── Registry ─────────────────────────────────────────────────────────
MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "US": US_CONFIG,
    "AR": AR_CONFIG,
}


def get_market_config(market_id: str = "US") -> MarketConfig:
    """Return config for a given market id (case-insensitive)."""
    key = market_id.upper()
    if key not in MARKET_CONFIGS:
        available = ", ".join(MARKET_CONFIGS.keys())
        raise ValueError(f"Unknown market '{market_id}'. Available: {available}")
    return MARKET_CONFIGS[key]
