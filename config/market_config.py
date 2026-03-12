# market_config.py
# Multi-market configuration system.
# Each market defines its own ticker map, system prompt, decimal conventions, etc.

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


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
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_eval_retries: int = 5
    min_eval_score: float = 0.95


# ── US market (default) ──────────────────────────────────────────────
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
)


# ── Argentina market (placeholder – fill in real tickers) ────────────
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
