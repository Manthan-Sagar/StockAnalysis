"""
Universe Construction Module
Defines the static equity universe (~100 liquid large/mid-cap names across
multiple GICS sectors) and provides sector mappings for cross-sectional modeling.

SURVIVORSHIP BIAS CAVEAT: This universe uses currently-listed tickers and therefore
suffers from survivorship bias — delisted, acquired, and dropped names are excluded.
The pragmatic trade-off is acknowledged; a production system would reconstruct
point-in-time index membership per rebalance date.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Static universe: ~100 liquid large / mid-cap US equities, 2012-2024
# Selected across sectors to avoid tech-only concentration bias
# ---------------------------------------------------------------------------

UNIVERSE: Dict[str, List[str]] = {
    # ── Technology (20) ───────────────────────────────────────────────
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        "ADBE", "CRM", "INTC", "CSCO", "ORCL",
        "TXN", "AVGO", "QCOM", "IBM", "NOW",
        "AMD", "MU", "AMAT", "LRCX", "KLAC",
    ],
    # ── Healthcare (12) ──────────────────────────────────────────────
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABT", "TMO",
        "MRK", "LLY", "AMGN", "MDT", "BMY",
        "GILD", "ISRG",
    ],
    # ── Financials (12) ──────────────────────────────────────────────
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "BLK", "SCHW", "AXP", "USB",
        "PNC", "TFC",
    ],
    # ── Industrials (10) ─────────────────────────────────────────────
    "Industrials": [
        "HON", "UPS", "UNP", "BA", "CAT",
        "GE", "MMM", "LMT", "RTX", "DE",
    ],
    # ── Energy (8) ───────────────────────────────────────────────────
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "MPC", "PSX", "VLO",
    ],
    # ── Consumer Discretionary (10) ──────────────────────────────────
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD",
        "SBUX", "LOW", "TJX", "BKNG", "CMG",
    ],
    # ── Consumer Staples (8) ─────────────────────────────────────────
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT",
        "CL", "MDLZ", "MO",
    ],
    # ── Communication Services (6) ───────────────────────────────────
    "Communication Services": [
        "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS",
    ],
    # ── Materials (4) ────────────────────────────────────────────────
    "Materials": [
        "LIN", "APD", "ECL", "NEM",
    ],
    # ── Utilities (4) ────────────────────────────────────────────────
    "Utilities": [
        "NEE", "DUK", "SO", "D",
    ],
    # ── Real Estate (4) ──────────────────────────────────────────────
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX",
    ],
}

# Benchmark instruments (not traded in the portfolio — used for
# market-beta residualization and performance comparison)
BENCHMARKS: List[str] = ["SPY", "QQQ"]

# Date range for the study
START_DATE: str = "2012-01-01"
END_DATE: str = "2024-12-31"


def get_all_tickers() -> List[str]:
    """Return a flat, sorted list of all tickers in the universe."""
    tickers: List[str] = []
    for sector_tickers in UNIVERSE.values():
        tickers.extend(sector_tickers)
    return sorted(set(tickers))


def get_sector_map() -> Dict[str, str]:
    """Return a mapping {ticker: sector} for every ticker in the universe."""
    mapping: Dict[str, str] = {}
    for sector, tickers in UNIVERSE.items():
        for t in tickers:
            mapping[t] = sector
    return mapping


def get_sector_codes() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Return integer-encoded sector mappings for LightGBM categorical features.

    Returns
    -------
    sector_to_code : dict
        {sector_name: int_code}
    code_to_sector : dict
        {int_code: sector_name}
    """
    sectors = sorted(UNIVERSE.keys())
    sector_to_code = {s: i for i, s in enumerate(sectors)}
    code_to_sector = {i: s for i, s in enumerate(sectors)}
    return sector_to_code, code_to_sector


def save_universe_metadata(output_dir: Path = Path("data/processed")) -> Path:
    """
    Persist universe metadata to JSON for reproducibility.

    Parameters
    ----------
    output_dir : Path
        Directory to write universe.json.

    Returns
    -------
    Path
        Written file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tickers = get_all_tickers()
    sector_map = get_sector_map()

    metadata = {
        "description": (
            "Static universe of ~100 liquid US large/mid-cap equities, 2012-2024. "
            "SURVIVORSHIP BIAS: uses currently-listed tickers only."
        ),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "n_tickers": len(all_tickers),
        "tickers": all_tickers,
        "benchmarks": BENCHMARKS,
        "sectors": {sector: tickers for sector, tickers in UNIVERSE.items()},
        "sector_map": sector_map,
    }

    path = output_dir / "universe.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[universe] Saved universe metadata -> {path}  ({len(all_tickers)} tickers)")
    return path


if __name__ == "__main__":
    save_universe_metadata()
    tickers = get_all_tickers()
    print(f"\nUniverse: {len(tickers)} tickers across {len(UNIVERSE)} sectors")
    for sector, names in UNIVERSE.items():
        print(f"  {sector:30s} ({len(names):2d}): {', '.join(names)}")
