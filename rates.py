"""Risk-free rate benchmarks from FRED."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "SOFR": ("SOFR", 1),
    "T1M": ("DGS1MO", 30),
    "T3M": ("DGS3MO", 91),
    "T6M": ("DGS6MO", 182),
    "T1Y": ("DGS1", 365),
    "T2Y": ("DGS2", 730),
    "T5Y": ("DGS5", 1825),
}


def _fred_latest(series_id: str, api_key: str) -> float | None:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    r = requests.get(FRED_BASE, params=params, timeout=10)
    r.raise_for_status()
    for obs in r.json().get("observations", []):
        if obs["value"] != ".":
            return float(obs["value"]) / 100
    return None


def get_riskfree_curve() -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set FRED_API_KEY in your .env (free signup at "
            "https://fred.stlouisfed.org/docs/api/api_key.html)"
        )
    rows = []
    for label, (sid, tenor_days) in SERIES.items():
        rate = _fred_latest(sid, api_key)
        if rate is not None:
            rows.append({"label": label, "tenor_days": tenor_days, "rate": rate})
    return pd.DataFrame(rows).sort_values("tenor_days").reset_index(drop=True)


def interpolate_rate(curve: pd.DataFrame, tenor_days: float) -> float:
    return float(np.interp(tenor_days, curve["tenor_days"], curve["rate"]))
