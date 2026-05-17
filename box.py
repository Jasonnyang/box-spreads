"""Box spread math + scanner.

A long box at strikes K1 < K2 expiring at T:
  legs: +call(K1) -put(K1) -call(K2) +put(K2)
  payoff at expiry: (K2 - K1), guaranteed if options are European-style cash-settled
  debit today: B = C(K1) - P(K1) - C(K2) + P(K2)
  implied continuously-compounded rate: r = ln((K2-K1) / B) / T
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd


@dataclass
class Box:
    expiry: str
    K1: float
    K2: float
    debit_mid: float
    debit_worst: float   # what you'd actually pay crossing the spread
    payoff: float        # K2 - K1
    t_years: float
    rate_mid: float      # continuously compounded, annualized
    rate_worst: float


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2


def years_to_expiry(expiry: str, asof: datetime | None = None) -> float:
    asof = asof or datetime.now()
    exp_dt = datetime.strptime(expiry, "%Y%m%d")
    return max((exp_dt - asof).total_seconds() / (365.25 * 86400), 1 / 365.25)


def best_box_for_expiry(
    chain: pd.DataFrame,
    expiry: str,
    min_width: float = 50,
    asof: datetime | None = None,
) -> Box | None:
    """Find the strike pair maximizing the implied financing rate for this expiry.

    `chain` columns: expiry, strike, right ('C'/'P'), bid, ask
    """
    sub = chain[chain["expiry"] == expiry]
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index="strike", columns="right", values=["bid", "ask"], aggfunc="first"
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    needed = ["bid_C", "ask_C", "bid_P", "ask_P"]
    pivot = pivot.dropna(subset=needed)
    strikes = sorted(pivot.index.tolist())

    t = years_to_expiry(expiry, asof)
    best: Box | None = None

    for i, K1 in enumerate(strikes):
        for K2 in strikes[i + 1:]:
            if (K2 - K1) < min_width:
                continue

            c_k1 = _mid(pivot.at[K1, "bid_C"], pivot.at[K1, "ask_C"])
            p_k1 = _mid(pivot.at[K1, "bid_P"], pivot.at[K1, "ask_P"])
            c_k2 = _mid(pivot.at[K2, "bid_C"], pivot.at[K2, "ask_C"])
            p_k2 = _mid(pivot.at[K2, "bid_P"], pivot.at[K2, "ask_P"])

            debit_mid = c_k1 - p_k1 - c_k2 + p_k2
            debit_worst = (
                pivot.at[K1, "ask_C"]
                - pivot.at[K1, "bid_P"]
                - pivot.at[K2, "bid_C"]
                + pivot.at[K2, "ask_P"]
            )

            payoff = K2 - K1
            if debit_mid <= 0 or debit_worst <= 0:
                continue

            rate_mid = math.log(payoff / debit_mid) / t
            rate_worst = math.log(payoff / debit_worst) / t

            if best is None or rate_worst > best.rate_worst:
                best = Box(
                    expiry=expiry, K1=K1, K2=K2,
                    debit_mid=debit_mid, debit_worst=debit_worst,
                    payoff=payoff, t_years=t,
                    rate_mid=rate_mid, rate_worst=rate_worst,
                )
    return best


def scan_term_structure(
    chain: pd.DataFrame, asof: datetime | None = None, min_width: float = 50
) -> pd.DataFrame:
    rows = []
    for expiry in sorted(chain["expiry"].unique()):
        b = best_box_for_expiry(chain, expiry, min_width=min_width, asof=asof)
        if b:
            rows.append(asdict(b))
    return pd.DataFrame(rows)
