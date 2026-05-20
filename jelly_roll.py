"""Jelly Roll: forward-starting financing trade built from two boxes.

A box at expiry T locks in financing from today -> T at rate r(T) where
  B(T) = (K2 - K1) * exp(-r(T) * T)

A jelly roll combines a short box at T1 and a long box at T2 (same K1, K2):
  legs at T1: -call(K1) +put(K1) +call(K2) -put(K2)
  legs at T2: +call(K1) -put(K1) -call(K2) +put(K2)

Cash flows:
  today: pay (B2 - B1)
  at T1: pay  (K2 - K1)          [from short box settling]
  at T2: receive (K2 - K1)       [from long box settling]

This is a synthetic zero-coupon loan from T1 to T2. The implied
forward continuously-compounded rate is:
  f(T1, T2) = (r2 * T2 - r1 * T1) / (T2 - T1)

Equivalently, from the executable debits/credits:
  f_lend(T1, T2)   = ln( box1_sell_worst / box2_buy_worst ) / (T2 - T1)
                     + adjustment for the payoff ratio (cancels when strikes match)
We compute it directly from r1, r2 at the chosen execution prices.

Sign conventions for the worst-case fills (you cross the spread on every leg):
  box_buy_worst  = ask_C(K1) - bid_P(K1) - bid_C(K2) + ask_P(K2)   (max debit to buy)
  box_sell_worst = bid_C(K1) - ask_P(K1) - ask_C(K2) + bid_P(K2)   (min credit to sell)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

import pandas as pd

from box import _mid, years_to_expiry


@dataclass
class BoxQuote:
    """A single box priced at a specific strike pair and expiry."""
    expiry: str
    K1: float
    K2: float
    t_years: float
    payoff: float          # K2 - K1
    debit_mid: float       # mid debit to buy the box
    buy_worst: float       # debit if you cross every leg buying
    sell_worst: float      # credit you receive if you cross every leg selling

    def rate_mid(self) -> float | None:
        if self.debit_mid <= 0:
            return None
        return math.log(self.payoff / self.debit_mid) / self.t_years

    def rate_buy_worst(self) -> float | None:
        """Rate you EARN by buying the box at worst price (lower bound on yield)."""
        if self.buy_worst <= 0:
            return None
        return math.log(self.payoff / self.buy_worst) / self.t_years

    def rate_sell_worst(self) -> float | None:
        """Rate you PAY by selling the box at worst price (upper bound on cost)."""
        if self.sell_worst <= 0:
            return None
        return math.log(self.payoff / self.sell_worst) / self.t_years


@dataclass
class JellyRoll:
    expiry_near: str
    expiry_far: str
    K1: float
    K2: float
    t1_years: float
    t2_years: float
    payoff: float                  # K2 - K1, identical at both expiries
    net_debit_mid: float           # B2_mid - B1_mid (pay today)
    net_debit_worst: float         # buy far at ask, sell near at bid: most you'd pay
    net_credit_worst: float        # sell far at bid, buy near at ask: least you'd receive
    fwd_rate_mid: float            # continuously compounded forward T1->T2
    fwd_rate_lend_worst: float     # forward rate you LOCK IN by lending (buy far / sell near at worst)
    fwd_rate_borrow_worst: float   # forward rate you LOCK IN by borrowing (sell far / buy near at worst)


def _pivot_chain(chain: pd.DataFrame, expiry: str) -> pd.DataFrame | None:
    sub = chain[chain["expiry"] == expiry]
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        index="strike", columns="right", values=["bid", "ask"], aggfunc="first"
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    needed = ["bid_C", "ask_C", "bid_P", "ask_P"]
    if not all(c in pivot.columns for c in needed):
        return None
    return pivot.dropna(subset=needed)


def _price_box(pivot: pd.DataFrame, K1: float, K2: float, expiry: str, t: float) -> BoxQuote | None:
    if K1 not in pivot.index or K2 not in pivot.index:
        return None
    c_k1 = _mid(pivot.at[K1, "bid_C"], pivot.at[K1, "ask_C"])
    p_k1 = _mid(pivot.at[K1, "bid_P"], pivot.at[K1, "ask_P"])
    c_k2 = _mid(pivot.at[K2, "bid_C"], pivot.at[K2, "ask_C"])
    p_k2 = _mid(pivot.at[K2, "bid_P"], pivot.at[K2, "ask_P"])

    debit_mid = c_k1 - p_k1 - c_k2 + p_k2
    buy_worst = (
        pivot.at[K1, "ask_C"] - pivot.at[K1, "bid_P"]
        - pivot.at[K2, "bid_C"] + pivot.at[K2, "ask_P"]
    )
    sell_worst = (
        pivot.at[K1, "bid_C"] - pivot.at[K1, "ask_P"]
        - pivot.at[K2, "ask_C"] + pivot.at[K2, "bid_P"]
    )
    return BoxQuote(
        expiry=expiry, K1=K1, K2=K2, t_years=t,
        payoff=K2 - K1,
        debit_mid=debit_mid, buy_worst=buy_worst, sell_worst=sell_worst,
    )


def _forward_rate(r1: float, t1: float, r2: float, t2: float) -> float:
    return (r2 * t2 - r1 * t1) / (t2 - t1)


def best_jelly_roll(
    chain: pd.DataFrame,
    expiry_near: str,
    expiry_far: str,
    min_width: float = 50,
    asof: datetime | None = None,
) -> JellyRoll | None:
    """Find the (K1, K2) pair that maximizes the lend-side worst forward rate.

    Lend-side = the forward rate you can actually lock in by buying the far box
    (paying the ask side) and selling the near box (hitting the bid side).
    If this exceeds the SOFR/Treasury forward, you have a clean financing arb.
    """
    p1 = _pivot_chain(chain, expiry_near)
    p2 = _pivot_chain(chain, expiry_far)
    if p1 is None or p2 is None:
        return None

    common_strikes = sorted(set(p1.index) & set(p2.index))
    if len(common_strikes) < 2:
        return None

    t1 = years_to_expiry(expiry_near, asof)
    t2 = years_to_expiry(expiry_far, asof)
    if t2 <= t1:
        return None

    best: JellyRoll | None = None
    for i, K1 in enumerate(common_strikes):
        for K2 in common_strikes[i + 1:]:
            if (K2 - K1) < min_width:
                continue

            b1 = _price_box(p1, K1, K2, expiry_near, t1)
            b2 = _price_box(p2, K1, K2, expiry_far, t2)
            if b1 is None or b2 is None:
                continue

            # Both directions must be priceable
            r1_mid = b1.rate_mid()
            r2_mid = b2.rate_mid()
            if r1_mid is None or r2_mid is None:
                continue

            # Lending forward: buy far box at ask, sell near box at bid.
            # The rate locked in equals the forward derived from those exact prices.
            r1_sell = b1.rate_sell_worst()   # rate corresponding to selling near
            r2_buy = b2.rate_buy_worst()     # rate corresponding to buying far
            if r1_sell is None or r2_buy is None:
                continue
            f_lend_worst = _forward_rate(r1_sell, t1, r2_buy, t2)

            # Borrowing forward: sell far box at bid, buy near box at ask.
            r1_buy = b1.rate_buy_worst()
            r2_sell = b2.rate_sell_worst()
            if r1_buy is None or r2_sell is None:
                continue
            f_borrow_worst = _forward_rate(r1_buy, t1, r2_sell, t2)

            f_mid = _forward_rate(r1_mid, t1, r2_mid, t2)

            jr = JellyRoll(
                expiry_near=expiry_near, expiry_far=expiry_far,
                K1=K1, K2=K2,
                t1_years=t1, t2_years=t2,
                payoff=K2 - K1,
                net_debit_mid=b2.debit_mid - b1.debit_mid,
                net_debit_worst=b2.buy_worst - b1.sell_worst,
                net_credit_worst=b2.sell_worst - b1.buy_worst,
                fwd_rate_mid=f_mid,
                fwd_rate_lend_worst=f_lend_worst,
                fwd_rate_borrow_worst=f_borrow_worst,
            )
            if best is None or jr.fwd_rate_lend_worst > best.fwd_rate_lend_worst:
                best = jr
    return best


def scan_forward_curve(
    chain: pd.DataFrame,
    asof: datetime | None = None,
    min_width: float = 50,
    expiries: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build the implied forward financing curve from consecutive expiry pairs."""
    if expiries is None:
        expiries = sorted(chain["expiry"].unique())
    expiries = list(expiries)

    rows = []
    for near, far in zip(expiries, expiries[1:]):
        jr = best_jelly_roll(chain, near, far, min_width=min_width, asof=asof)
        if jr:
            rows.append(asdict(jr))
    return pd.DataFrame(rows)
