"""Scan SPX implied FORWARD financing curve via jelly rolls.

For each consecutive pair of expiries (T1 < T2), build the best jelly roll
(short box at T1 + long box at T2) and report the forward financing rate
locked in by the trade. Compare to the SOFR/Treasury forward.

Run:
  pip install -r requirements.txt
  cp .env.example .env  # add FRED_API_KEY
  # Start TWS/Gateway, ensure API is enabled on port 7497 (paper)
  python scan_spx_forward.py
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ibkr_client import (
    connect,
    fetch_quotes,
    get_index_contract,
    get_option_chain_metadata,
)
from jelly_roll import scan_forward_curve
from rates import forward_rate, get_riskfree_curve


SYMBOL = "SPX"
EXCHANGE = "CBOE"
TRADING_CLASS = "SPXW"
STRIKE_WINDOW_PCT = 0.05
MAX_STRIKES = 12
MAX_EXPIRIES = 6
MIN_BOX_WIDTH = 100


def main():
    ib = connect()
    try:
        index = get_index_contract(ib, SYMBOL, EXCHANGE)
        [ticker] = ib.reqTickers(index)
        spot = ticker.marketPrice() or ticker.close
        print(f"{SYMBOL} spot: {spot:.2f}")

        meta = get_option_chain_metadata(ib, index)
        meta = meta[meta["trading_class"] == TRADING_CLASS]

        lo, hi = spot * (1 - STRIKE_WINDOW_PCT), spot * (1 + STRIKE_WINDOW_PCT)
        strikes_in_window = sorted(s for s in meta["strike"].unique() if lo <= s <= hi)
        stride = max(1, len(strikes_in_window) // MAX_STRIKES)
        strikes = strikes_in_window[::stride]

        expiries = sorted(meta["expiry"].unique())[:MAX_EXPIRIES]
        print(f"Scanning {len(expiries)} expiries x {len(strikes)} strikes "
              f"= {len(expiries) * len(strikes) * 2} contracts")

        chain_frames = []
        for exp in expiries:
            print(f"  fetching {exp}...")
            chain_frames.append(
                fetch_quotes(ib, SYMBOL, exp, strikes,
                             exchange="SMART", trading_class=TRADING_CLASS)
            )
        chain = pd.concat(chain_frames, ignore_index=True)

        fwd = scan_forward_curve(chain, min_width=MIN_BOX_WIDTH)
        if fwd.empty:
            print("No valid jelly rolls found.")
            return

        try:
            curve = get_riskfree_curve()
            fwd["benchmark_fwd_rate"] = fwd.apply(
                lambda row: forward_rate(
                    curve, row["t1_years"] * 365.25, row["t2_years"] * 365.25
                ),
                axis=1,
            )
            fwd["lend_spread_bps"] = (fwd["fwd_rate_lend_worst"] - fwd["benchmark_fwd_rate"]) * 1e4
            fwd["borrow_spread_bps"] = (fwd["fwd_rate_borrow_worst"] - fwd["benchmark_fwd_rate"]) * 1e4
        except RuntimeError as e:
            print(f"Skipping benchmark comparison: {e}")

        out = f"{SYMBOL}_forward_curve_{datetime.now():%Y%m%d_%H%M}.csv"
        fwd.to_csv(out, index=False)
        print(f"\nSaved {out}")
        print(fwd.to_string(index=False))
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
