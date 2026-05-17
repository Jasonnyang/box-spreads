"""MVP: scan SPX implied financing curve and compare to risk-free benchmarks.

Run:
  pip install -r requirements.txt
  cp .env.example .env  # add FRED_API_KEY
  # Start TWS/Gateway, ensure API is enabled on port 7497 (paper)
  python scan_spx.py
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from box import scan_term_structure
from ibkr_client import (
    connect,
    fetch_quotes,
    get_index_contract,
    get_option_chain_metadata,
)
from rates import get_riskfree_curve, interpolate_rate


SYMBOL = "SPX"
EXCHANGE = "CBOE"
TRADING_CLASS = "SPXW"     # PM-settled weeklys; "SPX" for AM-settled monthlies
STRIKE_WINDOW_PCT = 0.05   # +/- 5% of spot
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

        ts = scan_term_structure(chain, min_width=MIN_BOX_WIDTH)
        if ts.empty:
            print("No valid boxes found.")
            return

        try:
            curve = get_riskfree_curve()
            ts["benchmark_rate"] = ts["t_years"].apply(
                lambda t: interpolate_rate(curve, t * 365.25)
            )
            ts["spread_bps_mid"] = (ts["rate_mid"] - ts["benchmark_rate"]) * 1e4
            ts["spread_bps_worst"] = (ts["rate_worst"] - ts["benchmark_rate"]) * 1e4
        except RuntimeError as e:
            print(f"Skipping benchmark comparison: {e}")

        out = f"{SYMBOL}_term_structure_{datetime.now():%Y%m%d_%H%M}.csv"
        ts.to_csv(out, index=False)
        print(f"\nSaved {out}")
        print(ts.to_string(index=False))
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
