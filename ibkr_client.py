"""IBKR connection + option chain fetching via ib_insync.

Prerequisites:
  - TWS or IB Gateway running locally and logged in
  - API access enabled (Configuration > API > Settings > Enable ActiveX and Socket Clients)
  - Default ports: 7497 (TWS paper), 7496 (TWS live), 4002 (Gateway paper), 4001 (Gateway live)
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from ib_insync import IB, Index, Option


def connect(host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    # 3 = delayed-frozen, works without a market-data subscription
    ib.reqMarketDataType(3)
    return ib


def get_index_contract(ib: IB, symbol: str, exchange: str = "CBOE") -> Index:
    contract = Index(symbol, exchange, "USD")
    ib.qualifyContracts(contract)
    return contract


def get_option_chain_metadata(ib: IB, underlying: Index) -> pd.DataFrame:
    """Return DataFrame of available (expiry, strike, trading_class) for the index."""
    chains = ib.reqSecDefOptParams(
        underlying.symbol, "", underlying.secType, underlying.conId
    )
    rows = []
    for chain in chains:
        for expiry in chain.expirations:
            for strike in chain.strikes:
                rows.append({
                    "expiry": expiry,
                    "strike": strike,
                    "trading_class": chain.tradingClass,
                    "exchange": chain.exchange,
                    "multiplier": chain.multiplier,
                })
    return pd.DataFrame(rows)


def fetch_quotes(
    ib: IB,
    symbol: str,
    expiry: str,
    strikes: Iterable[float],
    exchange: str = "SMART",
    trading_class: str | None = None,
) -> pd.DataFrame:
    """Snapshot bid/ask for both calls and puts at given strikes/expiry."""
    contracts = []
    for strike in strikes:
        for right in ("C", "P"):
            opt = Option(symbol, expiry, strike, right, exchange, currency="USD")
            if trading_class:
                opt.tradingClass = trading_class
            contracts.append(opt)

    ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts)

    rows = []
    for t in tickers:
        c = t.contract
        rows.append({
            "expiry": c.lastTradeDateOrContractMonth,
            "strike": c.strike,
            "right": c.right,
            "bid": t.bid if t.bid and t.bid > 0 else None,
            "ask": t.ask if t.ask and t.ask > 0 else None,
            "last": t.last if t.last and t.last > 0 else None,
        })
    return pd.DataFrame(rows)
