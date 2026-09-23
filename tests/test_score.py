import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import score_model as sm


def info(eps=5, price=100, **kw):
    base = {"revenueGrowth": .1, "earningsGrowth": .1, "returnOnEquity": .2, "returnOnAssets": .08,
            "operatingMargins": .2, "profitMargins": .15, "grossMargins": .5, "ebitdaMargins": .25,
            "freeCashflow": 5e8, "marketCap": 1e10, "totalRevenue": 4e9, "netIncomeToCommon": 6e8,
            "debtToEquity": 60, "beta": 1.0, "currentRatio": 1.5, "currentPrice": price,
            "trailingEps": eps, "bookValue": 30, "ebitda": 1e9, "enterpriseValue": 1.1e10,
            "pegRatio": 1.5, "shortName": "X", "sector": "Technology"}
    base.update(kw); return base


def test_loss_maker_ranks_last_in_valuation():
    infos = {f"T{i}": info(eps=e) for i, e in enumerate([-8, 2, 4, 6, 8, 10])}
    uni = sm.build_universe(infos, pd.DataFrame())
    fs = sm.factor_scores(uni.loc["T0"].to_dict(), uni)
    ey = [p for m, v, p in fs["Valuation"]["det"] if m == "earnings_yield"][0]
    assert ey < 20                       # antes quedaba N/A (neutral)


def test_no_double_count_pe():
    metricas = [m for m, _ in sm.FACTORES["Valuation"]]
    assert "pe" not in metricas and "earnings_yield" in metricas


def test_rank_universe_and_backtest():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, .018, (1500, 40)), 0)), idx,
                          [f"T{i}" for i in range(40)])
    spx = prices.mean(axis=1)
    infos = {t: info(eps=rng.normal(5, 3), revenueGrowth=rng.normal(.1, .1)) for t in prices.columns}
    uni = sm.build_universe(infos, prices)
    rk = sm.rank_universe(uni)
    assert rk["Score"].notna().all() and rk["Momentum"].between(0, 100).all()
    f = sm.bt_features(prices)
    assert set(f) == set(sm.BT_FEATS)


def test_rank_universe_empty():
    assert sm.rank_universe(pd.DataFrame()).empty
