"""Pruebas con datos sinteticos (no requieren internet). Ejecuta: python -m pytest tests"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import data, factors as fx, screener as sc


def synth(n_tk=120, days=800, seed=0, planted=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=days)
    drift = rng.normal(0, 0.0006, n_tk)
    rets = rng.normal(0, 0.018, (days, n_tk))
    if planted:  # momentum plantado: el rendimiento pasado de 60 dias predice el siguiente
        for t in range(61, days):
            past = rets[t-60:t-5].sum(0)
            rets[t] += planted * (past - past.mean()) / 40
    rets += drift
    tk = [f"T{i:03d}" for i in range(n_tk)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, 0)), idx, tk)
    vol = pd.DataFrame(rng.lognormal(14, .3, (days, n_tk)), idx, tk)
    sectors = pd.Series([f"S{i % 6}" for i in range(n_tk)], tk)
    return close, vol, sectors


def test_sp500_list():
    u = data.sp500_universe()
    assert len(u) > 490 and "BRK-B" in set(u["Ticker"]) and u["Sector"].nunique() == 11


def test_parse_tickers():
    assert data.parse_tickers("aapl, msft; brk.b\nwalmex.mx aapl") == ["AAPL", "MSFT", "BRK-B", "WALMEX.MX"]


def test_download_retries_and_report():
    close, vol, _ = synth(10, 300)
    calls = {"n": 0}
    def fake(tks, period):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("rate limit")
        tks = [t for t in tks if t != "T009"]  # T009 nunca existe
        return pd.concat({"Close": close[tks], "Volume": vol[tks]}, axis=1)
    c, v, rep = data.download_prices(list(close.columns), chunk=50, pause=0, downloader=fake)
    assert rep["fallidos"] == ["T009"] and len(rep["ok"]) == 9 and c.shape[1] == 9


def test_value_uses_yields_not_inverted_pe():
    f = pd.DataFrame({"earn_yield": [-0.10, 0.02, 0.08, 0.05, 0.03, 0.06],
                      "book_yield": [0.1] * 6, "fcf_yield": [0.03] * 6})
    v = fx.score_value(f)["Valor"]
    assert v.idxmin() == 0   # la empresa con perdidas queda al fondo, no arriba


def test_combine_renormalizes():
    a = pd.Series([100, 0, np.nan]); b = pd.Series([0, np.nan, np.nan])
    out = fx.combine({"a": a, "b": b}, min_parts=1)
    assert out[0] == 50 and out[1] == 0 and np.isnan(out[2])


def test_weekly_ranking_shapes():
    close, vol, sectors = synth()
    sig = fx.price_signals(close, vol)
    snap = fx.snapshot(sig)
    rk, det = sc.weekly_ranking(snap, sectors)
    assert rk["Score"].between(0, 100).all() and rk["Rank"].iloc[0] == 1
    assert set(sc.ESTRATEGIAS) <= set(rk.columns) and len(det) == len(rk)


def test_no_lookahead():
    """Cambiar precios FUTUROS no debe cambiar las senales de una fecha pasada."""
    close, vol, _ = synth(30, 400)
    d = close.index[300]
    a = fx.snapshot(fx.price_signals(close, vol), d)
    c2 = close.copy(); c2.iloc[301:] *= 3
    b = fx.snapshot(fx.price_signals(c2, vol), d)
    pd.testing.assert_frame_equal(a, b)


def test_backtest_detects_planted_signal_and_rejects_noise():
    close, vol, sectors = synth(150, 700, seed=1, planted=0.6)
    r = sc.walk_forward(close, vol, sectors, top_n=15)
    assert r["Momentum"]["resumen"]["IC t-stat"] > 2
    close, vol, sectors = synth(150, 700, seed=2, planted=0.0)
    r = sc.walk_forward(close, vol, sectors, top_n=15)
    assert abs(r["Momentum"]["resumen"]["IC t-stat"]) < 3

