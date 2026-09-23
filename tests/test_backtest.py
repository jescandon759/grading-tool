import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import backtest as bt, research as rs
from test_core import synth


def env(planted, seed, n=150, days=1600):
    close, vol, sectors = synth(n, days, seed=seed, planted=planted)
    bench = close.mean(axis=1)
    mask_fn = lambda d: pd.DataFrame(True, index=pd.DatetimeIndex(d), columns=close.columns)
    return close, vol, bench, mask_fn, sectors


def test_lag_and_turnover():
    close, vol, bench, mask_fn, _ = env(0, 0, 60, 400)
    d = bt.rebalance_dates(close.index, every=21)
    s = pd.DataFrame(np.tile(np.arange(60), (len(d), 1)), index=d, columns=close.columns, dtype=float)
    r = bt.evaluate(s, close, bench, top_n=10, cost_bps=100)
    assert (r["entrada"] > r.index).all()                      # opera el dia siguiente
    assert r["Rotacion"].iloc[0] == 1.0                         # primera compra: 1x, no 2x
    assert r["Rotacion"].iloc[1:].max() < 0.5                   # mismo Top N: solo rebalanceo por deriva
    one = close.iloc[:, -10:]
    e0, e1 = close.index.get_loc(r["entrada"].iloc[0]), close.index.get_loc(r["salida"].iloc[0])
    exp = (one.iloc[e1] / one.iloc[e0] - 1).mean()
    assert abs(r["Top N"].iloc[0] - exp) < 1e-12


def test_membership_excludes_non_members():
    close, vol, bench, _, _ = env(0, 0, 60, 400)
    d = bt.rebalance_dates(close.index, every=21)
    mask = pd.DataFrame(True, index=d, columns=close.columns); mask.iloc[:, :30] = False
    feats = rs.features_a(close)
    sc = rs.score_a(feats, rs.CONFIGS_A[rs.PRINCIPAL_A], d, mask)
    assert sc.iloc[:, :30].isna().all().all()


def test_study_a_detects_signal_and_rejects_noise():
    t = time.time()
    close, vol, bench, mask_fn, sectors = env(0.6, 1)
    a = rs.study_a(close, bench, mask_fn, sectors, top_n=15)
    assert a["ablacion"].loc[rs.PRINCIPAL_A, "IC t-stat"] > 2
    assert not a["grid"].empty and len(a["check"]) == 10 and not a["wf"].empty
    close, vol, bench, mask_fn, sectors = env(0.0, 2)
    a0 = rs.study_a(close, bench, mask_fn, sectors, top_n=15)
    assert a0["veredicto"][0] != "Evidencia fuerte"
    print("A secs", time.time() - t)


def test_study_b_runs():
    close, vol, bench, mask_fn, sectors = env(0.6, 3, 120, 900)
    b = rs.study_b(close, vol, bench, mask_fn, sectors, top_n=15)
    assert set(b["por_estrategia"]) == set(rs.ESTR_B)
    m = b["por_estrategia"]["Momentum"]
    assert m["resumen"]["IC t-stat"] > 2 and not m["grid"].empty


def test_select_portfolio_buffer_and_cap():
    s = pd.Series({f"T{i}": 100 - i for i in range(30)})
    sec = pd.Series({f"T{i}": "A" if i < 10 else "BCD"[i % 3] for i in range(30)})
    assert bt.select_portfolio(s, 5) == ["T0", "T1", "T2", "T3", "T4"]
    # T8 ya estaba: con buffer 2x (Top 10) se queda aunque sea #9
    assert "T8" in bt.select_portfolio(s, 5, prev=["T8"], buffer=2.0)
    assert "T8" not in bt.select_portfolio(s, 5, prev=["T8"], buffer=1.0)
    capped = bt.select_portfolio(s, 10, sectors=sec, sector_cap=0.3)
    assert (sec[capped] == "A").sum() == 3 and len(capped) == 10


def test_regime_cash():
    close, vol, bench, mask_fn, _ = env(0, 0, 60, 500)
    d = bt.rebalance_dates(close.index, every=21)
    s = pd.DataFrame(np.tile(np.arange(60), (len(d), 1)), index=d, columns=close.columns, dtype=float)
    ro = pd.Series(False, index=d)
    r = bt.evaluate(s, close, bench, 10, 10, risk_on=ro)
    assert r["Efectivo"].all() and (r["Top N"] > 0).all() and (r["Top N"] < 0.01).all()
