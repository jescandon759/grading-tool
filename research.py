"""
Estudios de backtesting A y B sobre el MISMO pipeline (backtest.py):
mismo universo historico point-in-time, mismo rezago de ejecucion, mismo modelo de costos
y mismo benchmark con dividendos (SPY).

A) Momentum mensual del Investment Score   -> "¿puedo construir un portafolio con este ranking?"
B) Senales del screener semanal            -> "¿el score contiene informacion sobre el futuro?"
Ambos reportan las dos caras (factorial + portafolio), fuera de muestra y robustez.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import factors as fx
import score_model as sm
import screener as sc

# ---------------------------------------------------------------- A: MODELOS (ablacion)
CONFIGS_A = {
    "M1 Solo momentum (3/6/12m)": {"ret_3m": 1.0, "ret_6m": 1.5, "ret_12m": 1.5},
    "M2 Momentum + tendencia": {"ret_3m": 1.0, "ret_6m": 1.5, "ret_12m": 1.5, "dist_sma50": 1.0, "dist_sma200": 1.0},
    "M3 Momentum + tendencia + volatilidad": {"ret_3m": 1.0, "ret_6m": 1.5, "ret_12m": 1.5, "dist_sma50": 1.0,
                                              "dist_sma200": 1.0, "mom_vol": 0.8},
    "App actual (7 señales)": dict(sm.MOM_PESOS),
    "Momentum clásico 12-1": {"mom_12_1": 1.0},
}
PRINCIPAL_A = "App actual (7 señales)"
CLASICO = "Momentum clásico 12-1"
DIRS = {**sm.MOM_DIRS, "mom_12_1": 1}


def features_a(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    f = sm.bt_features(close)
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    return f


def score_a(feats, weights, dates, mask, sectors=None) -> pd.DataFrame:
    """Percentil de cada senal SOLO entre miembros de esa fecha; promedio ponderado.
    sectors != None -> percentil dentro del sector (version sector-neutral)."""
    num, den = None, 0.0
    elig = mask.loc[dates]
    for k, w in weights.items():
        x = feats[k].loc[dates].reindex(columns=elig.columns).where(elig)
        if sectors is None:
            r = x.rank(axis=1, pct=True) * 100
        else:
            r = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
            sec = sectors.reindex(x.columns).fillna("Otro")
            for g in sec.unique():
                cols = sec.index[sec == g]
                r[cols] = x[cols].rank(axis=1, pct=True) * 100
        r = (r if DIRS[k] > 0 else 100 - r).where(elig).fillna(50).where(elig)
        num = r * w if num is None else num + r * w
        den += w
    return num / den


def study_a(close, bench, mask_fn, sectors, top_n=10, cost_bps=10.0, every=21, grid=True):
    feats = features_a(close)
    dates = bt.rebalance_dates(close.index, every=every)
    mask = mask_fn(dates)
    ppy = 252 / every
    res = {n: bt.evaluate(score_a(feats, w, dates, mask), close, bench, top_n, cost_bps,
                          keep_tops=(n == PRINCIPAL_A)) for n, w in CONFIGS_A.items()}
    main = res[PRINCIPAL_A]
    out = {"ppy": ppy, "res": res, "main": main,
           "ablacion": pd.DataFrame({n: bt.summarize(r, ppy) for n, r in res.items()}).T}
    out["wf"], oos = bt.nested_walk_forward(res, ppy)
    out["oos"] = bt.summarize(oos, ppy) if not oos.empty else {}
    out["oos_exc"] = out["oos"].get("Exceso vs universo", np.nan)
    out["yearly"] = bt.by_year(main)
    out["alpha_mkt"] = bt.alpha_regression(main, ppy)
    out["alpha"] = bt.alpha_regression(main, ppy, {"Momentum clásico": res[CLASICO]["Top N neto"]})
    out["conc_sector"] = bt.sector_concentration(main, sectors)
    neutral = bt.evaluate(score_a(feats, CONFIGS_A[PRINCIPAL_A], dates, mask, sectors), close, bench, top_n, cost_bps)
    out["neutral"] = bt.summarize(neutral, ppy)
    out["corr"] = bt.signal_correlation({k: feats[k] for k in list(sm.MOM_PESOS) + ["mom_12_1"]}, dates, mask)
    out["costos"] = pd.Series({c: bt.summarize(bt.evaluate(score_a(feats, CONFIGS_A[PRINCIPAL_A], dates, mask),
                                                           close, bench, top_n, c), ppy)["Exceso vs universo"]
                               for c in (0, 10, 25, 50)}, name="Exceso vs universo")
    if grid:
        g = {}
        for ev in (15, 21, 30):
            d2 = bt.rebalance_dates(close.index, every=ev); m2 = mask_fn(d2)
            s2 = score_a(feats, CONFIGS_A[PRINCIPAL_A], d2, m2)
            for n in (5, 10, 15, 20, 30):
                g[(ev, n)] = bt.summarize(bt.evaluate(s2, close, bench, n, cost_bps), 252 / ev)["Exceso vs universo"]
        out["grid"] = pd.Series(g).unstack()
        out["grid"].index.name, out["grid"].columns.name = "Rebalanceo (días)", "Top N"
        out["grid_pos"] = float((out["grid"] > 0).mean().mean())
    else:
        out["grid"], out["grid_pos"] = pd.DataFrame(), np.nan
    out["construccion"] = construction_table(score_a(feats, CONFIGS_A[PRINCIPAL_A], dates, mask), close, bench,
                                             top_n, cost_bps, sectors, ppy)
    out["t_crit"] = bt.t_critico(len(CONFIGS_A))
    out["check"] = bt.checklist(bt.summarize(main, ppy), out["yearly"], out["oos_exc"], out["grid_pos"],
                                out["alpha"], out["t_crit"])
    out["veredicto"] = bt.veredicto(out["check"])
    return out


def construction_table(scores, close, bench, top_n, cost_bps, sectors, ppy) -> pd.DataFrame:
    """Mismo score, distintas reglas de portafolio: ¿cuanto aporta cada control?"""
    ro = bt.risk_on_signal(bench)
    variantes = {
        "Base (Top N simple)": {},
        "Buffer de rotación (sale fuera del Top 2N)": {"buffer": 2.0},
        "Tope sectorial 30%": {"sectors": sectors, "sector_cap": 0.3},
        "Filtro de régimen (efectivo si SPY < media 200d)": {"risk_on": ro},
        "Buffer + tope sectorial": {"buffer": 2.0, "sectors": sectors, "sector_cap": 0.3},
        "Todo junto": {"buffer": 2.0, "sectors": sectors, "sector_cap": 0.3, "risk_on": ro},
    }
    filas = {}
    for n, kw in variantes.items():
        r = bt.evaluate(scores, close, bench, top_n, cost_bps, **kw)
        s = bt.summarize(r, ppy)
        filas[n] = {k: s.get(k) for k in ["Top N anual (neto)", "Exceso vs universo", "Exceso vs SPY",
                                           "Sharpe (neto)", "Max caída", "Rotación por periodo"]}
        filas[n]["% tiempo en efectivo"] = float(r["Efectivo"].mean()) if not r.empty else np.nan
    return pd.DataFrame(filas).T


# ---------------------------------------------------------------- B: ESTRATEGIAS DEL SCREENER
ESTR_B = ["Momentum", "Reversion", "Baja volatilidad", "Eventos", "Compuesto (precio)"]


def scores_b(close, volume, dates, mask, sectors, progress=None) -> dict[str, pd.DataFrame]:
    sig = fx.price_signals(close, volume)
    out = {k: {} for k in ESTR_B + [CLASICO]}
    for i, d in enumerate(dates):
        if progress:
            progress(i / len(dates))
        el = mask.loc[d]
        snap = pd.DataFrame({k: v.loc[d] for k, v in sig.items()})
        snap = snap[el.reindex(snap.index).fillna(False) & close.loc[d].notna()]
        if len(snap) < 30:
            continue
        for k, v in sc._price_strategy_scores(snap, sectors).items():
            out[k][d] = v
        out[CLASICO][d] = fx.pct_rank(snap["mom_12_1"])
    return {k: pd.DataFrame(v).T.sort_index() for k, v in out.items()}


def study_b(close, volume, bench, mask_fn, sectors, top_n=20, cost_bps=10.0, grid=True, progress=None):
    dates = bt.rebalance_dates(close.index, weekly=True)
    mask = mask_fn(dates)
    S = scores_b(close, volume, dates, mask, sectors, progress)
    ppy = 52
    res = {k: bt.evaluate(S[k], close, bench, top_n, cost_bps, keep_tops=True) for k in S}
    out = {"ppy": ppy, "res": res, "por_estrategia": {}}
    out["wf"], oos = bt.nested_walk_forward({k: res[k] for k in ESTR_B}, ppy)
    out["oos_sistema"] = bt.summarize(oos, ppy) if not oos.empty else {}
    t_crit = bt.t_critico(len(ESTR_B))
    for k in ESTR_B:
        r = res[k]
        if r.empty:
            continue
        mitad = r.iloc[len(r) // 2:]
        oos_exc = bt._ann(mitad["Top N neto"], ppy) - bt._ann(mitad["Universo"], ppy)
        e = {"resumen": bt.summarize(r, ppy), "yearly": bt.by_year(r), "oos_exc": oos_exc,
             "alpha": bt.alpha_regression(r, ppy, {"Momentum clásico": res[CLASICO]["Top N neto"]}),
             "conc_sector": bt.sector_concentration(r, sectors)}
        if grid:
            g = {}
            for freq in (1, 2):
                s2 = S[k].iloc[::freq]
                for n in (10, 20, 30, 50):
                    g[("Semanal" if freq == 1 else "Quincenal", n)] = bt.summarize(
                        bt.evaluate(s2, close, bench, n, cost_bps), ppy / freq).get("Exceso vs universo", np.nan)
            e["grid"] = pd.Series(g).unstack()
            e["grid"].index.name, e["grid"].columns.name = "Frecuencia", "Top N"
            e["grid_pos"] = float((e["grid"] > 0).mean().mean())
        else:
            e["grid"], e["grid_pos"] = pd.DataFrame(), np.nan
        e["check"] = bt.checklist(e["resumen"], e["yearly"], oos_exc, e["grid_pos"], e["alpha"], t_crit)
        e["veredicto"] = bt.veredicto(e["check"])
        out["por_estrategia"][k] = e
    out["construccion"] = construction_table(S["Compuesto (precio)"], close, bench, top_n, cost_bps, sectors, ppy)
    out["t_crit"] = t_crit
    return out
