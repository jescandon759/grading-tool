"""
Screener semanal quant + validacion walk-forward.

Flujo del screener (pensado para correrse el fin de semana, para operar la semana):
  1) Precios de ~500 acciones del S&P 500 + tu lista extra (fuera del indice).
  2) Senales de precio para TODAS (barato: 1 descarga).
  3) Pre-filtro: liquidez + las mejores N por senales de precio.
  4) Solo para esas N: fundamentales y fecha de reporte (caro: 1 llamada por ticker).
  5) Score por estrategia (momentum, reversion, multifactor, eventos) + compuesto.
  Salida: ranking. SIN tamano de posicion: la decision es tuya.

Backtest walk-forward: cada viernes calcula las senales con datos hasta ese dia,
elige el Top N y mide el rendimiento de la SIGUIENTE semana. Nunca usa datos futuros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import factors as fx

ESTRATEGIAS = ["Momentum", "Reversion", "Multifactor", "Eventos"]
PESOS_DEFAULT = {"Momentum": 0.30, "Reversion": 0.20, "Multifactor": 0.30, "Eventos": 0.20}


# ---------------------------------------------------------------- RANKING DE HOY
def liquidity_filter(close: pd.DataFrame, volume: pd.DataFrame | None, min_dollar_vol=5e6,
                     min_price=3.0) -> pd.Index:
    last = close.iloc[-1]
    ok = last >= min_price
    if volume is not None and not volume.empty:
        dv = (close * volume.reindex_like(close)).iloc[-20:].mean()
        ok &= dv.reindex(last.index).fillna(0) >= min_dollar_vol
    return last.index[ok.fillna(False)]


def price_prescreen(snap: pd.DataFrame, sectors: pd.Series, n: int = 80) -> list[str]:
    """Candidatas para estudio profundo: las mejores de CADA estrategia de precio
    (no solo del compuesto) para no perder a las de reversion o eventos."""
    mom = fx.score_momentum(snap, sectors)["Momentum"]
    rev = fx.score_reversion(snap, sectors)["Reversion"]
    ev = fx.score_events(snap)["Eventos"]
    lv = fx.score_lowvol(snap, sectors)
    k = max(n // 4, 5)
    picks = []
    for s in (mom, rev, ev, (mom + lv) / 2):
        picks += list(s.dropna().sort_values(ascending=False).index[:k])
    return list(dict.fromkeys(picks))[:n]


def weekly_ranking(snap: pd.DataFrame, sectors: pd.Series, fund: pd.DataFrame | None = None,
                   earnings: pd.Series | None = None, weights: dict | None = None,
                   today=None) -> pd.DataFrame:
    """Tabla final: un renglon por accion con score por estrategia, compuesto y desglose."""
    w = weights or PESOS_DEFAULT
    mom = fx.score_momentum(snap, sectors)
    rev = fx.score_reversion(snap, sectors)
    mf = fx.score_multifactor(snap, fund, sectors)
    ev = fx.score_events(snap, fund, earnings, today=today)
    out = pd.DataFrame({
        "Sector": sectors.reindex(snap.index),
        "Momentum": mom["Momentum"], "Reversion": rev["Reversion"],
        "Multifactor": mf["Multifactor"], "Eventos": ev["Eventos"],
    })
    # Compuesto: la reversion solo aplica a algunas; donde no aplica no castiga
    out["Score"] = fx.combine({k: out[k] for k in ESTRATEGIAS}, w, min_parts=2)
    # Estrategia dominante = la que mas empuja el score (para saber "por que" esta arriba)
    out["Senal principal"] = out[ESTRATEGIAS].idxmax(axis=1, skipna=True)
    extras = pd.DataFrame({
        "Rend 1 sem": snap["ret_1w"], "Rend 3m": snap["mom_3m"], "Mom 12-1": snap["mom_12_1"],
        "Volatilidad": snap["vol_60"], "RSI": snap["rsi_14"],
        "Tendencia": snap["trend_up"].map({1.0: "Alcista", 0.0: "No"}),
    })
    if "Reporta en (dias)" in ev:
        extras["Reporta en (dias)"] = ev["Reporta en (dias)"]
    detalle = pd.concat([mom.drop(columns="Momentum"), rev.drop(columns="Reversion"),
                         mf.drop(columns="Multifactor"),
                         ev.drop(columns=["Eventos", "Reporta en (dias)"], errors="ignore")],
                        axis=1)
    detalle = detalle.loc[:, ~detalle.columns.duplicated()]
    res = pd.concat([out, extras], axis=1).sort_values("Score", ascending=False)
    res.insert(0, "Rank", range(1, len(res) + 1))
    return res, detalle.reindex(res.index)


# ---------------------------------------------------------------- BACKTEST WALK-FORWARD
def _price_strategy_scores(snap, sectors):
    """Versiones solo-precio (unicas que se pueden validar sin sesgo de mirar al futuro:
    los fundamentales de Yahoo son los de HOY, no los que se conocian en cada fecha)."""
    mom = fx.score_momentum(snap, sectors)["Momentum"]
    rev = fx.score_reversion(snap, sectors)["Reversion"]
    lv = fx.score_lowvol(snap, sectors)
    ev = fx.score_events(snap)["Eventos"]
    comp = fx.combine({"m": mom, "r": rev, "l": lv, "e": ev},
                      {"m": .35, "r": .2, "l": .25, "e": .2}, min_parts=2)
    return {"Momentum": mom, "Reversion": rev, "Baja volatilidad": lv, "Eventos": ev,
            "Compuesto (precio)": comp}


def rebalance_dates(index: pd.DatetimeIndex, warmup: int = 260) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    fridays = s.groupby(idx.to_period("W-FRI")).last()   # ultimo dia habil de cada semana
    return [d for d in fridays if idx.get_loc(d) >= warmup]


def walk_forward(close: pd.DataFrame, volume: pd.DataFrame | None, sectors: pd.Series,
                 top_n: int = 20, cost_bps: float = 10.0, warmup: int = 260,
                 progress=None) -> dict:
    """Devuelve {estrategia: {"semanal": df, "resumen": dict}}."""
    sig = fx.price_signals(close, volume)
    fechas = rebalance_dates(close.index, warmup)
    if len(fechas) < 10:
        raise ValueError("Se necesitan mas datos: minimo ~1 anio de calentamiento + 10 semanas.")
    px = close.loc[fechas]
    fwd = px.shift(-1) / px - 1                       # rendimiento de la SIGUIENTE semana
    res = {k: [] for k in ["Momentum", "Reversion", "Baja volatilidad", "Eventos", "Compuesto (precio)"]}
    prev = {k: set() for k in res}
    for i, d in enumerate(fechas[:-1]):
        if progress:
            progress(i / (len(fechas) - 1))
        snap = pd.DataFrame({k: v.loc[d] for k, v in sig.items()})
        snap = snap[close.loc[d].notna()]
        f = fwd.loc[d].reindex(snap.index)
        univ = f.mean()
        for name, sc in _price_strategy_scores(snap, sectors).items():
            both = pd.concat([sc, f], axis=1, keys=["s", "f"]).dropna()
            if len(both) < max(30, top_n * 2):
                continue
            ic = both["s"].corr(both["f"], method="spearman")
            q = pd.qcut(both["s"].rank(method="first"), 5, labels=False)
            qret = both["f"].groupby(q).mean()
            top = set(both["s"].nlargest(top_n).index)
            turnover = 1.0 if not prev[name] else len(top - prev[name]) / top_n
            prev[name] = top
            gross = both.loc[list(top), "f"].mean()
            res[name].append({"fecha": d, "IC": ic, "Top N": gross,
                              "Top N neto": gross - turnover * 2 * cost_bps / 1e4,
                              "Universo": univ, "Rotacion": turnover,
                              **{f"Q{j + 1}": qret.get(j, np.nan) for j in range(5)}})
    out = {}
    for name, rows in res.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).set_index("fecha")
        out[name] = {"semanal": df, "resumen": summarize(df)}
    return out


def _max_dd(r: pd.Series) -> float:
    curve = (1 + r.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    ic = df["IC"]
    ann = lambda r: float((1 + r).prod() ** (52 / max(len(r), 1)) - 1)
    exc = df["Top N neto"] - df["Universo"]
    return {
        "Semanas": n,
        "IC promedio": float(ic.mean()),
        "IC t-stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(n)) if n > 2 and ic.std() > 0 else np.nan,
        "% semanas IC>0": float((ic > 0).mean()),
        "Spread Q5-Q1 semanal": float((df["Q5"] - df["Q1"]).mean()),
        "Top N anual (neto)": ann(df["Top N neto"]),
        "Universo anual": ann(df["Universo"]),
        "Exceso anual": ann(df["Top N neto"]) - ann(df["Universo"]),
        "% semanas le gana al universo": float((exc > 0).mean()),
        "Max caida Top N": _max_dd(df["Top N neto"]),
        "Rotacion semanal": float(df["Rotacion"].mean()),
    }


def verdict(r: dict) -> tuple[str, str]:
    """Traduce el resumen a una etiqueta honesta."""
    t, ic, exc = r["IC t-stat"], r["IC promedio"], r["Exceso anual"]
    if np.isnan(t):
        return "Sin datos", "gray"
    if t >= 2 and exc > 0:
        return "Senal con evidencia", "green"
    if t >= 1 and ic > 0:
        return "Senal debil", "orange"
    if t <= -2:
        return "Funciona AL REVES", "red"
    return "Sin evidencia (ruido)", "red"
