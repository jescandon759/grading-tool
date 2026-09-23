"""
Motor de backtesting unificado (lo usan A y B con el MISMO pipeline):

    Datos historicos -> Senales -> Score -> Test factorial (IC, quintiles)
                                         -> Test de portafolio (Top N, costos)
                     -> Fuera de muestra (walk-forward anidado) -> Robustez -> Veredicto

Reglas anti-trampa:
- Universo point-in-time: en cada fecha solo cuentan las empresas que ERAN miembros del S&P 500.
- Ejecucion con rezago: la senal se calcula al cierre del dia D y se opera al cierre de D+1.
- Benchmark con dividendos (SPY ajustado), igual que las acciones.
- Costos sobre la rotacion real: sum|peso nuevo - peso anterior ya movido por el mercado| x costo.
- Correccion por pruebas multiples: el t-stat exigido sube con el numero de configuraciones probadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RF = 0.04  # tasa libre de riesgo anual para Sharpe y alpha


# ---------------------------------------------------------------- FECHAS
def rebalance_dates(index: pd.DatetimeIndex, every: int | None = None, weekly: bool = False,
                    warmup: int = 260) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if weekly:
        s = pd.Series(idx, index=idx)
        d = s.groupby(idx.to_period("W-FRI")).last().values
        d = pd.DatetimeIndex(d)
        return d[[idx.get_loc(x) >= warmup for x in d]]
    return idx[warmup::every]


# ---------------------------------------------------------------- EVALUACION
def select_portfolio(scores: pd.Series, top_n: int, prev: list | None = None, buffer: float = 1.0,
                     sectors: pd.Series | None = None, sector_cap: float | None = None) -> list:
    """Construccion del portafolio (misma regla en backtest, paper trading y correo):
    - buffer: una accion que ya tienes se queda mientras siga dentro del Top (buffer x N). Baja rotacion.
    - sector_cap: maximo % de posiciones en un mismo sector (p.ej. 0.3)."""
    s = scores.dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    max_sec = int(np.ceil(sector_cap * top_n)) if sector_cap else top_n
    sec = (sectors.reindex(s.index).fillna("Otro") if sectors is not None else pd.Series("x", index=s.index))
    keep = [t for t in (prev or []) if t in rank.index and rank[t] <= buffer * top_n]
    keep = sorted(keep, key=lambda t: rank[t])
    out, cnt = [], {}
    for t in keep + [t for t in s.index if t not in keep]:
        if len(out) >= top_n:
            break
        g = sec[t]
        if cnt.get(g, 0) >= max_sec:
            continue
        out.append(t); cnt[g] = cnt.get(g, 0) + 1
    if len(out) < top_n:                      # si el tope no deja llenar el portafolio, se relaja
        out += [t for t in s.index if t not in out][:top_n - len(out)]
    return out


def evaluate(scores: pd.DataFrame, close: pd.DataFrame, bench: pd.Series, top_n: int = 10,
             cost_bps: float = 10.0, lag: int = 1, keep_tops: bool = False, buffer: float = 1.0,
             sectors: pd.Series | None = None, sector_cap: float | None = None,
             risk_on: pd.Series | None = None) -> pd.DataFrame:
    """scores: fechas-de-senal x tickers (NaN = no elegible). Devuelve una fila por periodo.
    risk_on: Serie booleana por fecha de senal; False = filtro de regimen manda a efectivo."""
    idx = close.index
    close_ff = close.ffill()                     # si una accion deja de cotizar, sale a su ultimo precio
    pos = idx.get_indexer(scores.index)
    ent = pos + lag
    ok = (pos >= 0) & (ent < len(idx))
    scores, ent = scores[ok], ent[ok]
    rows, w_prev, r_prev, prev = [], None, None, None
    for i in range(len(ent) - 1):
        e0, e1 = ent[i], ent[i + 1]
        s = scores.iloc[i]
        p0 = close.iloc[e0]
        s = s[s.notna() & p0.reindex(s.index).notna()]
        if len(s) < max(20, 2 * top_n):
            continue
        f = (close_ff.iloc[e1][s.index] / p0[s.index] - 1).astype(float)
        both = pd.DataFrame({"s": s, "f": f}).dropna()
        if len(both) < max(20, 2 * top_n):
            continue
        ic = both["s"].corr(both["f"], method="spearman")
        q = pd.qcut(both["s"].rank(method="first"), 5, labels=False)
        qret = both["f"].groupby(q).mean()
        dias = (idx[e1] - idx[e0]).days
        cash = risk_on is not None and not bool(risk_on.get(scores.index[i], True))
        if cash:
            top = []
            w_new = pd.Series(dtype=float)
            gross = RF * dias / 365
        else:
            top = select_portfolio(both["s"], top_n, prev, buffer, sectors, sector_cap)
            w_new = pd.Series(1 / len(top), index=top)
            gross = both.loc[top, "f"].mean()
        if w_prev is None:
            turnover = w_new.sum()                           # primera compra: solo entra
        else:
            drift = w_prev * (1 + r_prev.reindex(w_prev.index).fillna(0))
            drift = drift / drift.sum() if drift.sum() > 0 else drift
            turnover = w_new.sub(drift, fill_value=0).abs().sum()   # compras + ventas
        rows.append({
            "fecha": scores.index[i], "entrada": idx[e0], "salida": idx[e1],
            "IC": ic, "Top N": gross, "Top N neto": gross - turnover * cost_bps / 1e4,
            "Universo": both["f"].mean(), "Benchmark": bench.iloc[e1] / bench.iloc[e0] - 1,
            "Rotacion": turnover, "n": len(both), "Efectivo": cash,
            **{f"Q{j + 1}": qret.get(j, np.nan) for j in range(5)},
            **({"tops": list(top)} if keep_tops else {}),
        })
        w_prev = w_new if len(w_new) else None
        r_prev = both.loc[top, "f"] if len(top) else None
        prev = top
    return pd.DataFrame(rows).set_index("fecha") if rows else pd.DataFrame()


def risk_on_signal(bench: pd.Series, ma: int = 200) -> pd.Series:
    """Filtro de regimen: True si el SPY esta arriba de su media de 200 dias (tendencia sana)."""
    return bench > bench.rolling(ma, min_periods=int(ma * .9)).mean()


# ---------------------------------------------------------------- METRICAS
def _ann(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (ppy / len(r)) - 1) if len(r) else np.nan


def _mdd(r: pd.Series) -> float:
    c = (1 + r.fillna(0)).cumprod()
    return float((c / c.cummax() - 1).min())


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def summarize(df: pd.DataFrame, ppy: float) -> dict:
    if df.empty:
        return {}
    spread = df["Q5"] - df["Q1"]
    qm = df[[f"Q{i}" for i in range(1, 6)]].mean()
    vol = df["Top N neto"].std() * np.sqrt(ppy)
    top_a = _ann(df["Top N neto"], ppy)
    return {
        "Periodos": len(df),
        "IC promedio": float(df["IC"].mean()), "IC t-stat": t_stat(df["IC"]),
        "% periodos IC>0": float((df["IC"] > 0).mean()),
        "Spread Q5-Q1 anual": float(spread.mean() * ppy),
        "Monotonía quintiles": float(stats.spearmanr(range(5), qm.values)[0]),
        "Top N anual (neto)": top_a, "Universo anual": _ann(df["Universo"], ppy),
        "SPY anual": _ann(df["Benchmark"], ppy),
        "Exceso vs universo": top_a - _ann(df["Universo"], ppy),
        "Exceso vs SPY": top_a - _ann(df["Benchmark"], ppy),
        "Sharpe (neto)": (top_a - RF) / vol if vol > 0 else np.nan,
        "Max caída": _mdd(df["Top N neto"]),
        "Rotación por periodo": float(df["Rotacion"].mean()),
        "% periodos gana al universo": float((df["Top N neto"] > df["Universo"]).mean()),
    }


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.year)
    return pd.DataFrame({
        "Exceso vs universo": g.apply(lambda x: (1 + x["Top N neto"]).prod() - (1 + x["Universo"]).prod()),
        "Spread Q5-Q1": g.apply(lambda x: (x["Q5"] - x["Q1"]).sum()),
        "IC promedio": g["IC"].mean(),
    })


def alpha_regression(df: pd.DataFrame, ppy: float, controls: dict[str, pd.Series] | None = None) -> dict:
    """OLS: (Top N neto - rf) = a + b1*(SPY - rf) + b2*control2 + ...  -> alpha anual y su t-stat.
    Responde: ¿queda rendimiento propio despues de quitar mercado (y momentum clasico)?"""
    rf = RF / ppy
    y = df["Top N neto"] - rf
    X = pd.DataFrame({"Mercado (SPY)": df["Benchmark"] - rf})
    for k, v in (controls or {}).items():
        X[k] = v.reindex(df.index) - rf
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < 12:
        return {}
    Xm = np.column_stack([np.ones(len(data)), data.drop(columns="y").values])
    beta, *_ = np.linalg.lstsq(Xm, data["y"].values, rcond=None)
    resid = data["y"].values - Xm @ beta
    s2 = resid @ resid / (len(data) - Xm.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xm.T @ Xm)))
    out = {"Alpha anual": float(beta[0] * ppy), "Alpha t-stat": float(beta[0] / se[0]) if se[0] > 0 else np.nan}
    for i, k in enumerate(data.drop(columns="y").columns, start=1):
        out[f"Beta {k}"] = float(beta[i])
    return out


def sector_concentration(df: pd.DataFrame, sectors: pd.Series) -> float:
    """Promedio del % del Top N que cae en su sector mas repetido."""
    if "tops" not in df:
        return np.nan
    vals = []
    for tops in df["tops"]:
        if not tops:
            continue
        s = sectors.reindex(tops).fillna("Otro")
        vals.append(s.value_counts(normalize=True).iloc[0])
    return float(np.mean(vals))


def signal_correlation(features: dict[str, pd.DataFrame], dates, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Correlacion promedio (Spearman, entre acciones) de las senales en las fechas de rebalanceo."""
    mats = []
    for d in list(dates)[::3]:
        m = pd.DataFrame({k: v.loc[d] for k, v in features.items()})
        if mask is not None:
            m = m[mask.loc[d].reindex(m.index).fillna(False)]
        m = m.dropna()
        if len(m) > 30:
            mats.append(m.rank().corr())
    return sum(mats) / len(mats) if mats else pd.DataFrame()


# ---------------------------------------------------------------- FUERA DE MUESTRA
def nested_walk_forward(results: dict[str, pd.DataFrame], ppy: float, train_years: int = 5,
                        min_train_years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seleccion de modelo walk-forward: cada anio de prueba elige el modelo con mejor exceso neto
    en los `train_years` anios previos y lo evalua en el anio siguiente (que no vio).
    Devuelve (tabla por anio, periodos fuera de muestra concatenados)."""
    years = sorted(set().union(*[set(r.index.year) for r in results.values() if not r.empty]))
    filas, oos = [], []
    for y in years:
        train = [a for a in years if y - train_years <= a < y]
        if len(train) < min_train_years:
            continue
        mejor, mejor_exc = None, -np.inf
        for name, r in results.items():
            tr = r[r.index.year.isin(train)]
            if len(tr) < 6:
                continue
            exc = _ann(tr["Top N neto"], ppy) - _ann(tr["Universo"], ppy)
            if exc > mejor_exc:
                mejor, mejor_exc = name, exc
        if mejor is None:
            continue
        te = results[mejor][results[mejor].index.year == y]
        if te.empty:
            continue
        oos.append(te)
        filas.append({"Año de prueba": y, "Entrenado con": f"{train[0]}–{train[-1]}",
                      "Modelo elegido": mejor, "Exceso en entrenamiento": mejor_exc,
                      "Exceso fuera de muestra": float((1 + te["Top N neto"]).prod() - (1 + te["Universo"]).prod())})
    return pd.DataFrame(filas), (pd.concat(oos) if oos else pd.DataFrame())


def t_critico(n_pruebas: int, alpha: float = 0.05) -> float:
    """t exigido tras corregir por pruebas multiples (Bonferroni, dos colas)."""
    return float(stats.norm.ppf(1 - alpha / (2 * max(n_pruebas, 1))))


# ---------------------------------------------------------------- VEREDICTO
def checklist(res: dict, yearly: pd.DataFrame, oos_exc: float, grid_pos: float, alpha: dict,
              t_crit: float) -> list[tuple[str, str, bool]]:
    """Todas las condiciones a la vez; ninguna por si sola basta."""
    pct = lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "n/d"
    yr_spread = float((yearly["Spread Q5-Q1"] > 0).mean()) if not yearly.empty else np.nan
    yr_exc = float((yearly["Exceso vs universo"] > 0).mean()) if not yearly.empty else np.nan
    a_t = alpha.get("Alpha t-stat", np.nan)
    return [
        ("IC positivo", f"{res['IC promedio']:.3f}", res["IC promedio"] > 0),
        (f"IC significativo (t ≥ {t_crit:.2f}, corregido por pruebas múltiples)", f"t = {res['IC t-stat']:.2f}",
         pd.notna(res["IC t-stat"]) and res["IC t-stat"] >= t_crit),
        ("Q5 rinde más que Q1", pct(res["Spread Q5-Q1 anual"]) + " anual", res["Spread Q5-Q1 anual"] > 0),
        ("Quintiles ordenados (Q1<Q2<…<Q5)", f"monotonía {res['Monotonía quintiles']:.2f}", res["Monotonía quintiles"] >= 0.8),
        ("Spread Q5-Q1 positivo en ≥60% de los años", f"{yr_spread:.0%}" if pd.notna(yr_spread) else "n/d",
         pd.notna(yr_spread) and yr_spread >= 0.6),
        ("Gana al universo después de costos", pct(res["Exceso vs universo"]) + " anual", res["Exceso vs universo"] > 0),
        ("Fuera de muestra positivo", pct(oos_exc) + " anual", pd.notna(oos_exc) and oos_exc > 0),
        ("Robustez temporal: gana en ≥60% de los años", f"{yr_exc:.0%}" if pd.notna(yr_exc) else "n/d",
         pd.notna(yr_exc) and yr_exc >= 0.6),
        ("Robustez de parámetros: ≥70% de variantes ganan", f"{grid_pos:.0%}" if pd.notna(grid_pos) else "n/d",
         pd.notna(grid_pos) and grid_pos >= 0.7),
        ("Alpha propio tras quitar mercado y momentum clásico", f"{pct(alpha.get('Alpha anual'))} (t = {a_t:.2f})"
         if pd.notna(a_t) else "n/d", pd.notna(a_t) and alpha.get("Alpha anual", 0) > 0),
    ]


def veredicto(items: list[tuple[str, str, bool]]) -> tuple[str, str]:
    n = sum(ok for *_, ok in items); tot = len(items)
    if n == tot:
        return "Evidencia fuerte", "green"
    if n >= tot - 2 and items[1][2] and items[6][2]:      # exige significancia y fuera de muestra
        return "Evidencia moderada", "orange"
    return "Sin evidencia suficiente", "red"
