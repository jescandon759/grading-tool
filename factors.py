"""
Factores y scoring (sin Streamlit, sin red: todo es calculo puro y testeable).

Principios del modelo v2:
1. Winsorizar (recortar extremos) antes de comparar: un dato raro no domina el score.
2. Percentiles (0-100) en vez de z-scores: robustos a outliers y faciles de leer.
3. Comparar contra el SECTOR cuando el grupo es suficientemente grande (sector-neutral);
   si no, contra todo el universo. Asi un banco no se compara con una tecnologica.
4. Valor con YIELDS (utilidad/precio, libros/precio, FCF/precio), no con multiplos:
   el P/E invertido hacia que empresas con perdidas parecieran "baratas".
5. Todo score se descompone por factor para que se vea de donde sale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- HERRAMIENTAS
def winsorize(s: pd.Series, p: float = 0.02) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 5:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentil 0-100 (NaN se queda NaN)."""
    s = winsorize(s)
    r = s.rank(pct=True, ascending=higher_is_better)
    return r * 100


def group_rank(s: pd.Series, groups: pd.Series | None, higher_is_better=True, min_group=6):
    """Percentil dentro del grupo (sector) si el grupo tiene >= min_group datos,
    si no, percentil contra todo el universo."""
    glob = pct_rank(s, higher_is_better)
    if groups is None:
        return glob
    groups = groups.reindex(s.index)
    out = glob.copy()
    for g, idx in s.groupby(groups).groups.items():
        sub = s.loc[idx]
        if sub.notna().sum() >= min_group:
            out.loc[idx] = pct_rank(sub, higher_is_better)
    return out


def combine(parts: dict[str, pd.Series], weights: dict[str, float] | None = None,
            min_parts: int = 1) -> pd.Series:
    """Promedio ponderado de percentiles ignorando NaN (re-normaliza pesos).
    Si hay menos de `min_parts` componentes con dato -> NaN."""
    df = pd.DataFrame(parts)
    w = pd.Series(weights or {k: 1.0 for k in parts}).reindex(df.columns).fillna(0)
    mask = df.notna()
    num = (df.fillna(0) * w).sum(axis=1)
    den = (mask * w).sum(axis=1)
    out = num / den.replace(0, np.nan)
    out[mask.sum(axis=1) < min_parts] = np.nan
    return out


# ---------------------------------------------------------------- SENALES DE PRECIO
def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def price_signals(close: pd.DataFrame, volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Panel de senales (fechas x tickers). Cada valor usa SOLO datos hasta esa fecha,
    por eso sirve tanto para el ranking de hoy como para el backtest walk-forward."""
    close = close.astype(float)
    r1 = close.pct_change(fill_method=None)
    ma20, ma50, ma200 = (close.rolling(n, min_periods=int(n * .8)).mean() for n in (20, 50, 200))
    sd20 = close.rolling(20, min_periods=16).std()
    hi252 = close.rolling(252, min_periods=200).max()
    sig = {
        # Momentum clasico 12-1: rendimiento de 12 meses saltando el ultimo mes
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "mom_6_1": close.shift(21) / close.shift(126) - 1,
        "mom_3m": close / close.shift(63) - 1,
        "ret_1w": close / close.shift(5) - 1,
        "vol_60": r1.rolling(60, min_periods=45).std() * np.sqrt(TRADING_DAYS),
        "rsi_14": rsi(close, 14),
        "z_ma20": (close - ma20) / sd20,
        "near_high": close / hi252,           # 1.0 = en maximo de 52 semanas
        "trend_up": ((close > ma200) & (ma50 > ma200)).astype(float).where(ma200.notna()),
    }
    if volume is not None and not volume.empty:
        v = volume.reindex_like(close).astype(float)
        sig["vol_surge"] = v.rolling(5, min_periods=4).mean() / v.rolling(60, min_periods=45).mean()
        # gap del dia: movimiento fuerte con volumen = evento
        sig["gap_1d"] = (close / close.shift(1) - 1).abs()
    return sig


def snapshot(sig: dict[str, pd.DataFrame], when=None) -> pd.DataFrame:
    """Toma la fila de cada senal en la fecha `when` (default: ultima)."""
    out = {}
    for k, df in sig.items():
        row = df.iloc[-1] if when is None else df.loc[:when].iloc[-1]
        out[k] = row
    return pd.DataFrame(out)


# ---------------------------------------------------------------- ESTRATEGIAS (cross-section)
def score_momentum(s: pd.DataFrame, sectors=None) -> pd.DataFrame:
    parts = {
        "Mom 12-1": group_rank(s["mom_12_1"], sectors),
        "Mom 6-1": group_rank(s["mom_6_1"], sectors),
        "Cerca de max 52s": pct_rank(s["near_high"]),
    }
    sc = combine(parts, {"Mom 12-1": .45, "Mom 6-1": .30, "Cerca de max 52s": .25}, min_parts=2)
    # Momentum en tendencia bajista es una trampa comun: penaliza
    sc = sc - 15 * (1 - s["trend_up"].fillna(0))
    return pd.DataFrame(parts).assign(Momentum=sc.clip(0, 100))


def score_reversion(s: pd.DataFrame, sectors=None) -> pd.DataFrame:
    """Reversion a la media de corto plazo: caidas fuertes de 1 semana DENTRO de una
    tendencia alcista (compra el retroceso, no el cuchillo que cae)."""
    parts = {
        "Caida 1 sem": pct_rank(s["ret_1w"], higher_is_better=False),
        "Bajo su media 20d": pct_rank(s["z_ma20"], higher_is_better=False),
        "RSI bajo": pct_rank(s["rsi_14"], higher_is_better=False),
    }
    sc = combine(parts, min_parts=2)
    sc = sc.where(s["trend_up"] == 1)          # solo aplica en tendencia alcista
    return pd.DataFrame(parts).assign(Reversion=sc)


def score_lowvol(s: pd.DataFrame, sectors=None) -> pd.Series:
    return group_rank(s["vol_60"], sectors, higher_is_better=False)


def score_quality(f: pd.DataFrame, sectors=None) -> pd.DataFrame:
    parts = {
        "ROE": group_rank(f.get("roe"), sectors),
        "Margen bruto": group_rank(f.get("gross"), sectors),
        "Margen operativo": group_rank(f.get("oper"), sectors),
        "Poca deuda": group_rank(f.get("de"), sectors, higher_is_better=False),
    }
    return pd.DataFrame(parts).assign(Calidad=combine(parts, min_parts=2))


def score_value(f: pd.DataFrame, sectors=None) -> pd.DataFrame:
    parts = {
        "Utilidad/Precio": group_rank(f.get("earn_yield"), sectors),
        "Libros/Precio": group_rank(f.get("book_yield"), sectors),
        "FCF/Precio": group_rank(f.get("fcf_yield"), sectors),
    }
    return pd.DataFrame(parts).assign(Valor=combine(parts, min_parts=2))


def score_events(s: pd.DataFrame, f: pd.DataFrame | None = None,
                 earnings: pd.Series | None = None, today=None, days_ahead=7) -> pd.DataFrame:
    """Event-driven: actividad anormal (volumen, gap, ruptura de maximos) + catalizadores
    conocidos (reporte trimestral en los proximos dias, upside de analistas)."""
    parts = {}
    if "vol_surge" in s:
        parts["Volumen anormal"] = pct_rank(s["vol_surge"])
    if "gap_1d" in s:
        parts["Movimiento fuerte"] = pct_rank(s["gap_1d"])
    parts["Ruptura de max"] = (s["near_high"] >= 0.98).astype(float).where(s["near_high"].notna()) * 100
    if f is not None and "upside" in f:
        parts["Upside analistas"] = pct_rank(f["upside"].reindex(s.index))
    df = pd.DataFrame(parts)
    df["Eventos"] = combine(parts, min_parts=2)
    if earnings is not None and len(earnings):
        today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
        e = pd.to_datetime(earnings.reindex(s.index))
        dias = (e - today).dt.days
        df["Reporta en (dias)"] = dias.where((dias >= 0) & (dias <= 30))
        df.loc[(dias >= 0) & (dias <= days_ahead), "Eventos"] += 10  # catalizador cercano
        df["Eventos"] = df["Eventos"].clip(0, 100)
    return df


def score_multifactor(s: pd.DataFrame, f: pd.DataFrame | None, sectors=None,
                      weights=None) -> pd.DataFrame:
    parts = {"Momentum": group_rank(s["mom_12_1"], sectors),
             "Baja volatilidad": score_lowvol(s, sectors)}
    if f is not None and not f.empty:
        f = f.reindex(s.index)
        parts["Calidad"] = score_quality(f, sectors)["Calidad"]
        parts["Valor"] = score_value(f, sectors)["Valor"]
    w = weights or {"Momentum": .3, "Baja volatilidad": .2, "Calidad": .3, "Valor": .2}
    return pd.DataFrame(parts).assign(Multifactor=combine(parts, w, min_parts=2))
