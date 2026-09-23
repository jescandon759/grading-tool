"""
Motor del Investment Score (sin Streamlit, testeable).

Cambios v2 frente a la version original:
- Valuacion con YIELDS (utilidad/precio, libros/precio, ventas/precio, EBITDA/EV):
  una empresa con perdidas queda al fondo en vez de quedar "N/A" (neutral).
  Tambien elimina el doble conteo P/E + earnings yield (eran la misma metrica).
- Pares del MISMO SECTOR (S&P 500) por default: un banco ya no se compara con Nvidia.
- Descarga de fundamentales en paralelo, con reintentos y reporte de fallidos.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

FACTORES = {
    "Growth": [("revenue_growth", +1), ("earnings_growth", +1)],
    "Profitability": [("roe", +1), ("roa", +1), ("operating_margin", +1), ("net_margin", +1)],
    "CashFlow": [("fcf_yield", +1), ("fcf_margin", +1), ("cash_conversion", +1)],
    "Risk": [("debt_to_equity", -1), ("beta", -1), ("volatility", -1),
             ("current_ratio", +1), ("max_drawdown", +1)],
    "Valuation": [("earnings_yield", +1), ("book_yield", +1), ("sales_yield", +1),
                  ("ebitda_ev", +1), ("peg", -1)],
    "Quality": [("gross_margin", +1), ("ebitda_margin", +1)],
}
PESOS = {"Growth": 0.18, "Profitability": 0.22, "CashFlow": 0.15,
         "Risk": 0.15, "Valuation": 0.22, "Quality": 0.08}
BANDAS = [(90, "STRONG BUY"), (80, "BUY"), (70, "WEAK BUY / WATCHLIST"),
          (60, "HOLD"), (50, "WEAK HOLD"), (0, "AVOID")]
ETIQUETAS = {
    "revenue_growth": "Crecimiento ventas", "earnings_growth": "Crecimiento utilidades",
    "roe": "ROE", "roa": "ROA", "operating_margin": "Margen operativo", "net_margin": "Margen neto",
    "fcf_yield": "FCF yield", "fcf_margin": "Margen FCF", "cash_conversion": "Conversión a caja",
    "debt_to_equity": "Deuda/Capital", "beta": "Beta", "volatility": "Volatilidad",
    "current_ratio": "Razón corriente", "max_drawdown": "Peor caída",
    "earnings_yield": "Utilidad/Precio (1/PE)", "book_yield": "Libros/Precio (1/PB)",
    "sales_yield": "Ventas/Precio (1/PS)", "ebitda_ev": "EBITDA/EV", "peg": "PEG",
    "gross_margin": "Margen bruto", "ebitda_margin": "Margen EBITDA",
}
# Yahoo usa nombres de sector distintos a GICS
YAHOO_SECTOR = {
    "Technology": "Tecnologia", "Communication Services": "Comunicaciones",
    "Consumer Cyclical": "Consumo discrecional", "Consumer Defensive": "Consumo basico",
    "Financial Services": "Financiero", "Healthcare": "Salud", "Industrials": "Industrial",
    "Energy": "Energia", "Basic Materials": "Materiales", "Utilities": "Servicios publicos",
    "Real Estate": "Bienes raices",
}
MOM_PESOS = {"ret_1m": 0.5, "ret_3m": 1.0, "ret_6m": 1.5, "ret_12m": 1.5,
             "dist_sma50": 1.0, "dist_sma200": 1.0, "mom_vol": 0.8}
MOM_DIRS = {"ret_1m": 1, "ret_3m": 1, "ret_6m": 1, "ret_12m": 1,
            "dist_sma50": 1, "dist_sma200": 1, "mom_vol": -1}


def _g(d, k):
    v = d.get(k)
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _div(a, b, need_pos_b=True):
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0 or (need_pos_b and b <= 0):
        return np.nan
    return a / b


def build_metrics(info: dict, prices=None) -> dict:
    m = {}
    m["revenue_growth"] = _g(info, "revenueGrowth"); m["earnings_growth"] = _g(info, "earningsGrowth")
    m["roe"] = _g(info, "returnOnEquity"); m["roa"] = _g(info, "returnOnAssets")
    m["operating_margin"] = _g(info, "operatingMargins"); m["net_margin"] = _g(info, "profitMargins")
    m["gross_margin"] = _g(info, "grossMargins"); m["ebitda_margin"] = _g(info, "ebitdaMargins")
    fcf, mcap = _g(info, "freeCashflow"), _g(info, "marketCap")
    rev, ni = _g(info, "totalRevenue"), _g(info, "netIncomeToCommon")
    m["fcf_yield"] = _div(fcf, mcap); m["fcf_margin"] = _div(fcf, rev)
    m["cash_conversion"] = _div(fcf, ni)
    m["debt_to_equity"] = _g(info, "debtToEquity"); m["beta"] = _g(info, "beta")
    m["current_ratio"] = _g(info, "currentRatio")
    # ---- Valuacion en yields: negativos permitidos (perdidas -> al fondo del ranking)
    price = _g(info, "currentPrice")
    if not np.isfinite(price):
        price = _g(info, "regularMarketPrice")
    eps, bvps = _g(info, "trailingEps"), _g(info, "bookValue")
    ey = _div(eps, price)
    if not np.isfinite(ey):
        pe = _g(info, "trailingPE"); ey = 1 / pe if np.isfinite(pe) and pe != 0 else np.nan
    m["earnings_yield"] = ey
    by = _div(bvps, price)
    if not np.isfinite(by):
        pb = _g(info, "priceToBook"); by = 1 / pb if np.isfinite(pb) and pb != 0 else np.nan
    m["book_yield"] = by
    m["sales_yield"] = _div(rev, mcap)
    ebitda, ev = _g(info, "ebitda"), _g(info, "enterpriseValue")
    m["ebitda_ev"] = _div(ebitda, ev)
    if not np.isfinite(m["ebitda_ev"]):
        eve = _g(info, "enterpriseToEbitda"); m["ebitda_ev"] = 1 / eve if np.isfinite(eve) and eve > 0 else np.nan
    peg = _g(info, "pegRatio")
    if not np.isfinite(peg):
        peg = _g(info, "trailingPegRatio")
    m["peg"] = peg if np.isfinite(peg) and peg > 0 else np.nan   # PEG negativo no tiene sentido
    # ---- Riesgo de precio
    if prices is not None and len(pd.Series(prices).dropna()) > 30:
        p = pd.Series(prices).dropna(); ret = p.pct_change().dropna()
        m["volatility"] = float(ret.std() * np.sqrt(252))
        m["max_drawdown"] = float((p / p.cummax() - 1).min())
    else:
        m["volatility"] = m["max_drawdown"] = np.nan
    return m


def momentum_metrics(prices) -> dict:
    p = pd.Series(prices).dropna()
    if len(p) < 210:
        return {}
    ret = lambda d: (p.iloc[-1] / p.iloc[-d] - 1) if len(p) > d else np.nan
    sma50, sma200 = p.rolling(50).mean().iloc[-1], p.rolling(200).mean().iloc[-1]
    rd = p.pct_change().dropna()
    return {"ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_12m": ret(252),
            "dist_sma50": p.iloc[-1] / sma50 - 1 if sma50 > 0 else np.nan,
            "dist_sma200": p.iloc[-1] / sma200 - 1 if sma200 > 0 else np.nan,
            "mom_vol": rd.iloc[-63:].std() * np.sqrt(252) if len(rd) >= 63 else np.nan}


def momentum_score(df: pd.DataFrame) -> pd.Series:
    num = pd.Series(0.0, index=df.index); den = 0.0
    for c, w in MOM_PESOS.items():
        if c not in df or df[c].notna().sum() < 3:
            continue
        r = df[c].rank(pct=True) * 100
        num += (r if MOM_DIRS[c] > 0 else 100 - r).fillna(50) * w; den += w
    return num / den if den > 0 else pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------- DESCARGA
LAST_ERRORS: dict[str, str] = {}   # ticker -> ultimo motivo de fallo (para mensajes claros)


def _info_one(t, retries=3):
    import yfinance as yf
    for k in range(retries + 1):
        try:
            info = yf.Ticker(t).info or {}
            if len(info) > 5:
                LAST_ERRORS.pop(t, None)
                return t, info
            LAST_ERRORS[t] = "respuesta vacía (ticker inexistente o sin cobertura en Yahoo)"
        except Exception as e:
            msg = str(e)
            LAST_ERRORS[t] = ("Yahoo está limitando las consultas (rate limit)"
                              if ("Too Many" in msg or "Rate" in msg or "429" in msg) else msg[:160])
        time.sleep(1.5 * (k + 1))            # espera creciente: 1.5, 3, 4.5 s
    return t, None


def fetch_infos(tickers, workers=8, fetcher=None):
    """Info de Yahoo en paralelo. Devuelve ({ticker: info}, [fallidos])."""
    fetcher = fetcher or _info_one
    out, bad = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(fetcher, t) for t in tickers]):
            t, info = f.result()
            (out.__setitem__(t, info) if info else bad.append(t))
    return out, bad


def build_universe(infos: dict, close: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for t, info in infos.items():
        p = close[t] if t in close.columns else None
        m = build_metrics(info, p)
        if p is not None:
            m.update(momentum_metrics(p))
        m["_name"] = info.get("shortName") or t
        m["_price"] = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan
        m["_sector"] = YAHOO_SECTOR.get(info.get("sector"), info.get("sector") or info.get("sector_es") or "")
        m["_fuente"] = info.get("fuente", "Yahoo Finance")
        m["_ultimo_reporte"] = info.get("ultimo_reporte")
        rows[t] = m
    df = pd.DataFrame(rows).T
    for c in df.columns:
        if not c.startswith("_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- SCORE
def _percentil(valor, serie, direccion):
    s = serie.dropna()
    if not np.isfinite(valor) or len(s) < 3:
        return np.nan
    pct = (s <= valor).mean() * 100.0
    return pct if direccion > 0 else 100.0 - pct


def factor_scores(met: dict, uni: pd.DataFrame) -> dict:
    out = {}
    for factor, metricas in FACTORES.items():
        pcts, det = [], []
        for m, d in metricas:
            val = met.get(m, np.nan)
            val = float(val) if isinstance(val, (int, float, np.floating)) else np.nan
            p = _percentil(val, uni[m] if m in uni else pd.Series(dtype=float), d)
            det.append((m, val, p))
            if np.isfinite(p):
                pcts.append(p)
        out[factor] = {"score": float(np.mean(pcts)) if pcts else np.nan,
                       "n": len(pcts), "total": len(metricas), "det": det}
    return out


def investment_score(fs: dict) -> float:
    num = den = 0.0
    for f, i in fs.items():
        if np.isfinite(i["score"]):
            num += PESOS[f] * i["score"]; den += PESOS[f]
    return num / den if den > 0 else np.nan


def confianza(fs: dict) -> float:
    disp = sum(i["n"] for i in fs.values()); tot = sum(i["total"] for i in fs.values())
    return 100.0 * disp / tot if tot else 0.0


def recomendacion(s):
    if not np.isfinite(s):
        return "N/A"
    return next(e for u, e in BANDAS if s >= u)


def nivel_riesgo(rs):
    return "N/A" if not np.isfinite(rs) else ("BAJO" if rs >= 66 else "MEDIO" if rs >= 33 else "ALTO")


def explicar(fs: dict):
    val = {f: i["score"] for f, i in fs.items() if np.isfinite(i["score"])}
    orden = sorted(val.items(), key=lambda kv: kv[1], reverse=True)
    pos = [f for f, s in orden if s >= 60][:3]
    rie = [f for f, s in orden[::-1] if s < 50][:3]
    todas = [(ETIQUETAS.get(m, m), p) for i in fs.values() for m, v, p in i["det"] if np.isfinite(p)]
    ok = sorted(todas, key=lambda x: x[1], reverse=True)
    return pos, rie, [n for n, p in ok if p >= 70][:4], [n for n, p in ok[::-1] if p <= 30][:4]


def rank_universe(uni: pd.DataFrame) -> pd.DataFrame:
    """Investment Score de TODAS las empresas del universo (misma regla que la individual)."""
    cols = ["Empresa", "Sector", "Score", "Confianza", *FACTORES, "Momentum", "Recomendación"]
    if uni is None or uni.empty:
        return pd.DataFrame(columns=cols)
    filas = {}
    for t in uni.index:
        fs = factor_scores(uni.loc[t].to_dict(), uni)
        filas[t] = {"Empresa": uni.loc[t, "_name"], "Sector": uni.loc[t, "_sector"],
                    "Score": investment_score(fs), "Confianza": confianza(fs),
                    **{f: fs[f]["score"] for f in FACTORES}}
    df = pd.DataFrame(filas).T
    for c in ["Score", "Confianza", *FACTORES]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Momentum"] = momentum_score(uni)
    df["Recomendación"] = df["Score"].map(recomendacion)
    return df.sort_values("Score", ascending=False)


# ---------------------------------------------------------------- SENALES PARA BACKTEST
BT_FEATS = list(MOM_PESOS)


def bt_features(prices: pd.DataFrame) -> dict:
    return {"ret_1m": prices / prices.shift(21) - 1, "ret_3m": prices / prices.shift(63) - 1,
            "ret_6m": prices / prices.shift(126) - 1, "ret_12m": prices / prices.shift(252) - 1,
            "dist_sma50": prices / prices.rolling(50).mean() - 1,
            "dist_sma200": prices / prices.rolling(200).mean() - 1,
            "mom_vol": prices.pct_change(fill_method=None).rolling(63).std() * np.sqrt(252)}
