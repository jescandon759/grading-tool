"""
Investment Score - App web (archivo único, autocontenido).
Ejecuta:  streamlit run score_app.py
Calcula un Score 0-100 por factores para una acción, con datos de Yahoo Finance.
No predice precios ni es asesoría financiera.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Investment Score", page_icon="🎯", layout="wide")

# ================= CONFIG =================
FACTORES = {
    "Growth": [("revenue_growth", +1), ("earnings_growth", +1)],
    "Profitability": [("roe", +1), ("roa", +1), ("operating_margin", +1), ("net_margin", +1)],
    "CashFlow": [("fcf_yield", +1), ("fcf_margin", +1), ("cash_conversion", +1)],
    "Risk": [("debt_to_equity", -1), ("beta", -1), ("volatility", -1),
             ("current_ratio", +1), ("max_drawdown", +1)],
    "Valuation": [("pe", -1), ("peg", -1), ("pb", -1), ("ps", -1),
                  ("ev_ebitda", -1), ("earnings_yield", +1)],
    "Quality": [("gross_margin", +1), ("ebitda_margin", +1)],
}
PESOS = {"Growth": 0.18, "Profitability": 0.22, "CashFlow": 0.15,
         "Risk": 0.15, "Valuation": 0.22, "Quality": 0.08}
MULTIPLOS_NO_NEGATIVOS = ["pe", "peg", "pb", "ps", "ev_ebitda"]
BANDAS = [(90, "STRONG BUY"), (80, "BUY"), (70, "WEAK BUY / WATCHLIST"),
          (60, "HOLD"), (50, "WEAK HOLD"), (0, "AVOID")]
ETIQUETAS = {
    "revenue_growth": "Crecimiento ventas", "earnings_growth": "Crecimiento utilidades",
    "roe": "ROE", "roa": "ROA", "operating_margin": "Margen operativo", "net_margin": "Margen neto",
    "fcf_yield": "FCF yield", "fcf_margin": "Margen FCF", "cash_conversion": "Conversión a caja",
    "debt_to_equity": "Deuda/Capital", "beta": "Beta", "volatility": "Volatilidad",
    "current_ratio": "Razón corriente", "max_drawdown": "Peor caída",
    "pe": "P/E", "peg": "PEG", "pb": "P/B", "ps": "P/S", "ev_ebitda": "EV/EBITDA",
    "earnings_yield": "Earnings yield", "gross_margin": "Margen bruto", "ebitda_margin": "Margen EBITDA",
}
UNIVERSO_DEFAULT = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
    "GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "AMZN", "TSLA", "HD", "MCD",
    "NKE", "SBUX", "LOW", "PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "AXP", "LLY", "UNH", "JNJ", "MRK",
    "ABBV", "PFE", "TMO", "CAT", "GE", "BA", "HON", "UPS", "RTX", "DE",
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "SHW", "FCX", "NEM", "APD",
    "NEE", "DUK", "SO", "D", "AEP", "PLD", "AMT", "EQIX", "SPG", "O",
]

def nivel_riesgo(rs):
    return "N/A" if np.isnan(rs) else ("BAJO" if rs >= 66 else "MEDIO" if rs >= 33 else "ALTO")

# ================= MOTOR =================
def _g(d, k):
    v = d.get(k)
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan

def build_metrics(info, prices=None):
    m = {}
    m["revenue_growth"] = _g(info, "revenueGrowth"); m["earnings_growth"] = _g(info, "earningsGrowth")
    m["roe"] = _g(info, "returnOnEquity"); m["roa"] = _g(info, "returnOnAssets")
    m["operating_margin"] = _g(info, "operatingMargins"); m["net_margin"] = _g(info, "profitMargins")
    m["gross_margin"] = _g(info, "grossMargins"); m["ebitda_margin"] = _g(info, "ebitdaMargins")
    fcf = _g(info, "freeCashflow"); mcap = _g(info, "marketCap")
    rev = _g(info, "totalRevenue"); ni = _g(info, "netIncomeToCommon")
    m["fcf_yield"] = fcf / mcap if (np.isfinite(fcf) and np.isfinite(mcap) and mcap > 0) else np.nan
    m["fcf_margin"] = fcf / rev if (np.isfinite(fcf) and np.isfinite(rev) and rev > 0) else np.nan
    m["cash_conversion"] = fcf / ni if (np.isfinite(fcf) and np.isfinite(ni) and ni > 0) else np.nan
    m["debt_to_equity"] = _g(info, "debtToEquity"); m["beta"] = _g(info, "beta")
    m["current_ratio"] = _g(info, "currentRatio")
    if prices is not None and len(pd.Series(prices).dropna()) > 30:
        p = pd.Series(prices).dropna(); ret = p.pct_change().dropna()
        m["volatility"] = float(ret.std() * np.sqrt(252))
        curva = (1 + ret).cumprod()
        m["max_drawdown"] = float(((curva - curva.cummax()) / curva.cummax()).min())
    else:
        m["volatility"] = np.nan; m["max_drawdown"] = np.nan
    m["pe"] = _g(info, "trailingPE"); m["peg"] = _g(info, "pegRatio")
    m["pb"] = _g(info, "priceToBook"); m["ps"] = _g(info, "priceToSalesTrailing12Months")
    m["ev_ebitda"] = _g(info, "enterpriseToEbitda")
    pe = m["pe"]; m["earnings_yield"] = (1.0 / pe) if (np.isfinite(pe) and pe > 0) else np.nan
    return m

def momentum_metrics(prices):
    """7 señales de momentum de una serie de precios (necesita ~1 año+)."""
    p = pd.Series(prices).dropna()
    if len(p) < 210:
        return {}
    ret = lambda d: (p.iloc[-1] / p.iloc[-d] - 1) if len(p) > d else np.nan
    sma50 = p.rolling(50).mean().iloc[-1]
    sma200 = p.rolling(200).mean().iloc[-1]
    rd = p.pct_change().dropna()
    return {"ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_12m": ret(252),
            "dist_sma50": (p.iloc[-1] / sma50 - 1) if sma50 > 0 else np.nan,
            "dist_sma200": (p.iloc[-1] / sma200 - 1) if sma200 > 0 else np.nan,
            "mom_vol": rd.iloc[-63:].std() * np.sqrt(252) if len(rd) >= 63 else np.nan}

MOM_PESOS = {"ret_1m": 0.5, "ret_3m": 1.0, "ret_6m": 1.5, "ret_12m": 1.5,
             "dist_sma50": 1.0, "dist_sma200": 1.0, "mom_vol": 0.8}
MOM_DIRS = {"ret_1m": 1, "ret_3m": 1, "ret_6m": 1, "ret_12m": 1,
            "dist_sma50": 1, "dist_sma200": 1, "mom_vol": -1}

def momentum_score(df):
    cols = [c for c in MOM_PESOS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    num = pd.Series(0.0, index=df.index); den = 0.0
    for c in cols:
        if df[c].notna().sum() < 3:
            continue
        r = df[c].rank(pct=True) * 100
        pct = r if MOM_DIRS[c] > 0 else (100 - r)
        num = num + pct.fillna(50) * MOM_PESOS[c]; den += MOM_PESOS[c]
    return num / den if den > 0 else pd.Series(np.nan, index=df.index)

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_all(tickers):
    import yfinance as yf
    try:
        prices = yf.download(list(tickers), period="2y", auto_adjust=True, progress=False)["Close"]
    except Exception:
        prices = pd.DataFrame()
    rows = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}
        p = prices[t] if (hasattr(prices, "columns") and t in prices.columns) else None
        m = build_metrics(info, p)
        if p is not None:
            m.update(momentum_metrics(p))
        m["_name"] = info.get("shortName") or t
        m["_price"] = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan
        rows[t] = m
    df = pd.DataFrame(rows).T
    for c in df.columns:
        if not c.startswith("_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_universe(df):
    df = df.copy()
    for col in MULTIPLOS_NO_NEGATIVOS:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    return df.replace([np.inf, -np.inf], np.nan)

def _percentil(valor, serie, direccion):
    s = serie.dropna()
    if np.isnan(valor) or len(s) < 3:
        return np.nan
    pct = (s <= valor).mean() * 100.0
    return pct if direccion > 0 else (100.0 - pct)

def factor_scores(met, uni):
    out = {}
    for factor, metricas in FACTORES.items():
        pcts, det = [], []
        for m, d in metricas:
            val = met.get(m, np.nan)
            serie = uni[m] if m in uni.columns else pd.Series(dtype=float)
            p = _percentil(val, serie, d)
            det.append((m, val, p))
            if not np.isnan(p):
                pcts.append(p)
        out[factor] = {"score": float(np.mean(pcts)) if pcts else np.nan,
                       "n": len(pcts), "total": len(metricas), "det": det}
    return out

def investment_score(fs):
    num = den = 0.0
    for f, i in fs.items():
        if not np.isnan(i["score"]):
            num += PESOS[f] * i["score"]; den += PESOS[f]
    return (num / den) if den > 0 else np.nan

def confianza(fs):
    disp = sum(i["n"] for i in fs.values()); tot = sum(i["total"] for i in fs.values())
    return 100.0 * disp / tot if tot else 0.0

def recomendacion(s):
    for u, e in BANDAS:
        if s >= u:
            return e
    return "AVOID"

def explicar(fs):
    val = {f: i["score"] for f, i in fs.items() if not np.isnan(i["score"])}
    orden = sorted(val.items(), key=lambda kv: kv[1], reverse=True)
    pos = [f for f, s in orden if s >= 60][:3]
    rie = [f for f, s in orden[::-1] if s < 50][:3]
    todas = [(ETIQUETAS.get(m, m), p) for f, i in fs.items() for m, v, p in i["det"] if not np.isnan(p)]
    ok = sorted(todas, key=lambda x: x[1], reverse=True)
    fuertes = [n for n, p in ok if p >= 70][:4]
    debiles = [n for n, p in ok[::-1] if p <= 30][:4]
    return pos, rie, fuertes, debiles

# ---- Contexto: noticias (Yahoo) + analistas ----
_POS = {"surge", "surges", "soar", "soars", "jump", "jumps", "rally", "beat", "beats", "tops",
        "upgrade", "upgraded", "raises", "record", "strong", "growth", "gains", "rises", "boost",
        "outperform", "buy", "bullish", "wins", "expands", "profit", "rebound"}
_NEG = {"probe", "antitrust", "lawsuit", "sue", "sued", "plunge", "plunges", "miss", "misses",
        "downgrade", "downgraded", "cuts", "cut", "falls", "fall", "drop", "drops", "slump", "fraud",
        "warning", "weak", "layoffs", "bearish", "recall", "decline", "loss", "sinks", "tumbles"}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_context(ticker):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    titulares = []
    try:
        for n in (tk.news or [])[:10]:
            t = n.get("title") or (n.get("content") or {}).get("title")
            if t:
                titulares.append(t)
    except Exception:
        pass
    try:
        info = tk.info
    except Exception:
        info = {}
    analistas = {"target_mean": info.get("targetMeanPrice"), "target_high": info.get("targetHighPrice"),
                 "target_low": info.get("targetLowPrice"), "n": info.get("numberOfAnalystOpinions"),
                 "reco": info.get("recommendationKey"),
                 "current": info.get("currentPrice") or info.get("regularMarketPrice"),
                 "rev_growth": info.get("revenueGrowth"), "earn_growth": info.get("earningsGrowth")}
    return titulares, analistas

def sentimiento_titulares(titulares):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    sia = SentimentIntensityAnalyzer(); out = []
    for t in titulares:
        v = sia.polarity_scores(t)["compound"]
        w = set(t.lower().replace(",", " ").replace(".", " ").split())
        c = max(-1, min(1, v + 0.35 * (len(w & _POS) - len(w & _NEG))))
        et = "🟢 Positivo" if c > 0.1 else "🔴 Negativo" if c < -0.1 else "⚪ Neutral"
        out.append((t, c, et))
    return out

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_backtest(ticker):
    import yfinance as yf
    try:
        px = yf.download([ticker, "^GSPC"], period="10y", auto_adjust=True, progress=False)["Close"]
    except Exception:
        px = pd.DataFrame()
    fin = None
    try:
        fin = yf.Ticker(ticker).income_stmt
    except Exception:
        fin = None
    return px, fin

def backtest_stock(px_stock, px_spx, rf=0.04):
    s = px_stock.dropna(); m = px_spx.dropna()
    idx = s.index.intersection(m.index)
    s, m = s.loc[idx], m.loc[idx]
    if len(s) < 260:
        return None
    cagr = lambda p: (p.iloc[-1] / p.iloc[0]) ** (252 / len(p)) - 1
    rs, rm = s.pct_change().dropna(), m.pct_change().dropna()
    j = rs.index.intersection(rm.index)
    rs, rm = rs.loc[j], rm.loc[j]
    vol = lambda r: r.std() * np.sqrt(252)
    mdd = lambda p: (p / p.cummax() - 1).min()
    # beta y alpha (CAPM)
    var_m = rm.var()
    beta = (rs.cov(rm) / var_m) if var_m > 0 else np.nan
    cagr_s, cagr_m = cagr(s), cagr(m)
    alpha = cagr_s - (rf + beta * (cagr_m - rf)) if not np.isnan(beta) else np.nan
    # sortino (solo castiga caidas)
    downside = rs[rs < 0].std() * np.sqrt(252)
    sortino = (cagr_s - rf) / downside if downside > 0 else np.nan
    roll_s = (s / s.shift(252) - 1).dropna()
    roll_m = (m / m.shift(252) - 1).dropna()
    k = roll_s.index.intersection(roll_m.index)
    excess = roll_s.loc[k] - roll_m.loc[k]
    out = {"years": len(s) / 252, "cagr_s": cagr_s, "cagr_m": cagr_m,
           "vol_s": vol(rs), "vol_m": vol(rm),
           "sharpe_s": (cagr_s - rf) / vol(rs) if vol(rs) > 0 else np.nan,
           "sharpe_m": (cagr_m - rf) / vol(rm) if vol(rm) > 0 else np.nan,
           "mdd_s": mdd(s), "mdd_m": mdd(m),
           "avg1y_s": roll_s.mean(), "avg1y_m": roll_m.mean(),
           "avg_excess": excess.mean(), "win_rate": (excess > 0).mean() * 100,
           "beta": beta, "alpha": alpha, "sortino": sortino,
           "best1y": roll_s.max(), "worst1y": roll_s.min()}
    return out, s / s.iloc[0], m / m.iloc[0], roll_s, roll_m

def tabla_financieros(stmt):
    if stmt is None or getattr(stmt, "empty", True):
        return None
    def fila(nombre):
        return stmt.loc[nombre] if nombre in stmt.index else None
    rev, ni = fila("Total Revenue"), fila("Net Income")
    if rev is None:
        return None
    data = []
    for c in stmt.columns:
        anio = c.year if hasattr(c, "year") else str(c)
        r = rev[c] if rev is not None else np.nan
        n = ni[c] if ni is not None else np.nan
        margin = (n / r) if (pd.notna(r) and pd.notna(n) and r) else np.nan
        data.append({"Año": anio, "Ventas": r, "Utilidad neta": n, "Margen neto": margin})
    df = pd.DataFrame(data).sort_values("Año").reset_index(drop=True)
    df["Crec. ventas"] = df["Ventas"].pct_change()
    return df

# ---------------- MOTOR DE BACKTEST (momentum, walk-forward) ----------------
BT_FEATS = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "dist_sma50", "dist_sma200", "mom_vol"]
BT_DIRS = {"ret_1m": 1, "ret_3m": 1, "ret_6m": 1, "ret_12m": 1,
           "dist_sma50": 1, "dist_sma200": 1, "mom_vol": -1}

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_prices_bt(universe):
    import yfinance as yf
    data = yf.download(list(universe) + ["^GSPC"], period="10y", auto_adjust=True, progress=False)["Close"]
    data = data.dropna(how="all").ffill()
    spx = data["^GSPC"].dropna()
    prices = data[[c for c in universe if c in data.columns]].dropna(how="all")
    prices, spx = prices.align(spx, join="inner", axis=0)
    return prices, spx

def bt_build_features(prices):
    f = {}
    f["ret_1m"] = prices / prices.shift(21) - 1
    f["ret_3m"] = prices / prices.shift(63) - 1
    f["ret_6m"] = prices / prices.shift(126) - 1
    f["ret_12m"] = prices / prices.shift(252) - 1
    f["dist_sma50"] = prices / prices.rolling(50).mean() - 1
    f["dist_sma200"] = prices / prices.rolling(200).mean() - 1
    f["mom_vol"] = prices.pct_change().rolling(63).std() * np.sqrt(252)
    return f

def bt_walk_forward(prices, spx, pesos, topn=10, rebal=21, start=252):
    feats = bt_build_features(prices)
    idxs = list(range(start, len(prices) - rebal, rebal))
    rs, rb, fechas, hit12 = [], [], [], []
    for i in idxs:
        s = pd.Series(0.0, index=prices.columns); den = 0.0
        for k in BT_FEATS:
            row = feats[k].iloc[i]
            if row.notna().sum() < 3:
                continue
            r = row.rank(pct=True) * 100
            s = s + (r if BT_DIRS[k] > 0 else 100 - r).fillna(50) * pesos[k]; den += pesos[k]
        s = (s / den) if den > 0 else s
        top = s.where(prices.iloc[i].notna()).dropna().sort_values(ascending=False).head(topn).index
        if len(top) == 0:
            continue
        p0, p1 = prices.iloc[i], prices.iloc[i + rebal]
        rs.append((p1[top] / p0[top] - 1).mean()); rb.append(spx.iloc[i + rebal] / spx.iloc[i] - 1)
        fechas.append(prices.index[i])
        if i + 252 < len(prices):
            r12s = (prices.iloc[i + 252][top] / p0[top] - 1).mean()
            hit12.append(1 if r12s > (spx.iloc[i + 252] / spx.iloc[i] - 1) else 0)
    eq_s = pd.Series(np.cumprod([1 + x for x in rs]), index=fechas)
    eq_b = pd.Series(np.cumprod([1 + x for x in rb]), index=fechas)
    return {"rs": np.array(rs), "rb": np.array(rb), "eq_s": eq_s, "eq_b": eq_b,
            "hit12": np.mean(hit12) if hit12 else np.nan, "n": len(rs)}

def bt_metricas(res):
    rs, rb = res["rs"], res["rb"]
    ann = lambda r: (np.prod(1 + r)) ** (12 / len(r)) - 1 if len(r) else np.nan
    vol = lambda r: r.std() * np.sqrt(12)
    dd = lambda eq: (eq / eq.cummax() - 1).min()
    return {"cagr_s": ann(rs), "cagr_b": ann(rb), "exceso": ann(rs) - ann(rb),
            "sharpe_s": (ann(rs) - 0.04) / vol(rs) if vol(rs) > 0 else np.nan,
            "sharpe_b": (ann(rb) - 0.04) / vol(rb) if vol(rb) > 0 else np.nan,
            "mdd_s": dd(res["eq_s"]), "mdd_b": dd(res["eq_b"]), "hit12": res["hit12"], "n": res["n"]}

def bt_prep(prices, spx, topn, rebal=21, start=252):
    feats = bt_build_features(prices)
    idxs = list(range(start, len(prices) - rebal, rebal)); pasos = []
    for i in idxs:
        pm = {}
        for k in BT_FEATS:
            r = feats[k].iloc[i].rank(pct=True) * 100
            pm[k] = (r if BT_DIRS[k] > 0 else 100 - r).fillna(50)
        pasos.append({"pmat": pd.DataFrame(pm), "fwd1": prices.iloc[i + rebal] / prices.iloc[i] - 1,
                      "b1": spx.iloc[i + rebal] / spx.iloc[i] - 1, "valid": prices.iloc[i].notna()})
    return pasos

def bt_eval(pasos, topn, pesos):
    rs, rb = [], []; w = np.array([pesos[k] for k in BT_FEATS]); w = w / w.sum() if w.sum() > 0 else w
    for p in pasos:
        sc = pd.Series(p["pmat"][BT_FEATS].values @ w, index=p["pmat"].index).where(p["valid"])
        top = sc.dropna().sort_values(ascending=False).head(topn).index
        if len(top) == 0:
            continue
        rs.append(p["fwd1"][top].mean()); rb.append(p["b1"])
    rs, rb = np.array(rs), np.array(rb)
    ann = lambda r: (np.prod(1 + r)) ** (12 / len(r)) - 1 if len(r) else np.nan
    return ann(rs) - ann(rb)

def bt_optimizar_oos(prices, spx, topn=10):
    from scipy.optimize import minimize
    pasos = bt_prep(prices, spx, topn); n = len(pasos); c = int(n * 0.7)
    train, test = pasos[:c], pasos[c:]
    base = dict.fromkeys(BT_FEATS, 1.0)
    bt_tr, bt_te = bt_eval(train, topn, base), bt_eval(test, topn, base)
    res = minimize(lambda wv: -bt_eval(train, topn, dict(zip(BT_FEATS, np.clip(wv, 0, None)))),
                   np.ones(len(BT_FEATS)), method="Nelder-Mead",
                   options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-4})
    w = np.clip(res.x, 0, None); po = dict(zip(BT_FEATS, w))
    return {"base_train": bt_tr, "base_test": bt_te,
            "opt_train": bt_eval(train, topn, po), "opt_test": bt_eval(test, topn, po),
            "pesos": {k: round(v, 2) for k, v in zip(BT_FEATS, w / w.sum() if w.sum() > 0 else w)}}

# ================= INTERFAZ =================
st.title("🎯 Investment Score")
st.caption("Califica una acción de 0 a 100 por factores (crecimiento, rentabilidad, caja, "
           "riesgo, valuación, calidad), comparándola contra un universo de pares. "
           "No predice precios ni es asesoría financiera.")

with st.expander("¿Cómo se calcula el puntaje? (transparencia total)"):
    st.markdown("""
1. Cada métrica se convierte en **percentil** (0–100) comparándola contra el universo de pares.
   Las de "menos es mejor" (P/E, deuda) se invierten. Robusto a valores extremos.
2. Cada **factor** es el promedio de los percentiles de sus métricas disponibles.
3. El **Score** es el promedio ponderado de los factores (pesos fijos, transparentes).
4. La **Confianza** es el % de datos realmente disponibles. Datos faltantes → N/A (no se inventan).

**Control de sesgos:** los múltiplos negativos (P/E, P/B ≤ 0 = pérdidas) se marcan N/A,
no se premian como "baratos". El score es **relativo** al universo, no una nota absoluta.
""")

col1, col2 = st.columns([2, 1])
ticker = col1.text_input("Ticker de la empresa (EE.UU.)", "AAPL").strip().upper()
with st.expander("Universo de pares (editable)"):
    uni_txt = st.text_area("Tickers separados por coma", ", ".join(UNIVERSO_DEFAULT), height=100)

if st.button("🎯 Analizar", type="primary", use_container_width=True):
    universo = [t.strip().upper() for t in uni_txt.split(",") if t.strip()]
    universo = list(dict.fromkeys([ticker] + universo))
    if not ticker:
        st.error("Escribe un ticker."); st.stop()
    with st.spinner("Descargando datos de Yahoo (~1-2 min la 1ª vez)..."):
        df = clean_universe(fetch_all(tuple(universo)))
    if ticker not in df.index or df.loc[ticker].drop([c for c in df.columns if c.startswith("_")]).notna().sum() == 0:
        st.error(f"No se pudieron obtener datos de {ticker}. Revisa el ticker."); st.stop()

    met = df.loc[ticker].to_dict()
    fs = factor_scores(met, df)
    score = investment_score(fs); conf = confianza(fs)
    pos, rie, fuertes, debiles = explicar(fs)
    reco = recomendacion(score) if not np.isnan(score) else "N/A"
    riesgo = nivel_riesgo(fs["Risk"]["score"])
    factores_dict = {f: fs[f]["score"] for f in fs}
    color = {"STRONG BUY": "🟢", "BUY": "🟢", "WEAK BUY / WATCHLIST": "🟡",
             "HOLD": "🟡", "WEAK HOLD": "🟠", "AVOID": "🔴"}.get(reco, "⚪")

    st.markdown(f"## {met.get('_name', ticker)}  ·  `{ticker}`")
    price = met.get("_price", np.nan)
    if not (isinstance(price, float) and np.isnan(price)):
        st.write(f"Precio actual: **${price}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Investment Score", f"{score:.0f} / 100" if not np.isnan(score) else "N/A")
    m2.metric("Recomendación", f"{color} {reco}")
    m3.metric("Nivel de riesgo", riesgo)
    m4.metric("Confianza (datos)", f"{conf:.0f}%")
    if conf < 60:
        st.warning("Confianza baja: a esta empresa le faltan datos. Interpreta el score con cautela.")

    # Gráfica de factores
    st.subheader("Desglose por factores")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    facs = ["Growth", "Profitability", "CashFlow", "Risk", "Valuation", "Quality"]
    vals = [factores_dict[f] if pd.notna(factores_dict[f]) else 0 for f in facs]
    colores = ["#2e8b57" if v >= 66 else "#e67e22" if v >= 33 else "#c0392b" for v in vals]
    ax.barh(facs[::-1], vals[::-1], color=colores[::-1])
    ax.set_xlim(0, 100); ax.axvline(50, color="gray", ls=":", lw=1)
    ax.set_xlabel("Puntaje del factor (0-100, relativo a los pares)")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)
    st.pyplot(fig)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**✅ Factores más fuertes**")
        for f in pos or ["(ninguno destaca)"]:
            st.write(f"- {f}")
        if fuertes:
            st.caption("Destaca en: " + ", ".join(fuertes))
    with cB:
        st.markdown("**⚠️ Principales riesgos**")
        for f in rie or ["(ninguno crítico)"]:
            st.write(f"- {f}")
        if debiles:
            st.caption("Más débil en: " + ", ".join(debiles))

    st.divider()

    # ---- Momentum / Timing ----
    st.subheader("📈 Momentum (¿buen momento para comprar?)")
    msc = momentum_score(df)[ticker]
    mrow = met
    if pd.notna(msc):
        etiqueta_m = "🟢 fuerte (subiendo)" if msc >= 60 else "🔴 débil (cayendo)" if msc < 40 else "🟡 neutral"
        mc1, mc2 = st.columns([1, 2])
        mc1.metric("Momentum Score", f"{msc:.0f} / 100", etiqueta_m, delta_color="off")
        with mc2:
            def _pm(x): return f"{x*100:+.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "n/d"
            st.write(f"1m {_pm(mrow.get('ret_1m'))} · 3m {_pm(mrow.get('ret_3m'))} · "
                     f"6m {_pm(mrow.get('ret_6m'))} · 12m {_pm(mrow.get('ret_12m'))}")
            st.write(f"vs SMA50: {_pm(mrow.get('dist_sma50'))} · vs SMA200: {_pm(mrow.get('dist_sma200'))} · "
                     f"volatilidad reciente: {_pm(mrow.get('mom_vol'))}")

        # Veredicto: fundamentales x momentum (lo que pediste)
        fund_bueno = (not np.isnan(score)) and score >= 52
        mom_bueno = msc >= 55; mom_malo = msc < 45
        if fund_bueno and mom_bueno:
            st.success("🟢 **Buena empresa Y buen momento.** Fundamentales sólidos con tendencia al alza.")
        elif fund_bueno and mom_malo:
            st.warning("🟡 **Buena empresa, pero MAL momento.** Los fundamentales se ven bien, pero la "
                       "acción viene cayendo. Podría ser oportunidad... o una señal temprana de problemas. "
                       "Muchos inversores esperan a que el precio se estabilice antes de entrar.")
        elif (not fund_bueno) and mom_bueno:
            st.warning("🟠 **Buen momento, pero fundamentales flojos.** Sube de precio sin sustento sólido — "
                       "cuidado con perseguir un rally que puede revertirse.")
        else:
            st.error("🔴 **Ni fundamentales ni momento.** Ni el negocio ni el precio acompañan ahora.")
        st.caption("El momentum de 6-12 meses tiene respaldo académico; el de 1 mes es ruidoso y suele "
                   "revertirse (por eso pesa menos). El momentum NO entra al Investment Score: es una "
                   "capa de TIMING aparte, no de calidad. Timing pasado ≠ timing futuro.")
    else:
        st.info("No hay suficiente historia de precios para calcular el momentum.")

    st.divider()

    # ---- Estimaciones de analistas (crecimiento/valor esperado) ----
    st.subheader("📊 Lo que esperan los analistas")
    try:
        titulares, an = fetch_context(ticker)
    except Exception:
        titulares, an = [], {}
    tm = an.get("target_mean"); cur = an.get("current") or price
    if tm and cur and not (isinstance(cur, float) and np.isnan(cur)):
        upside = (tm / cur - 1) * 100
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Precio actual", f"${cur:,.2f}")
        a2.metric("Precio objetivo (prom.)", f"${tm:,.2f}", f"{upside:+.1f}%")
        a3.metric("Rango objetivo", f"${an.get('target_low', 0):,.0f} – ${an.get('target_high', 0):,.0f}")
        a4.metric("Recomendación", str(an.get("reco", "n/d")).upper())

        # Contexto de dispersión: qué tan de acuerdo están
        lo, hi, n = an.get("target_low"), an.get("target_high"), an.get("n")
        if lo and hi and cur:
            disp = (hi - lo) / cur * 100
            acuerdo = "poca dispersión (bastante acuerdo)" if disp < 25 else \
                      "dispersión media" if disp < 60 else "MUCHA dispersión (poco acuerdo)"
            st.write(f"El rango va de **{(lo/cur-1)*100:+.0f}%** a **{(hi/cur-1)*100:+.0f}%** "
                     f"desde el precio de hoy → {acuerdo}. Basado en **{n or '?'}** analistas.")

        # Marca de sesgo según qué tan tibio es el upside
        if upside < 0:
            st.error("🔴 Ojo: el objetivo está POR DEBAJO del precio actual. Dado que los analistas "
                     "casi siempre son optimistas, un objetivo negativo es una señal notablemente mala.")
        elif upside < 10:
            st.warning("🟠 Upside tibio. Los analistas casi siempre proyectan alzas (sesgo del gremio), "
                       "así que un +" + f"{upside:.0f}%" + " en realidad es una señal débil, casi de 'ni fu ni fa'.")
        else:
            st.info("🟡 Recuerda: los analistas tienden a ser optimistas (ponen muchos más 'comprar' que "
                    "'vender'), así que un upside positivo es lo normal, no una señal especial. Míralo con esa lupa.")

        st.caption(f"El 'precio objetivo' es su expectativa a ~12 meses. Es una opinión profesional, "
                   "no una garantía — los analistas se equivocan seguido y tienen sesgo optimista estructural.")
    else:
        st.info("No hay estimaciones de analistas disponibles para esta acción.")
    rg, eg = an.get("rev_growth"), an.get("earn_growth")
    if rg is not None or eg is not None:
        st.write(f"Crecimiento reciente (últimos 12m): "
                 f"ventas **{rg*100:+.1f}%**" if isinstance(rg, (int, float)) else "ventas n/d",
                 f" · utilidades **{eg*100:+.1f}%**" if isinstance(eg, (int, float)) else " · utilidades n/d")

    # ---- Termómetro de noticias ----
    st.subheader("📰 Termómetro de noticias (contexto)")
    sent = sentimiento_titulares(titulares) if titulares else None
    if sent:
        prom = np.mean([c for _, c, _ in sent])
        etiqueta = "🟢 mayormente positivas" if prom > 0.1 else "🔴 mayormente negativas" if prom < -0.1 else "⚪ mixtas/neutrales"
        pos = sum(1 for _, c, _ in sent if c > 0.1); neg = sum(1 for _, c, _ in sent if c < -0.1)
        st.write(f"Titulares recientes: **{etiqueta}**  ({pos} positivos, {neg} negativos de {len(sent)})")
        for t, c, et in sent:
            st.write(f"{et}  ·  {t}")
        st.caption("⚠️ El sentimiento de noticias es una señal DÉBIL y ruidosa: el mercado ya "
                   "descontó la noticia en segundos. Léelo como contexto, no como predicción. "
                   "NO se usa en el Investment Score.")
    else:
        st.info("No hay noticias recientes disponibles (o falta la librería de sentimiento).")

    st.divider()

    # ---- Reporte de estados financieros anuales ----
    st.subheader("📄 Estados financieros anuales (Yahoo)")
    px_bt, fin = fetch_backtest(ticker)
    tf = tabla_financieros(fin)
    if tf is not None:
        show = tf.copy()
        show["Ventas"] = show["Ventas"].apply(lambda x: f"${x/1e9:,.1f}B" if pd.notna(x) else "n/d")
        show["Utilidad neta"] = show["Utilidad neta"].apply(lambda x: f"${x/1e9:,.1f}B" if pd.notna(x) else "n/d")
        show["Margen neto"] = show["Margen neto"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "n/d")
        show["Crec. ventas"] = show["Crec. ventas"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Yahoo suele dar solo 3-4 años. Pocos años = base estadística débil.")
    else:
        st.info("No hay estados financieros anuales disponibles para esta acción.")

    # ---- Backtest histórico de la acción ----
    st.subheader("⏳ Backtest histórico de la acción")
    bt = None
    if hasattr(px_bt, "columns") and ticker in px_bt.columns and "^GSPC" in px_bt.columns:
        bt = backtest_stock(px_bt[ticker], px_bt["^GSPC"])
    if bt is None:
        st.info("No hay suficiente historia de precios para el backtest.")
    else:
        o, sn, mn, roll_s, roll_m = bt
        st.write(f"Periodo analizado: **{o['years']:.1f} años** de historia de precios.")
        b1, b2, b3 = st.columns(3)
        b1.metric("CAGR (anualizado)", f"{o['cagr_s']*100:+.1f}%", f"{(o['cagr_s']-o['cagr_m'])*100:+.1f}% vs S&P")
        b2.metric("Retorno 1 año (prom.)", f"{o['avg1y_s']*100:+.1f}%", f"{o['avg_excess']*100:+.1f}% exceso")
        b3.metric("Gana al S&P", f"{o['win_rate']:.0f}% del tiempo")
        b4, b5, b6 = st.columns(3)
        b4.metric("Volatilidad", f"{o['vol_s']*100:.0f}%", f"S&P {o['vol_m']*100:.0f}%", delta_color="off")
        b5.metric("Sharpe", f"{o['sharpe_s']:.2f}", f"S&P {o['sharpe_m']:.2f}", delta_color="off")
        b6.metric("Peor caída", f"{o['mdd_s']*100:.0f}%", f"S&P {o['mdd_m']*100:.0f}%", delta_color="off")
        b7, b8, b9 = st.columns(3)
        b7.metric("Alpha (anual)", f"{o['alpha']*100:+.1f}%" if pd.notna(o['alpha']) else "n/d",
                  help="Rendimiento que le ganó al mercado AJUSTADO por su riesgo (CAPM). Positivo = valió la pena.")
        b8.metric("Beta", f"{o['beta']:.2f}" if pd.notna(o['beta']) else "n/d",
                  help="Cuánto se mueve con el mercado. >1 amplifica; <1 amortigua.")
        b9.metric("Sortino", f"{o['sortino']:.2f}" if pd.notna(o['sortino']) else "n/d",
                  help="Como Sharpe pero solo castiga las caídas, no las subidas.")
        st.write(f"Mejor año: **{o['best1y']*100:+.0f}%**  ·  Peor año: **{o['worst1y']*100:+.0f}%**  "
                 f"(retornos móviles a 1 año, el rango que ha vivido esta acción)")

        g1, g2 = st.columns(2)
        # Crecimiento en escala normal (facil: "$1 se volvio $X")
        fig1, ax1 = plt.subplots(figsize=(6, 3.6))
        ax1.plot(sn.index, sn.values, label=ticker, color="#2c6fbb", lw=2)
        ax1.plot(mn.index, mn.values, label="S&P 500", color="#7f8c8d", lw=2, ls="--")
        ax1.set_title("¿En cuánto se convirtió $1 invertido?")
        ax1.legend(fontsize=9); ax1.grid(alpha=.3)
        g1.pyplot(fig1)

        # Rendimiento por año en barras: verde=subió, rojo=bajó (facil de leer)
        def rend_anual(p):
            g = p.groupby(p.index.year)
            return (g.last() / g.first() - 1) * 100
        ya = rend_anual(sn)
        fig2, ax2 = plt.subplots(figsize=(6, 3.6))
        cols = ["#2e8b57" if v >= 0 else "#c0392b" for v in ya.values]
        ax2.bar([str(a) for a in ya.index], ya.values, color=cols)
        ax2.axhline(0, color="k", lw=.8)
        ax2.set_title(f"¿Cuánto subió o bajó {ticker} cada año?")
        ax2.set_ylabel("% en el año")
        for i, v in enumerate(ya.values):
            ax2.text(i, v + (2 if v >= 0 else -4), f"{v:+.0f}%", ha="center", fontsize=7)
        plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=8)
        g2.pyplot(fig2)

        st.caption("Izquierda: cuánto habría crecido tu dinero (azul = la acción, gris = el mercado). "
                   "Derecha: cuánto ganó (verde, arriba) o perdió (rojo, abajo) cada año. Los años rojos "
                   "son los sustos: así ves que no todo es subir.")

        st.warning("⚠️ Esto es el desempeño PASADO REAL de la acción vs el S&P 500 — NO prueba el "
                   "modelo de score, y el pasado no predice el futuro. Además tiene **survivorship "
                   "bias**: estás viendo una acción que sobrevivió; las que quebraron no aparecen. "
                   "Tómalo como historia, no como promesa.")

    st.divider()
    st.caption("Score RELATIVO al universo de pares, basado en datos actuales de Yahoo. "
               "Faltan DCF/valor intrínseco (próxima fase). NO es predicción ni recomendación de compra.")
else:
    st.info("Escribe un ticker y pulsa **Analizar**.")

# ---------------- SECCIÓN: BACKTEST DEL MODELO (MOMENTUM) ----------------
st.divider()
st.header("⏳ Backtest del modelo — capa de Momentum (experimental)")
st.warning("Esto NO prueba todo tu modelo. Solo backtestea la parte de MOMENTUM (lo único "
           "que se puede probar sin trampa con datos gratis). Tiene **survivorship bias** "
           "(solo acciones vivas) y usa ~9 años de un solo universo. El número del **TEST "
           "(fuera de muestra) manda sobre el TRAIN**. No es asesoría financiera.")

topn_bt = st.slider("Cuántas acciones compra el modelo cada mes", 5, 20, 10, key="topn_bt")
if st.button("⏳ Correr backtest (tarda ~1 min)", key="btn_bt", use_container_width=True):
    try:
        with st.spinner("Descargando 10 años de precios del universo..."):
            px_bt, sp_bt = fetch_prices_bt(tuple(UNIVERSO_DEFAULT))
        with st.spinner("Corriendo walk-forward y validación fuera de muestra..."):
            res_bt = bt_walk_forward(px_bt, sp_bt, dict.fromkeys(BT_FEATS, 1.0), topn=topn_bt)
            mbt = bt_metricas(res_bt)
            obt = bt_optimizar_oos(px_bt, sp_bt, topn=topn_bt)

        st.subheader("Resultado (pesos iguales, todo el periodo)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR estrategia", f"{mbt['cagr_s']*100:+.1f}%", f"{mbt['exceso']*100:+.1f}% vs S&P")
        k2.metric("CAGR S&P 500", f"{mbt['cagr_b']*100:+.1f}%")
        k3.metric("Sharpe", f"{mbt['sharpe_s']:.2f}", f"S&P {mbt['sharpe_b']:.2f}", delta_color="off")
        k4.metric("Gana al S&P (12m)", f"{mbt['hit12']*100:.0f}%")
        st.write(f"Peor caída estrategia **{mbt['mdd_s']*100:.0f}%** vs S&P {mbt['mdd_b']*100:.0f}%  ·  "
                 f"{mbt['n']} rebalanceos (~{mbt['n']/12:.1f} años).")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(res_bt["eq_s"].index, res_bt["eq_s"].values, label="Estrategia momentum", color="#2c6fbb", lw=1.6)
        ax.plot(res_bt["eq_b"].index, res_bt["eq_b"].values, label="S&P 500", color="#7f8c8d", lw=1.4, ls="--")
        ax.set_title("Crecimiento de $1 — estrategia vs S&P 500 (con survivorship bias)")
        ax.legend(); ax.grid(alpha=.3)
        st.pyplot(fig)

        st.subheader("Validación fuera de muestra (70% train / 30% test)")
        st.write(f"- Pesos **iguales**: exceso TRAIN **{obt['base_train']*100:+.1f}%** · TEST **{obt['base_test']*100:+.1f}%**")
        st.write(f"- Pesos **optimizados**: exceso TRAIN **{obt['opt_train']*100:+.1f}%** · TEST **{obt['opt_test']*100:+.1f}%**")
        if obt["opt_test"] < obt["base_test"] or obt["opt_test"] < obt["opt_train"] * 0.5:
            st.error("🔴 Veredicto: optimizar los pesos NO mejoró fuera de muestra → estaban "
                     "sobreajustados al pasado (overfitting). La señal simple es más honesta.")
        else:
            st.success("🟢 Veredicto: el exceso sobrevive fuera de muestra → señal más robusta (con cautela).")
        st.caption(f"Pesos optimizados (normalizados): {obt['pesos']}. "
                   "Recuerda: el TEST es el juez, no el TRAIN.")
    except Exception as e:
        st.warning(f"No se pudo correr el backtest: {e}")
