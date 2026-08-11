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

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_all(tickers):
    import yfinance as yf
    try:
        prices = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)["Close"]
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
    vol = lambda r: r.std() * np.sqrt(252)
    mdd = lambda p: (p / p.cummax() - 1).min()
    roll_s = (s / s.shift(252) - 1).dropna()
    roll_m = (m / m.shift(252) - 1).dropna()
    j = roll_s.index.intersection(roll_m.index)
    excess = roll_s.loc[j] - roll_m.loc[j]
    out = {"years": len(s) / 252, "cagr_s": cagr(s), "cagr_m": cagr(m),
           "vol_s": vol(rs), "vol_m": vol(rm),
           "sharpe_s": (cagr(s) - rf) / vol(rs) if vol(rs) > 0 else np.nan,
           "sharpe_m": (cagr(m) - rf) / vol(rm) if vol(rm) > 0 else np.nan,
           "mdd_s": mdd(s), "mdd_m": mdd(m),
           "avg1y_s": roll_s.mean(), "avg1y_m": roll_m.mean(),
           "avg_excess": excess.mean(), "win_rate": (excess > 0).mean() * 100}
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
    vals = [fs[f]["score"] if not np.isnan(fs[f]["score"]) else 0 for f in facs]
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
        st.caption(f"Basado en {an.get('n', 'varios')} analistas. El 'precio objetivo' es su expectativa "
                   "a ~12 meses — el crecimiento esperado más honesto que existe, pero los analistas "
                   "se equivocan seguido. NO es garantía.")
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

        g1, g2 = st.columns(2)
        fig1, ax1 = plt.subplots(figsize=(6, 3.6))
        ax1.plot(sn.index, sn.values, label=ticker, color="#2c6fbb", lw=1.5)
        ax1.plot(mn.index, mn.values, label="S&P 500", color="#7f8c8d", lw=1.5, ls="--")
        ax1.set_title("Crecimiento de $1 invertido"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
        g1.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 3.6))
        ax2.plot(roll_s.index, roll_s.values * 100, label=ticker, color="#2c6fbb", lw=1)
        ax2.plot(roll_m.index, roll_m.values * 100, label="S&P 500", color="#7f8c8d", lw=1, ls="--")
        ax2.axhline(0, color="k", lw=.8); ax2.set_title("Retorno móvil a 1 año (%)")
        ax2.legend(fontsize=8); ax2.grid(alpha=.3)
        g2.pyplot(fig2)

        st.warning("⚠️ Esto es el desempeño PASADO REAL de la acción vs el S&P 500 — NO prueba el "
                   "modelo de score, y el pasado no predice el futuro. Además tiene **survivorship "
                   "bias**: estás viendo una acción que sobrevivió; las que quebraron no aparecen. "
                   "Tómalo como historia, no como promesa.")

    st.divider()
    st.caption("Score RELATIVO al universo de pares, basado en datos actuales de Yahoo. "
               "Faltan DCF/valor intrínseco (próxima fase). NO es predicción ni recomendación de compra.")
else:
    st.info("Escribe un ticker y pulsa **Analizar**.")
