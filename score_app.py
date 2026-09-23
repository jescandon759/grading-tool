"""
Investment Score v2 - App web.
Ejecuta:  streamlit run score_app.py

Modulos:
  score_model.py  Investment Score (factores fundamentales) + backtest de momentum mensual
  data.py         descarga robusta de precios (lotes, reintentos, reporte de calidad) + S&P 500
  factors.py      senales cuantitativas (percentiles, sector-neutral)
  screener.py     screener semanal + validacion walk-forward semanal
No predice precios ni es asesoria financiera.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import factors as fx
import backtest as bt
import research as rs
import score_model as sm
import screener as sc

st.set_page_config(page_title="Investment Score", page_icon="🎯", layout="wide")
AZUL, VERDE, ROJO, NARANJA, GRIS = "#2c6fbb", "#2e8b57", "#c0392b", "#e67e22", "#8a8f98"
FACS = list(sm.FACTORES)
UNIVERSO_ORIGINAL = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
    "GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "AMZN", "TSLA", "HD", "MCD",
    "NKE", "SBUX", "LOW", "PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "AXP", "LLY", "UNH", "JNJ", "MRK",
    "ABBV", "PFE", "TMO", "CAT", "GE", "BA", "HON", "UPS", "RTX", "DE",
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "SHW", "FCX", "NEM", "APD",
    "NEE", "DUK", "SO", "D", "AEP", "PLD", "AMT", "EQIX", "SPG", "O",
]


class Alto(Exception):
    """Detiene solo la pestana actual (st.stop() detendria toda la app)."""


def alto(msg):
    st.error(msg)
    raise Alto


@contextmanager
def seccion(tab):
    with tab:
        try:
            yield
        except Alto:
            pass


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else "n/d"


def money(x, d=2):
    # "\$" evita que Streamlit lea "$215 – $405" como formula LaTeX
    return f"\\${x:,.{d}f}" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else "n/d"


# ================================================================ CACHE
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500():
    return data.sp500_universe()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_prices(tickers: tuple, period: str):
    return data.download_prices(list(tickers), period=period)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_infos(tickers: tuple):
    return sm.fetch_infos(list(tickers))


def get_universe(tickers: tuple):
    """Metricas de todo el universo: precios 2 anios + fundamentales. Devuelve (df, reporte)."""
    close, _, rep = get_prices(tickers, "2y")
    infos, bad = get_infos(tickers)
    rep = dict(rep); rep["sin_fundamentales"] = bad
    return sm.build_universe(infos, close), rep


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_fund(tickers: tuple):
    return data.fetch_fundamentals(list(tickers))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_earnings(tickers: tuple):
    return data.fetch_earnings_dates(list(tickers))


@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_quote(ticker):
    """Precio y estimados de analistas: cambian seguido, por eso cache de solo 15 min
    (los fundamentales para el score se guardan 24 h)."""
    return sm.fetch_infos([ticker])[0].get(ticker, {})


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_context(ticker):
    import yfinance as yf
    tk = yf.Ticker(ticker); titulares = []
    try:
        for n in (tk.news or [])[:10]:
            t = n.get("title") or (n.get("content") or {}).get("title")
            if t:
                titulares.append(t)
    except Exception:
        pass
    try:
        stmt = tk.income_stmt
    except Exception:
        stmt = None
    return titulares, stmt


def peers_for(ticker: str, info: dict) -> tuple[list, str]:
    """Pares del mismo sector: GICS si esta en el S&P 500, si no el sector de Yahoo."""
    u = get_sp500()
    if ticker in set(u["Ticker"]):
        sector = u.set_index("Ticker").loc[ticker, "Sector"]
    else:
        sector = sm.YAHOO_SECTOR.get(info.get("sector"), None)
    if not sector:
        return UNIVERSO_ORIGINAL, "lista original (sector desconocido)"
    return list(u.loc[u["Sector"] == sector, "Ticker"]), sector


def reporte_calidad(rep: dict, total: int):
    bad, hue = rep.get("fallidos", []), rep.get("con_huecos", {})
    sinf = rep.get("sin_fundamentales", [])
    n_bad = len(set(bad) | set(sinf))
    with st.expander(("⚠️" if n_bad else "✅") + f" Calidad de datos: {total - n_bad}/{total} completos"
                     + (f" · {len(hue)} con huecos" if hue else "")):
        if bad: st.write("**Sin precios:** " + ", ".join(bad))
        if sinf: st.write("**Sin fundamentales:** " + ", ".join(sinf))
        if hue: st.write("**Huecos >2% en precios:** " + ", ".join(f"{k} ({v}%)" for k, v in hue.items()))
        if not (bad or sinf or hue): st.write("Todo completo.")


def progress_cols(cols):
    return {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.0f") for c in cols}


def sentimiento(titulares):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    POS = {"surge", "surges", "soar", "soars", "jump", "jumps", "rally", "beat", "beats", "tops",
           "upgrade", "upgraded", "raises", "record", "strong", "growth", "gains", "rises", "boost",
           "outperform", "buy", "bullish", "wins", "expands", "profit", "rebound"}
    NEG = {"probe", "antitrust", "lawsuit", "sue", "sued", "plunge", "plunges", "miss", "misses",
           "downgrade", "downgraded", "cuts", "cut", "falls", "fall", "drop", "drops", "slump", "fraud",
           "warning", "weak", "layoffs", "bearish", "recall", "decline", "loss", "sinks", "tumbles"}
    sia = SentimentIntensityAnalyzer(); out = []
    for t in titulares:
        w = set(t.lower().replace(",", " ").replace(".", " ").split())
        c = max(-1, min(1, sia.polarity_scores(t)["compound"] + 0.35 * (len(w & POS) - len(w & NEG))))
        out.append((t, c, "🟢 Positivo" if c > 0.1 else "🔴 Negativo" if c < -0.1 else "⚪ Neutral"))
    return out


def backtest_stock(s, m, rf=0.04):
    idx = s.dropna().index.intersection(m.dropna().index)
    s, m = s.loc[idx], m.loc[idx]
    if len(s) < 260:
        return None
    cagr = lambda p: (p.iloc[-1] / p.iloc[0]) ** (252 / len(p)) - 1
    rs, rm = s.pct_change().dropna(), m.pct_change().dropna()
    vol = lambda r: r.std() * np.sqrt(252)
    beta = rs.cov(rm) / rm.var() if rm.var() > 0 else np.nan
    cs, cm = cagr(s), cagr(m)
    down = rs[rs < 0].std() * np.sqrt(252)
    roll_s, roll_m = (s / s.shift(252) - 1).dropna(), (m / m.shift(252) - 1).dropna()
    exc = roll_s - roll_m.reindex(roll_s.index)
    return {"years": len(s) / 252, "cagr_s": cs, "cagr_m": cm, "vol_s": vol(rs), "vol_m": vol(rm),
            "sharpe_s": (cs - rf) / vol(rs), "sharpe_m": (cm - rf) / vol(rm),
            "mdd_s": (s / s.cummax() - 1).min(), "mdd_m": (m / m.cummax() - 1).min(),
            "avg1y_s": roll_s.mean(), "avg_excess": exc.mean(), "win_rate": (exc > 0).mean() * 100,
            "beta": beta, "alpha": cs - (rf + beta * (cm - rf)) if pd.notna(beta) else np.nan,
            "sortino": (cs - rf) / down if down > 0 else np.nan,
            "best1y": roll_s.max(), "worst1y": roll_s.min()}, s / s.iloc[0], m / m.iloc[0]


# ================================================================ UI
st.title("🎯 Investment Score")
u = get_sp500()
tab_a, tab_r, tab_s, tab_b, tab_m = st.tabs(
    ["🎯 Analizar acción", "🏆 Ranking de pares", "📅 Screener semanal", "⏳ Backtests", "ℹ️ Metodología"])

# ---------------------------------------------------------------- 1. ANALIZAR
with seccion(tab_a):
    st.caption("Califica una acción de 0 a 100 por factores, contra empresas de su mismo sector. "
               "No predice precios ni es asesoría financiera.")
    c1, c2 = st.columns([2, 1])
    ticker = data.yahoo_symbol(c1.text_input("Ticker (EE.UU.)", "AAPL") or "")
    modo = c2.radio("Comparar contra", ["Su sector (S&P 500)", "Lista personalizada"], horizontal=False)
    uni_txt = ""
    if modo == "Lista personalizada":
        uni_txt = st.text_area("Tickers de pares (coma)", ", ".join(UNIVERSO_ORIGINAL), height=90)

    if st.button("🎯 Analizar", type="primary", width="stretch"):
        if not ticker:
            alto("Escribe un ticker.")
        with st.spinner("Buscando la empresa..."):
            infos1, _ = get_infos((ticker,))
        if ticker not in infos1:
            alto(f"Yahoo no devolvió datos de {ticker}. Revisa el ticker o intenta en un minuto.")
        if modo.startswith("Su sector"):
            peers, etiqueta = peers_for(ticker, infos1[ticker])
        else:
            peers, etiqueta = data.parse_tickers(uni_txt), "lista personalizada"
        universo = tuple(sorted(set(peers) | {ticker}))
        with st.spinner(f"Descargando {len(universo)} empresas ({etiqueta})... ~30-60 s la 1ª vez"):
            uni, rep = get_universe(universo)
        st.session_state["an"] = dict(ticker=ticker, uni=uni, rep=rep, etiqueta=etiqueta, n=len(universo))

    A = st.session_state.get("an")
    if A:
        ticker, uni = A["ticker"], A["uni"]
        reporte_calidad(A["rep"], A["n"])
        if ticker not in uni.index:
            alto("Sin datos de la empresa.")
        met = uni.loc[ticker].to_dict()
        fs = sm.factor_scores(met, uni)
        score, conf = sm.investment_score(fs), sm.confianza(fs)
        pos, rie, fuertes, debiles = sm.explicar(fs)
        reco = sm.recomendacion(score)
        ranking = sm.rank_universe(uni)
        puesto = int(ranking.index.get_loc(ticker)) + 1
        color = {"STRONG BUY": "🟢", "BUY": "🟢", "WEAK BUY / WATCHLIST": "🟡",
                 "HOLD": "🟡", "WEAK HOLD": "🟠", "AVOID": "🔴"}.get(reco, "⚪")

        st.markdown(f"## {met.get('_name', ticker)} · `{ticker}`")
        st.caption(f"Comparada contra **{len(uni)} empresas** ({A['etiqueta']}) · Precio {money(met.get('_price'))}")
        m = st.columns(5)
        m[0].metric("Investment Score", f"{score:.0f} / 100" if pd.notna(score) else "N/A")
        m[1].metric("Lugar entre sus pares", f"{puesto} de {len(ranking)}")
        m[2].metric("Lectura", f"{color} {reco}")
        m[3].metric("Nivel de riesgo", sm.nivel_riesgo(fs["Risk"]["score"]))
        m[4].metric("Confianza (datos)", f"{conf:.0f}%")
        if conf < 60:
            st.warning("Confianza baja: faltan datos. Interpreta el score con cautela.")

        g1, g2 = st.columns([1.2, 1])
        vals = [fs[f]["score"] for f in FACS]
        med = [ranking[f].median() for f in FACS]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=FACS, x=[v if pd.notna(v) else 0 for v in vals], orientation="h", name=ticker,
                             marker_color=[VERDE if (v or 0) >= 66 else NARANJA if (v or 0) >= 33 else ROJO for v in vals],
                             text=[f"{v:.0f}" if pd.notna(v) else "n/d" for v in vals], textposition="outside"))
        fig.add_trace(go.Scatter(y=FACS, x=med, mode="markers", name="Mediana de pares",
                                 marker=dict(symbol="line-ns-open", size=22, color="black", line_width=2)))
        fig.update_layout(title="Desglose por factores (percentil vs pares)", xaxis_range=[0, 105],
                          height=360, margin=dict(t=50, b=10), yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h"))
        g1.plotly_chart(fig, width="stretch")
        with g2:
            st.markdown("**✅ Factores más fuertes**")
            for f in pos or ["(ninguno destaca)"]: st.write(f"- {f}")
            if fuertes: st.caption("Destaca en: " + ", ".join(fuertes))
            st.markdown("**⚠️ Principales riesgos**")
            for f in rie or ["(ninguno crítico)"]: st.write(f"- {f}")
            if debiles: st.caption("Más débil en: " + ", ".join(debiles))

        with st.expander("Ver cada métrica (valor y percentil)"):
            filas = [{"Factor": f, "Métrica": sm.ETIQUETAS.get(k, k), "Valor": v, "Percentil": p}
                     for f in FACS for k, v, p in fs[f]["det"]]
            st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True,
                         column_config={**progress_cols(["Percentil"]),
                                        "Valor": st.column_config.NumberColumn(format="%.3f")})

        st.subheader("🏆 Cómo se ve frente a sus pares")
        top = ranking.head(10)
        if ticker not in top.index:
            top = pd.concat([top, ranking.loc[[ticker]]])
        cols = ["Empresa", "Score", "Momentum", *FACS, "Confianza"]
        st.dataframe(top[cols].style.apply(
            lambda r: ["background-color: rgba(44,111,187,.18)" if r.name == ticker else "" for _ in r], axis=1),
            width="stretch", column_config=progress_cols(["Score", "Momentum", *FACS, "Confianza"]))

        st.divider()
        st.subheader("📈 Momentum (¿buen momento?)")
        msc = ranking.loc[ticker, "Momentum"]
        if pd.notna(msc) and pd.notna(met.get("ret_12m")):
            mc1, mc2 = st.columns([1, 2])
            mc1.metric("Momentum Score", f"{msc:.0f} / 100",
                       "fuerte" if msc >= 60 else "débil" if msc < 40 else "neutral", delta_color="off")
            mc2.write(f"1m {pct(met.get('ret_1m'))} · 3m {pct(met.get('ret_3m'))} · 6m {pct(met.get('ret_6m'))} · "
                      f"12m {pct(met.get('ret_12m'))}  \nvs SMA50 {pct(met.get('dist_sma50'))} · "
                      f"vs SMA200 {pct(met.get('dist_sma200'))} · volatilidad 3m {pct(met.get('mom_vol'))}")
            fb, mb, mm = pd.notna(score) and score >= 52, msc >= 55, msc < 45
            if fb and mb: st.success("🟢 **Buena empresa Y buen momento.**")
            elif fb and mm: st.warning("🟡 **Buena empresa, mal momento.** Viene cayendo: oportunidad o alerta temprana.")
            elif (not fb) and mb: st.warning("🟠 **Buen momento, fundamentales flojos.** Cuidado con perseguir el rally.")
            elif not fb and mm: st.error("🔴 **Ni fundamentales ni momento.**")
            else: st.info("⚪ Señales mixtas.")
            st.caption("El momentum no entra al Investment Score: es una capa de timing aparte. "
                       "Revisa en ⏳ Backtests si esta señal ha funcionado de verdad.")
        else:
            st.info("No hay suficiente historia de precios.")

        st.subheader("📊 Lo que esperan los analistas")
        info = get_quote(ticker)
        st.caption(f"Datos de analistas actualizados cada 15 min · consulta: {pd.Timestamp.now(tz='America/Mexico_City'):%d-%b %H:%M}")
        tm, cur = info.get("targetMeanPrice"), info.get("currentPrice") or info.get("regularMarketPrice")
        lo, hi, na = info.get("targetLowPrice"), info.get("targetHighPrice"), info.get("numberOfAnalystOpinions")
        if tm and cur:
            up = tm / cur - 1
            a = st.columns(4)
            a[0].metric("Precio actual", money(cur)); a[1].metric("Objetivo a 12 meses (prom.)", money(tm), pct(up),
                         help="Promedio de los precios objetivo de los analistas, a 12 meses desde que cada uno lo publicó.")
            a[2].metric("Rango objetivo", f"{money(lo, 0)} – {money(hi, 0)}")
            a[3].metric("Recomendación", str(info.get("recommendationKey") or "n/d").upper())
            if lo and hi:
                disp = (hi - lo) / cur
                st.write(f"Rango de {pct(lo / cur - 1, 0)} a {pct(hi / cur - 1, 0)} → "
                         + ("poco acuerdo" if disp > .6 else "acuerdo medio" if disp > .25 else "bastante acuerdo")
                         + f" · {na or '?'} analistas.")
            if up < 0: st.error("🔴 Objetivo POR DEBAJO del precio: con el sesgo optimista de los analistas, es mala señal.")
            elif up < .10: st.warning(f"🟠 Upside tibio ({pct(up, 0)}): por el sesgo optimista del gremio, es señal débil.")
            else: st.info("🟡 Upside positivo es lo normal (sesgo optimista); no es señal especial.")
        else:
            st.info("Sin estimaciones de analistas.")

        titulares, stmt = fetch_context(ticker)
        st.subheader("📰 Termómetro de noticias")
        sent = sentimiento(titulares) if titulares else None
        if sent:
            prom = np.mean([c for _, c, _ in sent])
            st.write("Titulares recientes: **" + ("🟢 mayormente positivos" if prom > .1 else
                     "🔴 mayormente negativos" if prom < -.1 else "⚪ mixtos") + "**")
            for t, c, et in sent: st.write(f"{et} · {t}")
            st.caption("Señal débil y ruidosa (el mercado descuenta la noticia en segundos). No entra al score.")
        else:
            st.info("Sin noticias disponibles.")
        st.markdown(f"[Noticias en Yahoo](https://finance.yahoo.com/quote/{ticker}/news) · "
                    f"[Estados financieros](https://finance.yahoo.com/quote/{ticker}/financials) · "
                    f"[Reportes SEC](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}&type=10-&dateb=&owner=include&count=40)")

        st.subheader("📄 Estados financieros anuales")
        if stmt is not None and not getattr(stmt, "empty", True) and "Total Revenue" in stmt.index:
            rev = stmt.loc["Total Revenue"]; ni = stmt.loc["Net Income"] if "Net Income" in stmt.index else rev * np.nan
            tf = pd.DataFrame({"Año": [c.year for c in stmt.columns], "Ventas": rev.values / 1e9,
                               "Utilidad neta": ni.values / 1e9}).sort_values("Año")
            tf["Margen neto"] = tf["Utilidad neta"] / tf["Ventas"]; tf["Crec. ventas"] = tf["Ventas"].pct_change()
            st.dataframe(tf.style.format({"Ventas": "${:,.1f}B", "Utilidad neta": "${:,.1f}B",
                                          "Margen neto": "{:.1%}", "Crec. ventas": "{:+.1%}"}, na_rep="—"),
                         width="stretch", hide_index=True)
        else:
            st.info("Sin estados financieros anuales.")

        st.subheader("⏳ Historia de la acción vs S&P 500 (10 años)")
        px10, _, _ = get_prices((ticker, "^GSPC"), "10y")
        hist = backtest_stock(px10[ticker], px10["^GSPC"]) if {ticker, "^GSPC"} <= set(px10.columns) else None
        if hist:
            o, sn, mn = hist
            b = st.columns(4)
            b[0].metric("CAGR", pct(o["cagr_s"]), f"{pct(o['cagr_s'] - o['cagr_m'])} vs S&P")
            b[1].metric("Le gana al S&P (ventanas 1 año)", f"{o['win_rate']:.0f}%")
            b[2].metric("Sharpe", f"{o['sharpe_s']:.2f}", f"S&P {o['sharpe_m']:.2f}", delta_color="off")
            b[3].metric("Peor caída", pct(o["mdd_s"], 0), f"S&P {pct(o['mdd_m'], 0)}", delta_color="off")
            b = st.columns(4)
            b[0].metric("Alpha anual", pct(o["alpha"])); b[1].metric("Beta", f"{o['beta']:.2f}")
            b[2].metric("Sortino", f"{o['sortino']:.2f}"); b[3].metric("Mejor / peor año", f"{pct(o['best1y'], 0)} / {pct(o['worst1y'], 0)}")
            h1, h2 = st.columns(2)
            fig = go.Figure([go.Scatter(x=sn.index, y=sn, name=ticker, line_color=AZUL),
                             go.Scatter(x=mn.index, y=mn, name="S&P 500", line=dict(color=GRIS, dash="dash"))])
            fig.update_layout(title="¿En cuánto se convirtió $1?", height=360, margin=dict(t=50, b=10),
                              legend=dict(orientation="h"))
            h1.plotly_chart(fig, width="stretch")
            g = sn.groupby(sn.index.year); ya = g.last() / g.first() - 1
            fig = px.bar(x=ya.index.astype(str), y=ya.values, title=f"Rendimiento por año de {ticker}",
                         labels={"x": "", "y": ""}, color=ya.values > 0,
                         color_discrete_map={True: VERDE, False: ROJO})
            fig.update_yaxes(tickformat=".0%"); fig.update_layout(height=360, showlegend=False, margin=dict(t=50, b=10))
            h2.plotly_chart(fig, width="stretch")
            st.caption("Desempeño PASADO de esta acción; no prueba el score. Sesgo de supervivencia: es una acción que sobrevivió.")
        else:
            st.info("No hay suficiente historia de precios.")

# ---------------------------------------------------------------- 2. RANKING
with seccion(tab_r):
    st.caption("Investment Score de todas las empresas de un sector o lista: encuentra las mejores directamente.")
    r1, r2 = st.columns(2)
    alcance = r1.selectbox("Universo", ["Un sector del S&P 500", "Lista original (71)", "S&P 500 completo (5-8 min 1ª vez)"])
    sector_sel = r2.selectbox("Sector", sorted(u["Sector"].unique()), disabled=not alcance.startswith("Un sector"))
    if st.button("🏆 Calcular ranking", type="primary", width="stretch"):
        tick = (list(u.loc[u["Sector"] == sector_sel, "Ticker"]) if alcance.startswith("Un sector")
                else UNIVERSO_ORIGINAL if alcance.startswith("Lista") else list(u["Ticker"]))
        with st.spinner(f"Descargando {len(tick)} empresas..."):
            uni, rep = get_universe(tuple(sorted(tick)))
        rk = sm.rank_universe(uni)
        if alcance.startswith("S&P"):   # universo mixto: re-calcular por sector para comparar peras con peras
            partes = [sm.rank_universe(uni[uni["_sector"] == s]) for s in uni["_sector"].unique()
                      if (uni["_sector"] == s).sum() >= 8]
            rk = pd.concat(partes).sort_values("Score", ascending=False) if partes else rk
        st.session_state["rk"] = dict(rk=rk, rep=rep, n=len(tick))
    R = st.session_state.get("rk")
    if R:
        reporte_calidad(R["rep"], R["n"])
        rk = R["rk"]
        mc = st.slider("Confianza mínima de datos (%)", 0, 100, 60, 5)
        v = rk[rk["Confianza"] >= mc]
        cols = ["Empresa", "Sector", "Score", "Recomendación", "Momentum", *FACS, "Confianza"]
        st.dataframe(v[cols], width="stretch", height=560,
                     column_config=progress_cols(["Score", "Momentum", *FACS, "Confianza"]))
        fig = px.scatter(v.reset_index(), x="Score", y="Momentum", hover_name="index", color="Sector",
                         hover_data=["Empresa"], title="Fundamentales (Score) vs momentum")
        fig.add_hline(y=50, line_dash="dot", line_color=GRIS); fig.add_vline(x=50, line_dash="dot", line_color=GRIS)
        fig.update_layout(height=460, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Arriba a la derecha = buena empresa y buen momento. Descarga para seguir investigando.")
        st.download_button("⬇️ Descargar CSV", v.to_csv().encode(), "ranking_investment_score.csv", "text/csv")

# ---------------------------------------------------------------- 3. SCREENER SEMANAL
DEFAULT_EXTRA = "SOFI, RKLB, IONQ, CELH, DUOL, AFRM, HIMS, ASTS, CRDO, OKLO, NU, MELI"
with seccion(tab_s):
    st.caption("Ranking semanal: S&P 500 + tu lista fuera del índice. Estrategias: momentum, reversión, "
               "multifactor y eventos. Sin tamaño de posición: tú decides.")
    a1, a2 = st.columns([2, 1])
    extra_txt = a1.text_area("Acciones fuera del S&P 500 (coma)", DEFAULT_EXTRA, height=80)
    usar_sp = a2.checkbox("Incluir S&P 500", True)
    n_deep = a2.slider("Candidatas para estudio profundo", 30, 150, 80, 10,
                       help="A estas se les bajan fundamentales y fecha de reporte (lo lento).")
    with st.expander("Pesos por estrategia y filtros"):
        bcols = st.columns(4)
        w = {k: bcols[i].slider(k, 0, 100, int(sc.PESOS_DEFAULT[k] * 100), 5, key=f"w_{k}")
             for i, k in enumerate(sc.ESTRATEGIAS)}
        f1, f2 = st.columns(2)
        min_dv = f1.number_input("Volumen diario mínimo (USD millones)", 0.0, 500.0, 10.0, 1.0)
        min_px = f2.number_input("Precio mínimo (USD)", 0.0, 100.0, 5.0, 1.0)
    if st.button("📅 Correr screener de la semana", type="primary", width="stretch"):
        tick = list(dict.fromkeys((list(u["Ticker"]) if usar_sp else []) + data.parse_tickers(extra_txt)))
        sectors = u.set_index("Ticker")["Sector"].reindex(tick).fillna("Fuera del S&P")
        with st.spinner(f"Descargando precios de {len(tick)} acciones (~30-90 s la 1ª vez)..."):
            close, vol, rep = get_prices(tuple(tick), "2y")
        if close.empty:
            alto("Yahoo no devolvió datos. Intenta en unos minutos.")
        liq = sc.liquidity_filter(close, vol, min_dv * 1e6, min_px)
        snap = fx.snapshot(fx.price_signals(close[liq], vol.reindex(columns=liq) if not vol.empty else None))
        cand = sc.price_prescreen(snap, sectors, n_deep)
        with st.spinner(f"Fundamentales y fechas de reporte de {len(cand)} candidatas..."):
            fund, fund_bad = get_fund(tuple(sorted(cand)))
            earn = get_earnings(tuple(sorted(cand)))
        tw = sum(w.values()) or 1
        rank, det = sc.weekly_ranking(snap, sectors, fund, earn, {k: v / tw for k, v in w.items()})
        rank = rank.loc[[t for t in rank.index if t in cand]]; rank["Rank"] = range(1, len(rank) + 1)
        st.session_state["scr"] = dict(rank=rank, det=det, close=close, fund=fund, rep=rep, n=len(tick),
                                       liq=len(liq), fund_bad=fund_bad, fecha=close.index[-1], w=w)
    S = st.session_state.get("scr")
    if S:
        reporte_calidad(S["rep"], S["n"])
        st.caption(f"Cierre del {S['fecha']:%d-%b-%Y} · {S['liq']} pasan liquidez · {len(S['rank'])} estudiadas a fondo")
        fil = st.multiselect("Filtrar por señal principal", sc.ESTRATEGIAS, default=sc.ESTRATEGIAS)
        view = S["rank"][S["rank"]["Senal principal"].isin(fil)]
        cols = [c for c in ["Rank", "Sector", "Score", "Senal principal", *sc.ESTRATEGIAS, "Rend 1 sem",
                            "Rend 3m", "Volatilidad", "Tendencia", "Reporta en (dias)"] if c in view]
        cfg = progress_cols(["Score", *sc.ESTRATEGIAS])
        cfg.update({c: st.column_config.NumberColumn(c, format="percent") for c in ["Rend 1 sem", "Rend 3m", "Volatilidad"]})
        st.dataframe(view[cols].head(30), width="stretch", height=560, column_config=cfg)
        st.download_button("⬇️ Ranking completo (CSV)", S["rank"].to_csv().encode(),
                           f"screener_{S['fecha']:%Y%m%d}.csv", "text/csv")
        wv = S["w"]; top15 = view.head(15)
        contrib = top15[sc.ESTRATEGIAS].fillna(0).mul(pd.Series(wv) / (sum(wv.values()) or 1), axis=1)
        fig = px.bar(contrib.reset_index().melt(id_vars="index"), x="index", y="value", color="variable",
                     labels={"index": "", "value": "Aporte al score", "variable": "Estrategia"},
                     title="¿De dónde sale el score? (Top 15)")
        fig.update_layout(height=360, margin=dict(t=50, b=10)); st.plotly_chart(fig, width="stretch")
        sel = st.selectbox("🔎 Investigar una acción", list(view.index), key="scr_sel")
        if sel:
            r = S["rank"].loc[sel]
            d1, d2 = st.columns([1, 1.4])
            with d1:
                st.write(f"**{sel}** · Rank {int(r['Rank'])} · Score {r['Score']:.0f} · Señal **{r['Senal principal']}**")
                dd = S["det"].loc[sel].dropna()
                fig = px.bar(x=dd.values, y=dd.index, orientation="h", range_x=[0, 100],
                             labels={"x": "Percentil", "y": ""}, title="Desglose")
                fig.update_layout(height=max(300, 22 * len(dd)), margin=dict(t=40, b=10))
                st.plotly_chart(fig, width="stretch")
                if pd.notna(r.get("Reporta en (dias)", np.nan)):
                    st.warning(f"📣 Reporta en {int(r['Reporta en (dias)'])} días.")
                st.caption("Para el Investment Score completo, analízala en la pestaña 🎯.")
            with d2:
                full = S["close"][sel].dropna(); p = full.iloc[-260:]
                fig = go.Figure([go.Scatter(x=p.index, y=p, name="Precio", line_color=AZUL)] +
                                [go.Scatter(x=p.index, y=full.rolling(n_).mean().reindex(p.index), name=f"Media {n_}d",
                                            line=dict(color=c_, dash="dot")) for n_, c_ in [(50, NARANJA), (200, GRIS)]])
                fig.update_layout(title=f"{sel} · último año", height=400, margin=dict(t=50, b=10),
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 4. BACKTESTS
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def bt_data(years: int):
    """Mismo universo y datos para A y B: todos los que fueron miembros del S&P 500 en el periodo
    (point-in-time) + SPY con dividendos como benchmark."""
    inicio = pd.Timestamp.today() - pd.DateOffset(years=years)
    tick = data.members_between(inicio)
    close, vol, rep = data.download_prices(tick + ["SPY"], period=f"{years + 1}y")
    if "SPY" not in close:
        raise ValueError("Yahoo no devolvió SPY (benchmark). Intenta en unos minutos.")
    m = data.sp500_membership()
    ex = set(m.loc[m["end"].notna() & (m["end"] > inicio), "ticker"])
    cob = {"total": len(tick), "con_datos": len([t for t in tick if t in close]),
           "ex_miembros": len(ex), "ex_con_datos": len([t for t in ex if t in close])}
    return close, vol, rep, cob


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_a(years: int, top_n: int, cost: float):
    close, vol, rep, cob = bt_data(years)
    bench = close["SPY"]; px_ = close.drop(columns="SPY")
    mask_fn = lambda d: data.membership_mask(d, px_.columns)
    return rs.study_a(px_, bench, mask_fn, u.set_index("Ticker")["Sector"], top_n, cost), cob


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_b(years: int, top_n: int, cost: float):
    close, vol, rep, cob = bt_data(years)
    bench = close["SPY"]; px_ = close.drop(columns="SPY")
    v = vol.drop(columns="SPY", errors="ignore") if not vol.empty else None
    mask_fn = lambda d: data.membership_mask(d, px_.columns)
    return rs.study_b(px_, v, bench, mask_fn, u.set_index("Ticker")["Sector"], top_n, cost), cob


def show_check(check, ver):
    lab, col = ver
    {"green": st.success, "orange": st.warning, "red": st.error}[col](
        f"**Veredicto: {lab}** · cumple {sum(ok for *_, ok in check)} de {len(check)} condiciones")
    st.dataframe(pd.DataFrame([{"": "✅" if ok else "❌", "Condición": c, "Resultado": v} for c, v, ok in check]),
                 width="stretch", hide_index=True)


FMT = {"IC promedio": "{:.3f}", "IC t-stat": "{:.2f}", "% periodos IC>0": "{:.0%}", "Spread Q5-Q1 anual": "{:+.1%}",
       "Monotonía quintiles": "{:.2f}", "Top N anual (neto)": "{:.1%}", "Universo anual": "{:.1%}",
       "SPY anual": "{:.1%}", "Exceso vs universo": "{:+.1%}", "Exceso vs SPY": "{:+.1%}",
       "Sharpe (neto)": "{:.2f}", "Max caída": "{:.0%}", "Rotación por periodo": "{:.0%}",
       "% periodos gana al universo": "{:.0%}", "Periodos": "{:.0f}"}


def curva(r, titulo):
    c = pd.DataFrame({"Top N (neto)": (1 + r["Top N neto"]).cumprod(), "Universo (igual peso)": (1 + r["Universo"]).cumprod(),
                      "SPY (con dividendos)": (1 + r["Benchmark"]).cumprod()})
    fig = px.line(c, title=titulo, color_discrete_sequence=[AZUL, GRIS, NARANJA])
    fig.update_layout(height=360, margin=dict(t=50, b=10), legend=dict(orientation="h"), yaxis_title="", xaxis_title="")
    return fig


def quintiles(r, ppy):
    q = r[[f"Q{i}" for i in range(1, 6)]].mean() * ppy
    fig = px.bar(x=["Q1 peor", "Q2", "Q3", "Q4", "Q5 mejor"], y=q.values, title="Rendimiento anual por quintil del score",
                 color=q.values, color_continuous_scale="RdYlGn", labels={"x": "", "y": ""})
    fig.update_yaxes(tickformat=".0%"); fig.update_layout(height=360, margin=dict(t=50, b=10), coloraxis_showscale=False)
    return fig


def heat(g, titulo):
    fig = px.imshow(g, text_auto=".1%", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    title=titulo, aspect="auto")
    fig.update_layout(height=300, margin=dict(t=50, b=10), coloraxis_showscale=False)
    return fig


def show_alpha(al, al_mkt=None):
    cols = st.columns(3)
    if al_mkt:
        cols[0].metric("Alpha vs mercado", pct(al_mkt.get("Alpha anual")), f"t = {al_mkt.get('Alpha t-stat', np.nan):.2f}",
                       delta_color="off", help="Rendimiento propio después de quitar la exposición al S&P 500 (beta).")
    if al:
        cols[1].metric("Alpha vs mercado + momentum clásico", pct(al.get("Alpha anual")),
                       f"t = {al.get('Alpha t-stat', np.nan):.2f}", delta_color="off",
                       help="Si se va a ~0, el modelo solo 'empaqueta' el momentum clásico 12-1.")
        cols[2].metric("Beta de mercado", f"{al.get('Beta Mercado (SPY)', np.nan):.2f}")


with seccion(tab_b):
    st.info("**Mismo pipeline para A y B:** universo histórico del S&P 500 (solo cuentan las empresas que eran "
            "miembros en cada fecha) · la señal se calcula al cierre y se opera al **día siguiente** · benchmark "
            "**SPY con dividendos** · costos sobre la rotación real · el t exigido sube con el número de modelos probados. "
            "Solo se prueban señales de PRECIO (los fundamentales de Yahoo son los de hoy).")
    k1, k2 = st.columns(2)
    years = k1.selectbox("Años de prueba", [5, 10], 1, help="Se descarga 1 año extra para calcular señales.")
    cost = float(k2.slider("Costo + slippage por operación (pb)", 0, 50, 10, 5, help="10 pb = 0.10% por compra o venta."))
    ta, tb = st.tabs(["A) Momentum mensual del score", "B) Estrategias del screener semanal"])

    with ta:
        st.caption("Pregunta: **¿puedo construir un portafolio con este ranking?** Cada ~21 días compra el Top N por "
                   "momentum, igual peso, y lo mantiene hasta el siguiente rebalanceo.")
        top_a = st.slider("Top N", 5, 30, 10, 5, key="topa")
        if st.button("⏳ Correr estudio A (2-5 min la 1ª vez)", width="stretch"):
            with st.spinner("Descargando historia del universo point-in-time y corriendo pruebas..."):
                try:
                    st.session_state["A"] = run_a(years, top_a, cost)
                except Exception as ex:
                    alto(f"No se pudo correr el estudio A: {ex}")
        if "A" in st.session_state:
            A, cob = st.session_state["A"]; ppy = A["ppy"]
            st.caption(f"Cobertura: {cob['con_datos']}/{cob['total']} tickers con precios · ex-miembros con datos: "
                       f"{cob['ex_con_datos']}/{cob['ex_miembros']} (los que faltan suelen ser quiebras o adquisiciones: "
                       "el sesgo de supervivencia se reduce, no desaparece).")
            show_check(A["check"], A["veredicto"])
            m = bt.summarize(A["main"], ppy)
            k = st.columns(5)
            k[0].metric("Top N anual (neto)", pct(m["Top N anual (neto)"]))
            k[1].metric("vs universo", pct(m["Exceso vs universo"])); k[2].metric("vs SPY", pct(m["Exceso vs SPY"]))
            k[3].metric("Sharpe neto", f"{m['Sharpe (neto)']:.2f}"); k[4].metric("Peor caída", pct(m["Max caída"], 0))
            g1, g2 = st.columns(2)
            g1.plotly_chart(curva(A["main"], "App actual: crecimiento de $1"), width="stretch")
            g2.plotly_chart(quintiles(A["main"], ppy), width="stretch")

            st.subheader("1. Fuera de muestra (walk-forward anidado)")
            st.caption("Cada año elige el mejor modelo con los 5 años previos y lo prueba en el año siguiente, que no vio.")
            if not A["wf"].empty:
                st.dataframe(A["wf"].style.format({"Exceso en entrenamiento": "{:+.1%}", "Exceso fuera de muestra": "{:+.1%}"}),
                             width="stretch", hide_index=True)
                st.write(f"**Exceso anual fuera de muestra: {pct(A['oos_exc'])}** vs universo.")

            st.subheader("2. ¿Qué aporta cada bloque? (ablación)")
            st.dataframe(A["ablacion"][["IC promedio", "IC t-stat", "Spread Q5-Q1 anual", "Exceso vs universo",
                                        "Sharpe (neto)", "Max caída"]].style.format(FMT), width="stretch")
            st.caption(f"Se compararon {len(rs.CONFIGS_A)} modelos: el t exigido sube a {A['t_crit']:.2f}.")
            fig = px.imshow(A["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlación entre señales (1 = dicen lo mismo)")
            fig.update_layout(height=420, margin=dict(t=50, b=10)); st.plotly_chart(fig, width="stretch")

            st.subheader("3. Robustez")
            r1, r2 = st.columns([2, 1])
            if not A["grid"].empty:
                r1.plotly_chart(heat(A["grid"], f"Exceso anual vs universo · {A['grid_pos']:.0%} de variantes positivas"),
                                width="stretch")
            r2.dataframe(A["costos"].to_frame().style.format("{:+.1%}"), width="stretch")
            r2.caption("Estrés de costos (pb)")
            st.dataframe(A["yearly"].style.format("{:+.1%}").format({"IC promedio": "{:.3f}"}), width="stretch")

            st.subheader("4. ¿Alpha propio o factores conocidos?")
            show_alpha(A["alpha"], A["alpha_mkt"])
            n1, n2 = st.columns(2)
            n1.metric("Concentración sectorial del Top N", f"{A['conc_sector']:.0%}",
                      help="% promedio del portafolio en su sector más repetido. Alto = apuesta sectorial.")
            n2.metric("Versión sector-neutral: exceso", pct(A["neutral"].get("Exceso vs universo")),
                      help="Mismo score pero comparando cada acción solo contra su sector.")

    with tb:
        st.caption("Pregunta: **¿el score contiene información sobre el rendimiento futuro?** Cada viernes rankea, "
                   "opera el lunes y mide la semana.")
        top_b = st.slider("Top N", 10, 50, 20, 5, key="topb")
        if st.button("🧪 Correr estudio B (3-8 min la 1ª vez)", width="stretch"):
            with st.spinner("Calculando señales semana por semana..."):
                try:
                    st.session_state["B"] = run_b(years, top_b, cost)
                except Exception as ex:
                    alto(f"No se pudo correr el estudio B: {ex}")
        if "B" in st.session_state:
            B, cob = st.session_state["B"]; ppy = B["ppy"]
            filas = []
            for k, e in B["por_estrategia"].items():
                filas.append({"Estrategia": k, "Veredicto": e["veredicto"][0],
                              "Condiciones": f"{sum(ok for *_, ok in e['check'])}/{len(e['check'])}", **e["resumen"]})
            tabla = pd.DataFrame(filas).set_index("Estrategia")
            st.dataframe(tabla[["Veredicto", "Condiciones", "IC promedio", "IC t-stat", "Spread Q5-Q1 anual",
                                "Monotonía quintiles", "Exceso vs universo", "Sharpe (neto)", "Max caída"]]
                         .style.format(FMT), width="stretch")
            if not B["wf"].empty:
                with st.expander(f"Sistema completo fuera de muestra (elige cada año la mejor estrategia): "
                                 f"{pct(B['oos_sistema'].get('Exceso vs universo'))} anual"):
                    st.dataframe(B["wf"].style.format({"Exceso en entrenamiento": "{:+.1%}",
                                                       "Exceso fuera de muestra": "{:+.1%}"}),
                                 width="stretch", hide_index=True)
            est = st.selectbox("Detalle de", list(B["por_estrategia"]))
            e = B["por_estrategia"][est]; r = B["res"][est]
            show_check(e["check"], e["veredicto"])
            st.caption("En B, 'fuera de muestra' = segunda mitad del periodo (las reglas no se ajustaron con esos datos).")
            g1, g2 = st.columns(2)
            g1.plotly_chart(curva(r, f"{est}: crecimiento de $1"), width="stretch")
            g2.plotly_chart(quintiles(r, ppy), width="stretch")
            h1, h2 = st.columns([2, 1])
            if not e["grid"].empty:
                h1.plotly_chart(heat(e["grid"], f"Robustez: {e['grid_pos']:.0%} de variantes positivas"), width="stretch")
            h2.metric("Concentración sectorial", f"{e['conc_sector']:.0%}")
            show_alpha(e["alpha"])
            st.dataframe(e["yearly"].style.format("{:+.1%}").format({"IC promedio": "{:.3f}"}), width="stretch")

    st.caption("Siguiente paso recomendado antes de invertir: **paper trading** (seguir los picks semanales sin dinero real "
               "durante 2-3 meses y comparar contra el backtest).")

# ---------------------------------------------------------------- 5. METODOLOGIA
with seccion(tab_m):
    st.markdown("""
### Investment Score
1. Cada métrica se convierte en **percentil** (0–100) contra los pares. Las de "menos es mejor" se invierten.
2. Cada **factor** = promedio de sus percentiles disponibles. **Score** = promedio ponderado de factores:
   Growth 18%, Profitability 22%, CashFlow 15%, Risk 15%, Valuation 22%, Quality 8%.
3. **Confianza** = % de métricas con dato. Los datos faltantes no se inventan.

### Qué cambió en la v2
| Antes | Ahora | Por qué |
|---|---|---|
| Pares: 71 acciones de todos los sectores | Pares del **mismo sector** del S&P 500 (default) | Un banco no se compara con Nvidia: márgenes y deuda son de otro mundo |
| Valuación con P/E, P/B, P/S, EV/EBITDA; negativos = N/A | **Yields**: Utilidad/Precio, Libros/Precio, Ventas/Precio, EBITDA/EV | Con N/A, una empresa con pérdidas quedaba *neutral*; ahora queda al fondo, que es lo correcto |
| P/E y earnings yield juntos | Solo utilidad/precio | Eran la misma métrica contada dos veces |
| Descarga una por una, sin reintentos | En paralelo, con reintentos y reporte de calidad | Más rápido y sabes qué falta |
| Backtest con las acciones de hoy | Universo **histórico** del S&P 500 (miembros en cada fecha) | Reduce el sesgo de supervivencia |
| Opera al mismo cierre de la señal | Opera al **día siguiente** | En la vida real no puedes comprar al precio con el que calculaste |
| Benchmark ^GSPC sin dividendos | **SPY con dividendos** | Comparar peras con peras |
| Un solo corte train/test 70/30 | **Walk-forward anidado** año por año | Ver si funciona repetidamente fuera de muestra |
| Veredicto por t-stat | **Checklist de 10 condiciones** + corrección por pruebas múltiples | Un t ≥ 2 solo no basta |
| — | Ablación, correlación de señales, robustez de parámetros, alpha vs momentum clásico | Saber qué aporta cada pieza y si hay alpha propio |
| — | Ranking de pares y screener semanal | Encontrar candidatas, no solo calificar una |

### Lo que NO hace
- Las etiquetas (BUY, HOLD…) son **posiciones relativas** entre pares, no una recomendación: el score fundamental
  no se puede validar hacia atrás con datos gratuitos.
- No predice precios ni sugiere tamaño de posición.
""")
