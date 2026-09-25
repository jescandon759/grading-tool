"""Prueba de humo de la app completa con datos sinteticos (sin internet)."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest
import data, score_model as sm
from test_score import info


def _fake(mp):
    rng = np.random.default_rng(3)
    days = 1600
    idx = pd.bdate_range(end="2026-09-18", periods=days)

    def dl(tks, period):
        n = min(days, int(period[:-1]) * 260)
        r = rng.normal(0.0003, 0.017, (n, len(tks)))
        c = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), idx[-n:], tks)
        v = pd.DataFrame(rng.lognormal(14, .4, (n, len(tks))), idx[-n:], tks)
        return pd.concat({"Close": c, "Volume": v}, axis=1)

    def finfo(t):
        if t in ("SNDK", "ZZZZ", "NEWCO"):         # simula rate limit de Yahoo
            return t, None
        g = np.random.default_rng(abs(hash(t)) % 2**32)
        return t, info(eps=g.normal(4, 3), revenueGrowth=g.normal(.08, .1), grossMargins=g.uniform(.2, .8),
                       debtToEquity=g.uniform(0, 200), targetMeanPrice=110, targetLowPrice=None,
                       targetHighPrice=140, recommendationKey="buy", sector="Technology")

    def fund1(t):
        g = np.random.default_rng(abs(hash(t)) % 2**32)
        return t, {k: g.normal(1, .5) for k in data.FUND_FIELDS} | {"name": t, "sector_y": "x"}

    o_dl, o_fi = data.download_prices, sm.fetch_infos
    o_fu, o_ea = data.fetch_fundamentals, data.fetch_earnings_dates
    mp.setattr(data, "download_prices", lambda tickers, period="2y", **k: o_dl(tickers, period, pause=0, downloader=dl))
    mp.setattr(sm, "fetch_infos", lambda tickers, **k: o_fi(tickers, fetcher=finfo))
    mp.setattr(data, "fetch_fundamentals", lambda tickers, **k: o_fu(tickers, fetcher=fund1))
    mp.setattr(data, "fetch_earnings_dates", lambda tickers, **k: o_ea(tickers, fetcher=lambda t: (t, pd.Timestamp("2026-09-25"))))
    import sec_data
    def fake_sec(t, price=None, beta=None, **k):
        if t != "NEWCO":
            sec_data.LAST_ERROR[t] = "no aparece en la SEC"; return None
        d = info(eps=3, currentPrice=price or 40, sector="Technology")
        d.update({"fuente": "SEC EDGAR", "sector_es": "Tecnologia", "ultimo_reporte": "2026-06-30"})
        return d
    mp.setattr(sec_data, "build_info", fake_sec)
    import yfinance as yf
    class FakeT:
        def __init__(self, t): self.news = [{"title": "Company beats estimates, raises outlook"}]
        income_stmt = pd.DataFrame({pd.Timestamp("2024-12-31"): [1e11, 2e10], pd.Timestamp("2025-12-31"): [1.1e11, 2.3e10]},
                                   index=["Total Revenue", "Net Income"])
    mp.setattr(yf, "Ticker", FakeT)


def _click(at, label):
    [b for b in at.button if b.label.startswith(label)][0].click()
    at.run(timeout=900)
    assert not at.exception, [e.value for e in at.exception]
    errs = [e.value for e in at.error if not str(e.value).startswith(("**Veredicto", "🔴", "**Ni fundamentales"))]
    assert not errs, errs


def test_app(monkeypatch):
    _fake(monkeypatch)
    at = AppTest.from_file(str(ROOT / "score_app.py"), default_timeout=900)
    at.run(); assert not at.exception
    _click(at, "🎯 Analizar")
    assert "an" in at.session_state and len(at.session_state["an"]["uni"]) > 50
    # Yahoo falla para una accion del S&P 500 -> sale del snapshot semanal
    at.text_input[0].set_value("SNDK"); _click(at, "🎯 Analizar")
    assert "SNDK" in at.session_state["an"]["uni"].index
    # fuera del S&P, Yahoo falla, la SEC responde -> se analiza igual
    at.text_input[0].set_value("NEWCO"); _click(at, "🎯 Analizar")
    assert "NEWCO" in at.session_state["an"]["uni"].index
    assert any("Qué tan buena se ve por plazo" in str(h.value) for h in at.subheader)
    assert any("SEC (EDGAR)" in str(i.value) for i in at.info)
    # fuera del S&P y Yahoo falla -> mensaje claro, sin romper la app
    at.text_input[0].set_value("ZZZZ")
    [b for b in at.button if b.label.startswith("🎯 Analizar")][0].click(); at.run(timeout=900)
    assert not at.exception and any("No pude obtener datos de ZZZZ" in str(e.value) for e in at.error)
    _click(at, "🏆 Calcular ranking")
    _click(at, "📅 Correr screener")
    [sb for sb in at.selectbox if sb.label == "Años de prueba"][0].set_value(5)
    _click(at, "⏳ Correr estudio A")
    _click(at, "🧪 Correr estudio B")
    assert "A" in at.session_state and "B" in at.session_state
