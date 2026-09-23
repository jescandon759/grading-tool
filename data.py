"""
Capa de datos: descarga robusta desde Yahoo Finance.
- Descarga en lotes con reintentos (Yahoo corta peticiones grandes o rafagas).
- Devuelve SIEMPRE un reporte de calidad: que tickers fallaron y cuales tienen huecos.
- No depende de Streamlit: la app aplica el cache encima (st.cache_data).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Nombres de sector GICS (ingles) -> espanol
SECTOR_ES = {
    "Information Technology": "Tecnologia", "Communication Services": "Comunicaciones",
    "Consumer Discretionary": "Consumo discrecional", "Consumer Staples": "Consumo basico",
    "Financials": "Financiero", "Health Care": "Salud", "Industrials": "Industrial",
    "Energy": "Energia", "Materials": "Materiales", "Utilities": "Servicios publicos",
    "Real Estate": "Bienes raices",
}


# ---------------------------------------------------------------- UNIVERSO
def yahoo_symbol(t: str) -> str:
    """BRK.B -> BRK-B (formato de Yahoo)."""
    t = str(t).strip().upper()
    return t if t.endswith(".MX") else t.replace(".", "-")


def sp500_universe() -> pd.DataFrame:
    """Constituyentes del S&P 500 (archivo incluido en el repo: data/sp500.csv).
    Columnas: Ticker, Nombre, Sector. Actualizalo con scripts/update_sp500.py."""
    df = pd.read_csv(DATA_DIR / "sp500.csv")
    out = pd.DataFrame({
        "Ticker": df["Symbol"].map(yahoo_symbol),
        "Nombre": df["Security"],
        "Sector": df["GICS Sector"].map(SECTOR_ES).fillna(df["GICS Sector"]),
    })
    return out.drop_duplicates("Ticker").reset_index(drop=True)


def parse_tickers(txt: str) -> list[str]:
    """'aapl, msft; nvda\\nbrk.b' -> ['AAPL','MSFT','NVDA','BRK-B'] sin duplicados."""
    raw = [x for chunk in str(txt).replace(";", ",").replace("\n", ",").split(",")
           for x in chunk.split()]
    return list(dict.fromkeys(yahoo_symbol(x) for x in raw if x.strip()))


# ---------------------------------------------------------------- PRECIOS
def _split_download(raw: pd.DataFrame, tickers: list[str]):
    """yf.download devuelve formatos distintos segun 1 o varios tickers. Normaliza."""
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in lvl0 else pd.DataFrame()
        vol = raw["Volume"] if "Volume" in lvl0 else pd.DataFrame()
    else:  # un solo ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw else pd.DataFrame()
        vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in raw else pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if isinstance(vol, pd.Series):
        vol = vol.to_frame(tickers[0])
    return close, vol


def download_prices(tickers, period="2y", chunk=80, retries=3, pause=1.5, downloader=None):
    """Descarga cierres ajustados y volumen.

    Devuelve (close, volume, reporte). reporte = {"ok": [...], "fallidos": [...],
    "con_huecos": {ticker: % de dias faltantes}}.
    `downloader` permite inyectar datos falsos en pruebas.
    """
    if downloader is None:
        import yfinance as yf

        validos = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

        def downloader(tks, period):
            kw = {"period": period}
            if period not in validos and period.endswith("y"):   # p.ej. "6y" -> fecha de inicio
                kw = {"start": (pd.Timestamp.today() - pd.DateOffset(years=int(period[:-1]))).strftime("%Y-%m-%d")}
            return yf.download(tks, auto_adjust=True, progress=False, threads=True,
                               group_by="column", **kw)

    tickers = list(dict.fromkeys(tickers))
    closes, vols = [], []
    pendientes = tickers[:]
    for intento in range(retries):
        if not pendientes:
            break
        fallaron = []
        for i in range(0, len(pendientes), chunk):
            lote = pendientes[i:i + chunk]
            try:
                c, v = _split_download(downloader(lote, period), lote)
            except Exception:
                fallaron.extend(lote)
                continue
            c = c.dropna(axis=1, how="all") if not c.empty else c
            obtenidos = [t for t in lote if t in c.columns]
            fallaron.extend([t for t in lote if t not in obtenidos])
            if obtenidos:
                closes.append(c[obtenidos])
                if not v.empty:
                    vols.append(v.reindex(columns=obtenidos))
        pendientes = fallaron
        if pendientes and intento < retries - 1:
            time.sleep(pause * (intento + 1))

    close = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    volume = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    if not close.empty:
        close = close.loc[:, ~close.columns.duplicated()].sort_index()
        close = close.dropna(how="all")
        close.index = pd.to_datetime(close.index).tz_localize(None)
    if not volume.empty:
        volume = volume.loc[:, ~volume.columns.duplicated()].reindex(close.index)

    huecos = {}
    if not close.empty:
        # huecos = dias faltantes DESPUES de que el ticker empezo a cotizar
        for t in close.columns:
            s = close[t]
            first = s.first_valid_index()
            if first is None:
                continue
            pct = float(s.loc[first:].isna().mean())
            if pct > 0.02:
                huecos[t] = round(pct * 100, 1)
        close = close.ffill(limit=5)  # rellena huecos cortos (feriados distintos, etc.)

    reporte = {"ok": [t for t in tickers if t in close.columns],
               "fallidos": pendientes, "con_huecos": huecos}
    return close, volume, reporte


# ---------------------------------------------------------------- FUNDAMENTALES
FUND_FIELDS = {
    "name": "shortName", "sector_y": "sector", "mcap": "marketCap",
    "roe": "returnOnEquity", "roa": "returnOnAssets", "margin": "profitMargins",
    "gross": "grossMargins", "oper": "operatingMargins", "de": "debtToEquity",
    "pe": "trailingPE", "fpe": "forwardPE", "pb": "priceToBook",
    "rev_g": "revenueGrowth", "eps_g": "earningsGrowth", "fcf": "freeCashflow",
    "short_pct": "shortPercentOfFloat", "rec": "recommendationMean",
    "target": "targetMeanPrice", "price": "currentPrice",
    "eps": "trailingEps", "bvps": "bookValue",
}


def _one_info(t, retries=2):
    import yfinance as yf
    for k in range(retries):
        try:
            info = yf.Ticker(t).info or {}
            if len(info) > 5:
                row = {k2: info.get(v) for k2, v in FUND_FIELDS.items()}
                return t, row
        except Exception:
            pass
        time.sleep(0.8 * (k + 1))
    return t, None


def fundamentals_from_infos(infos: dict) -> pd.DataFrame:
    """Convierte respuestas crudas de Yahoo (.info) al formato de fundamentales del screener."""
    rows = {t: {k: i.get(v) for k, v in FUND_FIELDS.items()} for t, i in infos.items() if i}
    return _finish_fund(pd.DataFrame.from_dict(rows, orient="index"))


def _finish_fund(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    num = [c for c in df.columns if c not in ("name", "sector_y")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["fcf_yield"] = df["fcf"] / df["mcap"]
    df["earn_yield"] = (df["eps"] / df["price"]).fillna(1 / df["pe"])
    df["book_yield"] = (df["bvps"] / df["price"]).fillna(1 / df["pb"])
    df["upside"] = df["target"] / df["price"] - 1
    return df


def fetch_fundamentals(tickers, workers=8, fetcher=None):
    """Fundamentales actuales (NO historicos) en paralelo.
    Devuelve (df, fallidos). Calcula tambien FCF yield."""
    fetcher = fetcher or _one_info
    filas, fallidos = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetcher, t) for t in tickers]
        for f in as_completed(futs):
            t, row = f.result()
            if row is None:
                fallidos.append(t)
            else:
                filas[t] = row
    df = pd.DataFrame.from_dict(filas, orient="index")
    if df.empty:
        return df, fallidos
    num = [c for c in df.columns if c not in ("name", "sector_y")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["fcf_yield"] = df["fcf"] / df["mcap"]
    # Yields en vez de multiplos: una empresa con perdidas tiene P/E vacio o negativo
    # (y con P/E invertido pareceria "barata"). EPS/precio la manda correctamente al fondo.
    df["earn_yield"] = (df["eps"] / df["price"]).fillna(1 / df["pe"])
    df["book_yield"] = (df["bvps"] / df["price"]).fillna(1 / df["pb"])
    df["upside"] = df["target"] / df["price"] - 1
    return df, fallidos


# ---------------------------------------------------------------- EVENTOS
def _one_earnings(t):
    import yfinance as yf
    try:
        cal = yf.Ticker(t).calendar
        fechas = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if fechas:
            return t, pd.Timestamp(fechas[0])
    except Exception:
        pass
    return t, None


def fetch_earnings_dates(tickers, workers=8, fetcher=None):
    """Proxima fecha de reporte trimestral por ticker (o NaT)."""
    fetcher = fetcher or _one_earnings
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(fetcher, t) for t in tickers]):
            t, d = f.result()
            out[t] = d
    return pd.Series(out, dtype="datetime64[ns]")


# ---------------------------------------------------------------- FX
def native_ccy(ticker: str) -> str:
    return "MXN" if ticker.endswith(".MX") else "USD"


def to_base_currency(close: pd.DataFrame, base: str, fx: pd.DataFrame) -> pd.DataFrame:
    """Convierte cada columna de su moneda nativa a `base`. fx trae columnas tipo 'USDMXN=X'."""
    out = pd.DataFrame(index=close.index)
    for t in close.columns:
        nat = "USD" if t.startswith("^") else native_ccy(t)
        s = close[t]
        if nat != base:
            col = f"{nat}{base}=X"
            if col not in fx.columns:
                raise ValueError(f"No hay tipo de cambio {nat}->{base}")
            s = s * fx[col].reindex(s.index).ffill()
        out[t] = s
    return out


# ---------------------------------------------------------------- UNIVERSO HISTORICO (point-in-time)
def sp500_membership() -> pd.DataFrame:
    """Intervalos de pertenencia al S&P 500 (ticker, start, end) desde 2005.
    Fuente: github.com/fja05680/sp500 (historico de componentes). end vacio = sigue en el indice.
    Sirve para que el backtest solo elija entre empresas que ERAN miembros en cada fecha."""
    df = pd.read_csv(DATA_DIR / "sp500_membership.csv", parse_dates=["start", "end"])
    df["ticker"] = df["ticker"].map(yahoo_symbol)
    return df


def members_between(start, end=None) -> list[str]:
    """Todos los tickers que fueron miembros en algun momento del periodo."""
    m = sp500_membership()
    start = pd.Timestamp(start); end = pd.Timestamp(end or pd.Timestamp.today())
    ok = (m["start"] <= end) & (m["end"].isna() | (m["end"] > start))
    return sorted(m.loc[ok, "ticker"].unique())


def membership_mask(dates, tickers) -> pd.DataFrame:
    """Matriz booleana fechas x tickers: True si el ticker era miembro en esa fecha."""
    m = sp500_membership()
    dates = pd.DatetimeIndex(dates)
    mask = pd.DataFrame(False, index=dates, columns=list(tickers))
    for r in m.itertuples():
        if r.ticker not in mask.columns:
            continue
        sel = (dates >= r.start) & ((dates < r.end) if pd.notna(r.end) else True)
        mask.loc[sel, r.ticker] = True
    return mask
