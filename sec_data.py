"""
Fundamentales desde la SEC (EDGAR XBRL): fuente OFICIAL y gratuita, sin limites tipo Yahoo.
Sirve para analizar CUALQUIER empresa que reporte a la SEC (NYSE, Nasdaq y ADRs con 20-F),
este o no en el S&P 500. Se usa cuando Yahoo no responde.

Devuelve un dict con las MISMAS llaves que Yahoo (.info) para que el Investment Score no cambie:
totalRevenue, revenueGrowth, profitMargins, returnOnEquity, freeCashflow, marketCap, etc.

Limites honestos:
- Datos de los reportes 10-K / 10-Q (o 20-F): pueden tener semanas de antiguedad.
- Empresas que reportan en otra moneda: margenes, ROE y crecimiento si; valuacion (yields) no.
- No trae estimados de analistas (eso solo viene de Yahoo).
- Acciones que no reportan a la SEC (p.ej. BMV .MX) no se pueden cubrir por aqui.
"""
from __future__ import annotations

import json
import time
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

UA = "InvestmentScore research tool (github.com/jescandon759/grading-tool)"
LAST_ERROR: dict[str, str] = {}


def _get_json(url: str, ua: str | None = None, retries: int = 2):
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua or UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(0.6 * (k + 1))
    raise err


FETCH = _get_json          # se puede reemplazar en pruebas


@lru_cache(maxsize=1)
def _ticker_map() -> dict:
    d = FETCH("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in d.values()}


def cik_for(ticker: str) -> int | None:
    return _ticker_map().get(ticker.upper().replace(".", "-"))


# ---------------------------------------------------------------- SECTOR POR CODIGO SIC
def sector_from_sic(sic) -> str | None:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    reglas = [
        ((1311, 1389), "Energia"), ((2900, 2999), "Energia"), ((4610, 4619), "Energia"),
        ((1000, 1499), "Materiales"), ((1500, 1799), "Industrial"),
        ((2000, 2199), "Consumo basico"), ((2200, 2399), "Consumo discrecional"),
        ((2711, 2741), "Comunicaciones"), ((2400, 2799), "Materiales"),
        ((2830, 2836), "Salud"), ((2840, 2844), "Consumo basico"), ((2800, 2899), "Materiales"),
        ((3000, 3399), "Materiales"), ((3570, 3579), "Tecnologia"), ((3630, 3639), "Consumo discrecional"),
        ((3600, 3699), "Tecnologia"), ((3711, 3716), "Consumo discrecional"), ((3400, 3799), "Industrial"),
        ((3841, 3851), "Salud"), ((3800, 3899), "Tecnologia"), ((3900, 3999), "Consumo discrecional"),
        ((4800, 4899), "Comunicaciones"), ((4900, 4999), "Servicios publicos"), ((4000, 4799), "Industrial"),
        ((5122, 5122), "Salud"), ((5400, 5499), "Consumo basico"), ((5912, 5912), "Consumo basico"),
        ((5000, 5199), "Industrial"), ((5200, 5999), "Consumo discrecional"),
        ((6798, 6798), "Bienes raices"), ((6500, 6599), "Bienes raices"), ((6000, 6799), "Financiero"),
        ((7370, 7379), "Tecnologia"), ((7800, 7999), "Comunicaciones"), ((7300, 7399), "Industrial"),
        ((7000, 7299), "Consumo discrecional"), ((8000, 8099), "Salud"), ((8700, 8799), "Industrial"),
    ]
    for (lo, hi), sec in reglas:
        if lo <= s <= hi:
            return sec
    return "Industrial"


# ---------------------------------------------------------------- LECTURA DE XBRL
TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "Revenue"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "ProfitLossAttributableToOwnersOfParent", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gross_profit": ["GrossProfit"],
    "cost_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfSales"],
    "op_income": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
           "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortisationExpense"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic", "DilutedEarningsLossPerShare"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
               "EquityAttributableToOwnersOfParent", "Equity"],
    "cur_assets": ["AssetsCurrent", "CurrentAssets"],
    "cur_liab": ["LiabilitiesCurrent", "CurrentLiabilities"],
    "lt_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "NoncurrentPortionOfNoncurrentBorrowings"],
    "st_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings", "CommercialPaper",
                "CurrentPortionOfNoncurrentBorrowings"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "CashAndCashEquivalents"],
    "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding",
               "WeightedAverageNumberOfDilutedSharesOutstanding"],
}
FLOW = {"revenue", "net_income", "gross_profit", "cost_revenue", "op_income", "cfo", "capex", "da", "eps"}


def _entries(facts: dict, key: str) -> tuple[pd.DataFrame, str | None]:
    """Primer tag con datos (us-gaap, ifrs-full o dei). Devuelve (DataFrame, unidad)."""
    for tag in TAGS[key]:
        for tax in ("us-gaap", "ifrs-full", "dei"):
            node = facts.get(tax, {}).get(tag)
            if not node:
                continue
            units = node.get("units", {})
            if not units:
                continue
            unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units else next(iter(units)))
            df = pd.DataFrame(units[unit])
            if df.empty or "val" not in df:
                continue
            df["end"] = pd.to_datetime(df["end"])
            if "start" in df:
                df["start"] = pd.to_datetime(df["start"])
                df["dias"] = (df["end"] - df["start"]).dt.days
            df = df.sort_values(["end", "filed"]).drop_duplicates(["start", "end"] if "start" in df else ["end"], keep="last")
            return df, unit
    return pd.DataFrame(), None


def _ttm(df: pd.DataFrame) -> tuple[float, float]:
    """(ultimos 12 meses, 12 meses del anio anterior) para un concepto de flujo.
    TTM = ultimo anual + acumulado del anio en curso - acumulado equivalente del anio pasado."""
    if df.empty or "dias" not in df:
        return np.nan, np.nan
    anual = df[df["dias"].between(350, 380)].sort_values("end")
    if anual.empty:
        return np.nan, np.nan
    fy = anual.iloc[-1]
    prev_fy = anual.iloc[-2]["val"] if len(anual) > 1 else np.nan
    ytd = df[(df["end"] > fy["end"]) & (df["dias"] < 350) & (df["start"] >= fy["end"] - pd.Timedelta(days=5))]
    if ytd.empty:
        return float(fy["val"]), float(prev_fy) if pd.notna(prev_fy) else np.nan
    cur = ytd.sort_values(["end", "dias"]).iloc[-1]
    ant = df[(df["dias"].sub(cur["dias"]).abs() <= 10) &
             ((df["end"] - (cur["end"] - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=10))]
    if ant.empty:
        return float(fy["val"]), float(prev_fy) if pd.notna(prev_fy) else np.nan
    ttm = fy["val"] + cur["val"] - ant.iloc[-1]["val"]
    # TTM del anio anterior (para crecimiento): FY previo + YTD previo - YTD de dos anios atras
    ant2 = df[(df["dias"].sub(cur["dias"]).abs() <= 10) &
              ((df["end"] - (cur["end"] - pd.DateOffset(years=2))).abs() <= pd.Timedelta(days=10))]
    prev = prev_fy + ant.iloc[-1]["val"] - ant2.iloc[-1]["val"] if (pd.notna(prev_fy) and not ant2.empty) else prev_fy
    return float(ttm), float(prev) if pd.notna(prev) else np.nan


def _latest(df: pd.DataFrame, total: bool = False) -> float:
    if df.empty:
        return np.nan
    last = df["end"].max()
    v = df[df["end"] == last]["val"]
    return float(v.sum() if total else v.iloc[-1])


def _div(a, b):
    return a / b if (pd.notna(a) and pd.notna(b) and b not in (0,) and b > 0) else np.nan


def build_info(ticker: str, price: float | None = None, beta: float | None = None, ua: str = UA) -> dict | None:
    """Arma un dict tipo Yahoo .info con datos de la SEC. None si no hay datos."""
    try:
        cik = cik_for(ticker)
        if not cik:
            LAST_ERROR[ticker] = "no aparece en la SEC (no cotiza en EE.UU. o no reporta a la SEC)"
            return None
        facts = FETCH(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").get("facts", {})
        subm = FETCH(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    except Exception as e:  # noqa: BLE001
        LAST_ERROR[ticker] = f"la SEC no respondió ({str(e)[:80]})"
        return None

    E = {k: _entries(facts, k) for k in TAGS}
    unit = E["revenue"][1] or E["net_income"][1]
    en_usd = unit in (None, "USD")
    t = {k: _ttm(E[k][0]) for k in FLOW}
    rev, rev_prev = t["revenue"]
    ni, ni_prev = t["net_income"]
    gp = t["gross_profit"][0]
    if pd.isna(gp) and pd.notna(t["cost_revenue"][0]) and pd.notna(rev):
        gp = rev - t["cost_revenue"][0]
    op, da = t["op_income"][0], t["da"][0]
    ebitda = op + da if pd.notna(op) and pd.notna(da) else np.nan
    fcf = t["cfo"][0] - t["capex"][0] if pd.notna(t["cfo"][0]) and pd.notna(t["capex"][0]) else t["cfo"][0]
    assets, equity = _latest(E["assets"][0]), _latest(E["equity"][0])
    debt = np.nansum([_latest(E["lt_debt"][0]), _latest(E["st_debt"][0])])
    cash = _latest(E["cash"][0])
    shares = _latest(E["shares"][0], total=True)
    eps = t["eps"][0]

    info = {
        "shortName": subm.get("name") or facts.get("entityName") or ticker,
        "sector_es": sector_from_sic(subm.get("sic")), "sicDescription": subm.get("sicDescription"),
        "totalRevenue": rev, "netIncomeToCommon": ni,
        "revenueGrowth": _div(rev, rev_prev) - 1 if pd.notna(_div(rev, rev_prev)) else np.nan,
        "earningsGrowth": (ni / ni_prev - 1) if (pd.notna(ni) and pd.notna(ni_prev) and ni_prev > 0) else np.nan,
        "profitMargins": _div(ni, rev), "grossMargins": _div(gp, rev), "operatingMargins": _div(op, rev),
        "ebitdaMargins": _div(ebitda, rev), "returnOnEquity": _div(ni, equity), "returnOnAssets": _div(ni, assets),
        "freeCashflow": fcf, "ebitda": ebitda,
        "debtToEquity": _div(debt, equity) * 100 if pd.notna(_div(debt, equity)) else np.nan,
        "currentRatio": _div(_latest(E["cur_assets"][0]), _latest(E["cur_liab"][0])),
        "beta": beta, "fuente": "SEC EDGAR", "moneda_reportes": unit,
        "ultimo_reporte": str(E["revenue"][0]["end"].max().date()) if not E["revenue"][0].empty else None,
    }
    if price and pd.notna(price) and en_usd and pd.notna(shares) and shares > 0:
        mcap = price * shares
        info.update({"currentPrice": price, "marketCap": mcap, "trailingEps": eps if pd.notna(eps) else _div(ni, shares),
                     "bookValue": _div(equity, shares),
                     "enterpriseValue": mcap + (debt if pd.notna(debt) else 0) - (cash if pd.notna(cash) else 0)})
    elif price:
        info["currentPrice"] = price          # sin acciones o en otra moneda: sin valuacion
    datos = sum(pd.notna(v) for k, v in info.items() if isinstance(v, (int, float)))
    if datos < 5:
        LAST_ERROR[ticker] = "la SEC no tiene suficientes datos financieros de esta empresa"
        return None
    return info
