"""
Trabajo semanal (lo corre GitHub Actions cada sabado; tambien a mano: python scripts/weekly_job.py)

1. Screener de la semana (S&P 500 + lista extra) y portafolio con las mismas reglas del backtest
   (buffer de rotacion + tope sectorial).
2. Snapshot point-in-time: guarda fundamentales e Investment Score de todo el S&P 500 con fecha.
   Con el tiempo esto se vuelve TU historico para validar el score fundamental sin trampa.
3. Paper trading: registra los picks y mide su resultado real a 1, 4 y 12 semanas vs SPY.
4. reports/latest.json: resumen que lee la tarea programada para mandar el correo del domingo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest as bt  # noqa: E402
import data  # noqa: E402
import factors as fx  # noqa: E402
import score_model as sm  # noqa: E402
import screener as sc  # noqa: E402


def load_cfg(root: Path = ROOT) -> dict:
    return json.loads((root / "config" / "settings.json").read_text())


def _f(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), d)


# ---------------------------------------------------------------- PAPER TRADING
def update_paper(root: Path, picks: pd.DataFrame, fecha: pd.Timestamp, close: pd.DataFrame,
                 horizons: list[int]) -> tuple[pd.DataFrame, dict]:
    """picks.csv acumula las recomendaciones; performance se recalcula completa cada semana.
    Entrada = primer cierre DESPUES de la fecha de senal (igual que el backtest)."""
    path = root / "paper" / "picks.csv"
    hist = pd.read_csv(path, parse_dates=["fecha_senal"]) if path.exists() else pd.DataFrame()
    nuevos = picks.assign(fecha_senal=fecha)
    if not hist.empty:
        hist = hist[hist["fecha_senal"] != fecha]            # re-ejecutar la misma semana no duplica
    hist = pd.concat([hist, nuevos], ignore_index=True)
    hist.to_csv(path, index=False, date_format="%Y-%m-%d")

    idx = close.index
    rows = []
    for r in hist.itertuples():
        pos = idx.searchsorted(pd.Timestamp(r.fecha_senal), side="right")   # siguiente dia habil
        if pos >= len(idx) or r.ticker not in close:
            continue
        p0, s0 = close[r.ticker].iloc[pos], close["SPY"].iloc[pos]
        row = {"fecha_senal": r.fecha_senal, "ticker": r.ticker, "entrada": idx[pos], "precio_entrada": p0}
        for h in horizons:
            e = pos + 5 * h
            if e < len(idx) and pd.notna(p0):
                ret = close[r.ticker].iloc[e] / p0 - 1
                spy = close["SPY"].iloc[e] / s0 - 1
                row[f"ret_{h}s"], row[f"exceso_{h}s"] = ret, ret - spy
        rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(root / "paper" / "performance.csv", index=False, date_format="%Y-%m-%d")
    resumen = {}
    for h in horizons:
        c = f"exceso_{h}s"
        if not perf.empty and c in perf and perf[c].notna().sum() > 0:
            x = perf[c].dropna()
            resumen[f"{h} semanas"] = {"picks_medidos": int(len(x)), "exceso_promedio": _f(x.mean()),
                                       "tasa_acierto": _f((x > 0).mean(), 3)}
    return perf, resumen


# ---------------------------------------------------------------- JOB
def run(close: pd.DataFrame, vol: pd.DataFrame, infos: dict, earnings: pd.Series, cfg: dict,
        root: Path = ROOT, today: pd.Timestamp | None = None) -> dict:
    u = data.sp500_universe()
    sectors_all = u.set_index("Ticker")["Sector"]
    scfg, pcfg = cfg["screener"], cfg["portafolio"]
    fecha = close.index[-1]
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    stamp = f"{fecha:%Y-%m-%d}"

    # 1) SCREENER --------------------------------------------------------------
    px_ = close.drop(columns="SPY", errors="ignore")
    v_ = vol.drop(columns="SPY", errors="ignore") if not vol.empty else None
    tick = list(px_.columns)
    sectors = sectors_all.reindex(tick).fillna("Fuera del S&P")
    liq = sc.liquidity_filter(px_, v_, scfg["volumen_min_musd"] * 1e6, scfg["precio_min"])
    snap = fx.snapshot(fx.price_signals(px_[liq], v_.reindex(columns=liq) if v_ is not None else None))
    cand = sc.price_prescreen(snap, sectors, scfg["candidatas_estudio"])
    fund = data.fundamentals_from_infos({t: infos[t] for t in cand if t in infos})
    rank, det = sc.weekly_ranking(snap, sectors, fund, earnings, scfg["pesos"], today=today)
    rank = rank.loc[[t for t in rank.index if t in cand]]
    rank["Rank"] = range(1, len(rank) + 1)
    (root / "reports").mkdir(exist_ok=True)
    rank.to_csv(root / "reports" / f"screener_{stamp}.csv")

    # Portafolio con las reglas del backtest (buffer + tope sectorial)
    prev_path = root / "paper" / "picks.csv"
    prev = []
    if prev_path.exists():
        h = pd.read_csv(prev_path, parse_dates=["fecha_senal"])
        h = h[h["fecha_senal"] < fecha]
        if not h.empty:
            prev = list(h.loc[h["fecha_senal"] == h["fecha_senal"].max(), "ticker"])
    port = bt.select_portfolio(rank["Score"], pcfg["top_n"], prev, pcfg["buffer"], sectors, pcfg["tope_sector"])
    port = sorted(port, key=lambda t: rank.loc[t, "Rank"])
    picks = rank.loc[port, ["Rank", "Score", "Senal principal", "Sector"]].reset_index(names="ticker")
    picks.columns = ["ticker", "rank", "score", "senal", "sector"]

    # 2) SNAPSHOT POINT-IN-TIME -------------------------------------------------
    sp = [t for t in u["Ticker"] if t in infos]
    uni = sm.build_universe({t: infos[t] for t in sp}, px_)
    uni["_sector"] = sectors_all.reindex(uni.index).fillna(uni["_sector"])   # sector GICS, igual que la app
    (root / "snapshots" / "fund").mkdir(parents=True, exist_ok=True)
    (root / "snapshots" / "score").mkdir(parents=True, exist_ok=True)
    num = uni[[c for c in uni.columns if not c.startswith("_")]]
    num = num.assign(_name=uni["_name"], _price=px_.iloc[-1].reindex(uni.index),
                     _upside=data.fundamentals_from_infos({t: infos[t] for t in uni.index})["upside"].reindex(uni.index))
    num.assign(sector=uni["_sector"]).to_csv(root / "snapshots" / "fund" / f"{stamp}.csv.gz", compression="gzip")
    partes = [sm.rank_universe(uni[uni["_sector"] == s]) for s in uni["_sector"].unique()
              if (uni["_sector"] == s).sum() >= 8]
    score_tab = pd.concat(partes).sort_values("Score", ascending=False) if partes else pd.DataFrame()
    if not score_tab.empty:
        score_tab.assign(precio=px_.iloc[-1].reindex(score_tab.index)).to_csv(
            root / "snapshots" / "score" / f"{stamp}.csv.gz", compression="gzip")

    # 3) PAPER TRADING ----------------------------------------------------------
    perf, paper = update_paper(root, picks, fecha, close, cfg["paper_trading"]["horizontes_semanas"])

    # 4) RESUMEN PARA EL CORREO -------------------------------------------------
    spy = close["SPY"].dropna()
    ma200 = spy.rolling(200).mean().iloc[-1]
    fund_all = data.fundamentals_from_infos({t: infos[t] for t in port if t in infos})
    top = []
    for t in port:
        r = rank.loc[t]
        f = fund_all.loc[t] if t in fund_all.index else {}
        top.append({
            "ticker": t, "empresa": (f.get("name") if len(f) else None) or t, "sector": r["Sector"],
            "rank_screener": int(r["Rank"]), "score": _f(r["Score"], 1), "senal": r["Senal principal"],
            **{k: _f(r[k], 1) for k in sc.ESTRATEGIAS},
            "rend_1s": _f(r["Rend 1 sem"]), "rend_3m": _f(r["Rend 3m"]), "tendencia": r["Tendencia"],
            "reporta_en_dias": _f(r.get("Reporta en (dias)"), 0),
            "upside_analistas": _f(f.get("upside")) if len(f) else None,
            "investment_score": _f(score_tab.loc[t, "Score"], 1) if t in score_tab.index else None,
            "nuevo_esta_semana": t not in prev,
        })
    earn_top30 = [{"ticker": t, "dias": int(rank.loc[t, "Reporta en (dias)"])}
                  for t in rank.index[:30] if pd.notna(rank.loc[t].get("Reporta en (dias)", np.nan))
                  and rank.loc[t, "Reporta en (dias)"] <= 7]
    resumen = {
        "generado": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "datos_al_cierre": stamp,
        "regimen": {"spy": _f(spy.iloc[-1], 2), "media_200d": _f(ma200, 2),
                    "alcista": bool(spy.iloc[-1] > ma200),
                    "spy_semana": _f(spy.iloc[-1] / spy.iloc[-6] - 1)},
        "reglas": {"top_n": pcfg["top_n"], "buffer": pcfg["buffer"], "tope_sector": pcfg["tope_sector"]},
        "portafolio": top,
        "entran": [t for t in port if t not in prev], "salen": [t for t in prev if t not in port],
        "reportan_esta_semana_top30": earn_top30,
        "top_investment_score": [{"ticker": t, "empresa": score_tab.loc[t, "Empresa"], "sector": score_tab.loc[t, "Sector"],
                                  "score": _f(score_tab.loc[t, "Score"], 1), "momentum": _f(score_tab.loc[t, "Momentum"], 1)}
                                 for t in score_tab.index[:5]] if not score_tab.empty else [],
        "paper_trading": paper,
        "calidad_datos": {"tickers": len(tick), "sin_precio": len([t for t in cfg_tickers(cfg, u) if t not in close]),
                          "sin_fundamentales": len([t for t in sp if t not in infos]),
                          "liquidas": int(len(liq)), "candidatas": len(cand)},
        "destinatarios": cfg["correo"]["destinatarios"], "app_url": cfg["correo"]["app_url"],
    }
    (root / "reports" / "latest.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=str))
    return resumen


def cfg_tickers(cfg: dict, u: pd.DataFrame | None = None) -> list[str]:
    u = u if u is not None else data.sp500_universe()
    return list(dict.fromkeys(list(u["Ticker"]) + [data.yahoo_symbol(t) for t in cfg["screener"]["extra"]]))


def main():
    cfg = load_cfg()
    tick = cfg_tickers(cfg)
    print(f"Descargando precios de {len(tick)} + SPY...")
    close, vol, rep = data.download_prices(tick + ["SPY"], period="2y")
    print("Fallidos:", rep["fallidos"])
    if "SPY" not in close:
        raise SystemExit("Sin SPY: se aborta para no publicar un reporte incompleto.")
    print("Descargando fundamentales (en paralelo)...")
    infos, bad = sm.fetch_infos(tick, workers=6)
    print(f"Fundamentales: {len(infos)} ok, {len(bad)} fallidos")
    if len(infos) < 0.7 * len(tick):
        raise SystemExit("Menos del 70% de fundamentales: Yahoo fallo; se aborta.")
    # fechas de reporte solo para las candidatas (lo caro)
    px_ = close.drop(columns="SPY")
    u = data.sp500_universe()
    sectors = u.set_index("Ticker")["Sector"].reindex(px_.columns).fillna("Fuera del S&P")
    liq = sc.liquidity_filter(px_, vol.drop(columns="SPY", errors="ignore"),
                              cfg["screener"]["volumen_min_musd"] * 1e6, cfg["screener"]["precio_min"])
    snap = fx.snapshot(fx.price_signals(px_[liq], vol.reindex(columns=liq)))
    cand = sc.price_prescreen(snap, sectors, cfg["screener"]["candidatas_estudio"])
    earnings = data.fetch_earnings_dates(cand)
    res = run(close, vol, infos, earnings, cfg)
    print(json.dumps({k: res[k] for k in ("datos_al_cierre", "entran", "salen", "paper_trading")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
