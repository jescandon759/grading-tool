import sys, pathlib, json, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
import data
import weekly_job as wj
from test_score import info


def test_weekly_job(tmp_path):
    (tmp_path / "config").mkdir(); (tmp_path / "paper").mkdir()
    shutil.copy(ROOT / "config" / "settings.json", tmp_path / "config")
    cfg = wj.load_cfg(tmp_path)
    tick = wj.cfg_tickers(cfg) + ["SPY"]
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-09-18", periods=560)
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(3e-4, .017, (560, len(tick))), 0)), idx, tick)
    vol = pd.DataFrame(rng.lognormal(14, .4, (560, len(tick))), idx, tick)
    infos = {t: info(eps=rng.normal(4, 3), revenueGrowth=rng.normal(.08, .1), sector="Technology",
                     targetMeanPrice=60, currentPrice=50) for t in tick if t != "SPY"}
    earn = pd.Series({t: pd.Timestamp("2026-09-24") for t in tick[:40]})
    # semana 1 (datos hasta hace 30 dias) y semana 2 (hoy)
    r1 = wj.run(close.iloc[:-30], vol.iloc[:-30], infos, earn, cfg, tmp_path, today="2026-08-08")
    r2 = wj.run(close, vol, infos, earn, cfg, tmp_path, today="2026-09-19")
    assert len(r2["portafolio"]) == cfg["portafolio"]["top_n"]
    assert set(r2["entran"]) | (set(p["ticker"] for p in r2["portafolio"]) - set(r2["entran"])) == \
        set(p["ticker"] for p in r2["portafolio"])
    lat = json.loads((tmp_path / "reports" / "latest.json").read_text())
    assert lat["destinatarios"] == cfg["correo"]["destinatarios"] and lat["datos_al_cierre"] == "2026-09-18"
    picks = pd.read_csv(tmp_path / "paper" / "picks.csv")
    assert picks["fecha_senal"].nunique() == 2
    perf = pd.read_csv(tmp_path / "paper" / "performance.csv")
    assert perf["exceso_1s"].notna().sum() >= 10                 # la semana 1 ya se puede medir
    assert "1 semanas" in lat["paper_trading"]
    assert list((tmp_path / "snapshots" / "score").glob("*.csv.gz"))
    # re-ejecutar la misma semana no duplica
    wj.run(close, vol, infos, earn, cfg, tmp_path, today="2026-09-19")
    assert pd.read_csv(tmp_path / "paper" / "picks.csv")["fecha_senal"].nunique() == 2
