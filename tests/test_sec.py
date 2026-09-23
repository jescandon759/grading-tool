import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import sec_data


def _q(start, end, val, form="10-Q", fy=2026, fp="Q2"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": end, "fy": fy, "fp": fp}


def fake_facts():
    rev = [_q("2024-07-01", "2025-06-30", 1000, "10-K", 2025, "FY"),
           _q("2023-07-01", "2024-06-30", 800, "10-K", 2024, "FY"),
           _q("2025-07-01", "2025-12-31", 600), _q("2024-07-01", "2024-12-31", 450),
           _q("2023-07-01", "2023-12-31", 380)]
    ni = [dict(r, val=r["val"] * 0.2) for r in rev]
    inst = lambda v: [{"end": "2025-12-31", "val": v, "form": "10-Q", "filed": "2026-02-01", "fy": 2026, "fp": "Q2"}]
    return {"entityName": "Demo Corp", "facts": {
        "us-gaap": {"Revenues": {"units": {"USD": rev}}, "NetIncomeLoss": {"units": {"USD": ni}},
                    "GrossProfit": {"units": {"USD": [dict(r, val=r["val"] * 0.5) for r in rev]}},
                    "OperatingIncomeLoss": {"units": {"USD": [dict(r, val=r["val"] * 0.3) for r in rev]}},
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [dict(r, val=r["val"] * 0.25) for r in rev]}},
                    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [dict(r, val=r["val"] * 0.05) for r in rev]}},
                    "Assets": {"units": {"USD": inst(4000)}}, "StockholdersEquity": {"units": {"USD": inst(2000)}},
                    "LongTermDebt": {"units": {"USD": inst(500)}},
                    "AssetsCurrent": {"units": {"USD": inst(900)}}, "LiabilitiesCurrent": {"units": {"USD": inst(600)}},
                    "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": inst(300)}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": inst(100)}}}}}


def test_sec_build_info(monkeypatch):
    def fetch(url, **k):
        if "company_tickers" in url:
            return {"0": {"ticker": "DEMO", "cik_str": 123, "title": "Demo"}}
        if "companyfacts" in url:
            return fake_facts()
        return {"name": "Demo Corp", "sic": "7372", "sicDescription": "Software"}
    monkeypatch.setattr(sec_data, "FETCH", fetch)
    sec_data._ticker_map.cache_clear()
    i = sec_data.build_info("DEMO", price=50.0, beta=1.1)
    assert i["totalRevenue"] == 1000 + 600 - 450                    # TTM
    assert abs(i["revenueGrowth"] - (1150 / (800 + 450 - 380) - 1)) < 1e-9
    assert abs(i["profitMargins"] - 0.2) < 1e-9 and abs(i["grossMargins"] - 0.5) < 1e-9
    assert i["marketCap"] == 5000 and i["debtToEquity"] == 25 and i["sector_es"] == "Tecnologia"
    assert abs(i["currentRatio"] - 1.5) < 1e-9 and i["fuente"] == "SEC EDGAR"
    assert sec_data.build_info("NOPE") is None and "SEC" in sec_data.LAST_ERROR["NOPE"]


def test_sic_sectors():
    assert sec_data.sector_from_sic(2834) == "Salud" and sec_data.sector_from_sic(6798) == "Bienes raices"
    assert sec_data.sector_from_sic(3674) == "Tecnologia" and sec_data.sector_from_sic(1311) == "Energia"
