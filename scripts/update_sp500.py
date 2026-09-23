"""Actualiza las listas del S&P 500:
  data/sp500.csv             miembros actuales (github.com/datasets/s-and-p-500-companies)
  data/sp500_membership.csv  historico point-in-time (github.com/fja05680/sp500)
Uso: python scripts/update_sp500.py"""
import io, pathlib, urllib.parse, urllib.request
import pandas as pd

D = pathlib.Path(__file__).resolve().parents[1] / "data"
get = lambda u: urllib.request.urlopen(u, timeout=60).read()
(D / "sp500.csv").write_bytes(get(
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"))
name = urllib.parse.quote("S&P 500 Historical Components & Changes (Updated).csv")
h = pd.read_csv(io.BytesIO(get(f"https://raw.githubusercontent.com/fja05680/sp500/master/{name}")),
                parse_dates=["date"]).sort_values("date")
h = h[h["date"] >= "2005-01-01"]
rows, abiertos, prev = [], {}, set()
for dt, s in zip(h["date"], h["tickers"]):
    cur = {t.strip() for t in s.split(",") if t.strip()}
    for t in cur - prev: abiertos[t] = dt
    for t in prev - cur: rows.append((t, abiertos.pop(t), dt))
    prev = cur
rows += [(t, st, pd.NaT) for t, st in abiertos.items()]
out = pd.DataFrame(rows, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"])
out["ticker"] = out["ticker"].str.replace(".", "-", regex=False)
out.to_csv(D / "sp500_membership.csv", index=False, date_format="%Y-%m-%d")
print("OK:", len(out), "intervalos,", out["ticker"].nunique(), "tickers")
