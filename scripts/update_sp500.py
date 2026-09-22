"""Actualiza data/sp500.csv con la lista vigente del S&P 500.
Fuente: github.com/datasets/s-and-p-500-companies (se actualiza con los cambios del indice).
Uso: python scripts/update_sp500.py"""
import pathlib, urllib.request
URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
dest = pathlib.Path(__file__).resolve().parents[1] / "data" / "sp500.csv"
dest.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
print("OK:", sum(1 for _ in dest.open()) - 1, "empresas ->", dest)
