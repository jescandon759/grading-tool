# Investment Score v2

App de Streamlit (`score_app.py`) que califica acciones de 0 a 100 por factores fundamentales contra su sector.

| Pestaña | Qué hace |
|---|---|
| 🎯 Analizar acción | Score, desglose por factor vs mediana de pares, lugar en su sector, momentum, analistas, noticias, financieros, historia vs S&P |
| 🏆 Ranking de pares | Score de todo un sector, la lista original o el S&P 500 completo; mapa score vs momentum |
| 📅 Screener semanal | S&P 500 + lista extra: momentum, reversión, multifactor, eventos |
| ⏳ Backtests | Momentum mensual (con costos, IC, prueba fuera de muestra) y señales semanales |

Módulos: `score_model.py` (score + backtest mensual), `data.py` (descargas + S&P 500), `factors.py`, `screener.py`.
Pruebas sin internet: `python -m pytest tests`. Lista del índice: `python scripts/update_sp500.py`.

No es asesoría financiera. Datos gratuitos de Yahoo Finance.

## Automatización semanal
- **Sábado 9:00 (GitHub Actions, `.github/workflows/weekly.yml`)**: corre `scripts/weekly_job.py` →
  screener, portafolio (buffer 2N + tope sectorial 30%), snapshot point-in-time de fundamentales e
  Investment Score (`snapshots/`), paper trading (`paper/`) y resumen `reports/latest.json`.
- **Domingo 10:00 (tarea programada de Claude)**: lee `reports/latest.json` y manda el correo.
- Destinatarios, lista extra, pesos y reglas del portafolio: **`config/settings.json`**.
- Correr a mano: pestaña *Actions* → *Screener semanal* → *Run workflow*.

## Fuentes de datos
1. **Yahoo Finance** (precios, fundamentales, analistas).
2. **Snapshot semanal** (`snapshots/`) para el S&P 500 cuando Yahoo limita al servidor de la app.
3. **SEC EDGAR** (`sec_data.py`, fuente oficial) para cualquier empresa que reporte a la SEC cuando Yahoo falla.
   La SEC pide identificarse: pon tu correo en `config/settings.json` → `sec_user_agent` si alguna vez rechaza consultas.
