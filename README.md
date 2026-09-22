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
