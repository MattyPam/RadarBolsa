# RadarBolsa

Panel personal de seguimiento de mercados: caídas de índices, sectores y empresas,
estadística histórica de recuperación, alertas y plan de inversión por tramos.

- **Panel:** https://mattypam.github.io/RadarBolsa/
- **Refresco:** automático de lunes a viernes hacia las 11:30 (Madrid), con GitHub Actions. No consume Claude.
- **Alertas:** cuando aparece una alerta nueva, se abre un *issue* y GitHub te avisa por email.
- **Análisis cualitativo:** cada lunes, Claude escribe `docs/analisis/ultimo.json` (fichas del "por qué").
- **Qué se vigila y umbrales:** `config/universo.yaml`.

Lanzar a mano: pestaña **Actions → Actualización diaria → Run workflow**.

Las lecturas históricas describen lo ocurrido en el pasado tras caídas parecidas. No son predicciones ni recomendaciones de inversión.
