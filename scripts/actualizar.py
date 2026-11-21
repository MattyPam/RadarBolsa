"""
RadarBolsa · actualización diaria de datos.

Descarga precios (Yahoo Finance vía yfinance), calcula variaciones, caídas
desde máximo, tasas históricas de recuperación y alertas, y escribe los JSON
que lee el panel (docs/data/).

Se ejecuta solo en GitHub Actions (L-V 11:30 Madrid). No usa Claude.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = RAIZ / "config" / "universo.yaml"
SALIDA = RAIZ / "docs" / "data"
SERIES = SALIDA / "series"

HORIZONTES = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
UMBRALES = [5, 10, 15, 20, 30, 40]   # % de caída desde máximo de 52 semanas
DESDE = "1990-01-01"                 # historia usada para las estadísticas
MIN_CASOS = 8                        # por debajo, la estadística no es fiable

MONEDA_SUFIJO = {".DE": "EUR", ".AS": "EUR", ".PA": "EUR", ".MC": "EUR",
                 ".MI": "EUR", ".SW": "CHF", ".CO": "DKK", ".L": "USD"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def moneda(ticker: str) -> str:
    if ticker.startswith("^"):
        return "pts"
    if ticker.endswith("=X"):
        return ""
    for suf, cur in MONEDA_SUFIJO.items():
        if ticker.endswith(suf):
            return cur
    return "USD"


def nombre_fichero(ticker: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in ticker)


def r(x, nd=2):
    """Redondea y convierte NaN/inf en None para JSON."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, nd)


def var(s: pd.Series, n: int):
    if len(s) <= n:
        return None
    return r((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def descargar(tickers: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    import yfinance as yf

    series, errores = {}, []
    tickers = list(dict.fromkeys(t for t in tickers if t))
    for i in range(0, len(tickers), 15):
        lote = tickers[i:i + 15]
        for intento in range(3):
            try:
                df = yf.download(lote, start=DESDE, auto_adjust=True,
                                 group_by="ticker", threads=True,
                                 progress=False)
                break
            except Exception as e:  # noqa: BLE001
                print(f"Error descargando {lote} (intento {intento+1}): {e}")
                time.sleep(5 * (intento + 1))
        else:
            errores.extend(lote)
            continue
        for t in lote:
            try:
                s = df[t]["Close"].dropna()
            except Exception:  # noqa: BLE001
                s = pd.Series(dtype=float)
            if len(s) < 30:
                errores.append(t)
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")]
            series[t] = s.astype(float)
    return series, errores


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
def caida_52s(s: pd.Series) -> pd.Series:
    maximo = s.rolling(252, min_periods=1).max()
    return (s / maximo - 1) * 100


def episodios(dd: np.ndarray, umbral: float) -> list[int]:
    """Días en que la caída cruza -umbral. Para contar un episodio nuevo,
    el precio tiene que haberse recuperado antes hasta -umbral/2
    (así una misma crisis no cuenta 50 veces)."""
    entradas, armado = [], True
    for i, v in enumerate(dd):
        if armado and v <= -umbral:
            entradas.append(i)
            armado = False
        elif not armado and v > -umbral / 2:
            armado = True
    return entradas


def tasas_historicas(s: pd.Series) -> dict:
    """Para cada umbral de caída: en cuántos episodios el precio estaba más
    alto 1/3/6/12 meses después, mediana y escenario malo (percentil 10)."""
    valores = s.values
    dd = caida_52s(s).values
    out = {}
    for u in UMBRALES:
        idx = episodios(dd, u)
        res = {"episodios": len(idx), "ultimo": None}
        if idx:
            res["ultimo"] = s.index[idx[-1]].strftime("%Y-%m-%d")
        for h, n in HORIZONTES.items():
            rets = [valores[i + n] / valores[i] - 1 for i in idx if i + n < len(valores)]
            if rets:
                a = np.array(rets) * 100
                res[h] = {"n": len(a), "pct_sube": r((a > 0).mean() * 100, 0),
                          "mediana": r(np.median(a), 1),
                          "malo": r(np.percentile(a, 10), 1),
                          "bueno": r(np.percentile(a, 90), 1)}
            else:
                res[h] = {"n": 0}
        out[str(u)] = res
    # Referencia: comprar cualquier día al azar
    base = {}
    for h, n in HORIZONTES.items():
        if len(valores) > n + 50:
            a = (valores[n:] / valores[:-n] - 1) * 100
            base[h] = {"pct_sube": r((a > 0).mean() * 100, 0), "mediana": r(np.median(a), 1)}
    out["cualquier_dia"] = base
    return out


def lectura(dd52, tasas) -> dict:
    """Lectura histórica (no es una recomendación)."""
    tramo = None
    for u in UMBRALES:
        if dd52 is not None and dd52 <= -u:
            tramo = u
    if tramo is None:
        return {"nivel": "neutro", "tramo": None,
                "texto": "Sin caída relevante desde su máximo de 52 semanas."}
    base = tasas["cualquier_dia"].get("6m", {})
    # Usa el umbral más profundo alcanzado que tenga suficientes precedentes
    usado = None
    for u in sorted([u for u in UMBRALES if u <= tramo], reverse=True):
        if tasas[str(u)]["6m"].get("n", 0) >= MIN_CASOS:
            usado = u
            break
    if usado is None:
        n = tasas[str(tramo)]["6m"].get("n", 0)
        return {"nivel": "gris", "tramo": tramo,
                "texto": f"Caída ≥{tramo}% con solo {n} precedentes: la estadística no es "
                         "fiable; decide con el análisis cualitativo."}
    t = tasas[str(usado)]["6m"]
    pct, med = t["pct_sube"], t["mediana"]
    nota = ""
    if usado < tramo:
        n_prof = tasas[str(tramo)]["6m"].get("n", 0)
        nota = f" (caídas ≥{tramo}%: solo {n_prof} precedentes, se usa ≥{usado}%)"
    comp = ""
    if base.get("pct_sube") is not None:
        comp = f"; comprando cualquier día: {base['pct_sube']:.0f}%"
    txt = (f"Tras caídas ≥{usado}%{nota}, estaba más alto 6 meses después en "
           f"{pct:.0f}% de {t['n']} casos{comp}. Mediana {med:+.1f}%, "
           f"escenario malo {t['malo']:+.1f}%.")
    nivel = "verde" if pct >= 70 else ("ambar" if pct >= 55 else "rojo")
    return {"nivel": nivel, "tramo": tramo, "usado": usado, "texto": txt}


def metricas(s: pd.Series, hoy: str) -> dict:
    ultimo = s.iloc[-1]
    ult_fecha = s.index[-1].strftime("%Y-%m-%d")
    ano = s[s.index.year == s.index[-1].year]
    ytd = None
    prev = s[s.index.year < s.index[-1].year]
    if len(prev):
        ytd = r((ultimo / prev.iloc[-1] - 1) * 100)
    ult252 = s.iloc[-252:]
    rets = s.pct_change().iloc[-252:]
    return {
        "precio": r(ultimo, 4 if ultimo < 10 else 2),
        "fecha": ult_fecha,
        "en_sesion": ult_fecha == hoy,
        "var_1d": var(s, 1), "var_5d": var(s, 5), "var_1m": var(s, 21),
        "var_3m": var(s, 63), "var_1a": var(s, 252), "var_ytd": ytd,
        "max_52s": r(ult252.max()), "min_52s": r(ult252.min()),
        "caida_max52": r((ultimo / ult252.max() - 1) * 100),
        "caida_max_hist": r((ultimo / s.max() - 1) * 100),
        "volatilidad": r(rets.std() * math.sqrt(252) * 100, 1),
        "historia_desde": s.index[0].strftime("%Y"),
        "_n_ano": len(ano),
    }


def guardar_serie(ticker: str, s: pd.Series):
    diario = s.iloc[-400:]
    mensual = s.resample("ME").last().dropna().iloc[-121:]
    datos = {
        "ticker": ticker,
        "diario": [[d.strftime("%Y-%m-%d"), r(v, 4)] for d, v in diario.items()],
        "mensual": [[d.strftime("%Y-%m"), r(v, 4)] for d, v in mensual.items()],
    }
    (SERIES / f"{nombre_fichero(ticker)}.json").write_text(
        json.dumps(datos, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    SERIES.mkdir(parents=True, exist_ok=True)
    madrid = ZoneInfo("Europe/Madrid")
    ahora = datetime.now(timezone.utc).astimezone(madrid)
    hoy = ahora.strftime("%Y-%m-%d")

    grupos = [("indice", cfg.get("indices", [])), ("contexto", cfg.get("contexto", [])),
              ("sector", cfg.get("sectores", [])), ("empresa", cfg.get("empresas", []))]
    tickers = [e["ticker"] for _, lst in grupos for e in lst]
    tickers += [e.get("ticker_ucits") for e in cfg.get("sectores", []) if e.get("ticker_ucits")]
    series, errores = descargar(tickers)
    print(f"Series descargadas: {len(series)} · errores: {errores}")
    if len(series) < len(tickers) * 0.5:
        print("Demasiados errores de descarga; no se sobrescriben los datos.")
        sys.exit(1)

    elementos = []
    for tipo, lista in grupos:
        for e in lista:
            t = e["ticker"]
            if t not in series:
                continue
            s = series[t]
            m = metricas(s, hoy)
            item = {"ticker": t, "nombre": e.get("nombre", t), "tipo": tipo,
                    "moneda": moneda(t), **m,
                    "sector": e.get("sector"), "comprable": e.get("comprable"),
                    "isin": e.get("isin") or None}
            m.pop("_n_ano", None)
            item.pop("_n_ano", None)
            if tipo != "contexto":
                tasas = tasas_historicas(s)
                item["tasas"] = tasas
                item["lectura"] = lectura(m["caida_max52"], tasas)
            tu = e.get("ticker_ucits")
            if tipo == "sector" and tu and tu in series:
                su = series[tu]
                mu = metricas(su, hoy)
                item["ucits"] = {"ticker": tu, "moneda": moneda(tu),
                                 "comprable": e.get("comprable") or f"Xtrackers MSCI World {e.get('nombre')} UCITS ETF",
                                 "isin": e.get("isin") or None,
                                 **{k: mu[k] for k in ("precio", "fecha", "var_1d", "var_5d", "var_1m",
                                                       "var_1a", "caida_max52", "historia_desde")}}
                guardar_serie(tu, su)
            guardar_serie(t, s)
            elementos.append(item)

    # --- Alertas -------------------------------------------------------
    a = cfg.get("alertas", {})
    alertas = []
    for it in elementos:
        dd, v5 = it["caida_max52"], it["var_5d"]
        if it["tipo"] == "indice":
            if v5 is not None and v5 <= a.get("indice_5d", -5):
                alertas.append({"clave": f"{it['ticker']}|5d", "ticker": it["ticker"],
                                "texto": f"{it['nombre']}: {v5:+.1f}% en 5 sesiones"})
            if dd is not None and dd <= a.get("indice_desde_max", -10):
                alertas.append({"clave": f"{it['ticker']}|max", "ticker": it["ticker"],
                                "texto": f"{it['nombre']}: {dd:+.1f}% desde su máximo de 52 semanas"})
        elif it["tipo"] == "sector" and dd is not None and dd <= a.get("sector_desde_max", -15):
            alertas.append({"clave": f"{it['ticker']}|max", "ticker": it["ticker"],
                            "texto": f"Sector {it['nombre']}: {dd:+.1f}% desde su máximo de 52 semanas"})
        elif it["tipo"] == "empresa" and dd is not None and dd <= a.get("empresa_desde_max", -25):
            alertas.append({"clave": f"{it['ticker']}|max", "ticker": it["ticker"],
                            "texto": f"{it['nombre']}: {dd:+.1f}% desde su máximo de 52 semanas"})
    for al in alertas:
        it = next(x for x in elementos if x["ticker"] == al["ticker"])
        al["lectura"] = it.get("lectura", {}).get("texto")

    estado_f = SALIDA / "alertas_estado.json"
    previas = set()
    if estado_f.exists():
        previas = set(json.loads(estado_f.read_text(encoding="utf-8")).get("claves", []))
    nuevas = [al for al in alertas if al["clave"] not in previas]
    estado_f.write_text(json.dumps({"fecha": hoy, "claves": sorted(al["clave"] for al in alertas)},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    nuevas_f = RAIZ / "alertas_nuevas.md"
    if nuevas:
        lineas = [f"Nuevas alertas del {hoy} (datos de las {ahora:%H:%M}):", ""]
        for al in nuevas:
            lineas.append(f"- **{al['texto']}**  \n  {al.get('lectura') or ''}")
        lineas += ["", "Abre el panel para ver gráficos y detalle.",
                   "", "_Lectura histórica, no recomendación de inversión._"]
        nuevas_f.write_text("\n".join(lineas), encoding="utf-8")
    elif nuevas_f.exists():
        nuevas_f.unlink()

    # --- Plan por tramos ----------------------------------------------
    tr = cfg.get("tramos", {})
    ref = next((x for x in elementos if x["ticker"] == tr.get("referencia")), None)
    tramos = None
    if ref:
        cap = tr.get("capital", 0)
        niveles = []
        for n in tr.get("niveles", []):
            niveles.append({"caida": n["caida"], "peso": n["peso"],
                            "importe": round(cap * n["peso"] / 100),
                            "activado": ref["caida_max52"] is not None and ref["caida_max52"] <= n["caida"]})
        tramos = {"referencia": ref["ticker"], "nombre": ref["nombre"], "capital": cap,
                  "caida_actual": ref["caida_max52"], "max_52s": ref["max_52s"],
                  "precio": ref["precio"], "niveles": niveles}
        for n in niveles:
            n["precio_objetivo"] = r(ref["max_52s"] * (1 + n["caida"] / 100))

    resumen = {
        "actualizado": ahora.strftime("%Y-%m-%d %H:%M"),
        "hoy": hoy,
        "elementos": elementos,
        "alertas": alertas,
        "alertas_nuevas": [al["clave"] for al in nuevas],
        "tramos": tramos,
        "errores": errores,
        "metodo": {"horizontes": HORIZONTES, "umbrales": UMBRALES, "desde": DESDE,
                   "min_casos": MIN_CASOS},
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, separators=(",", ":")),
                                         encoding="utf-8")
    print(f"OK · {len(elementos)} elementos · {len(alertas)} alertas ({len(nuevas)} nuevas)")


if __name__ == "__main__":
    main()
