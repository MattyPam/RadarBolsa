"""
RadarBolsa · motor de datos (v2).

Cada día laborable (GitHub Actions, 11:30 Madrid, sin usar Claude):
  1. Descarga precios de índices, sectores, ~600 acciones (S&P 500 + Europa)
     y las posiciones de la cartera.
  2. Mide cada producto contra su "nivel habitual": el percentil 80 de sus
     cierres del último año (P80). Descuento = precio / P80 - 1.
  3. Calcula la estadística histórica: tras descuentos parecidos, cuántas veces
     estaba más alto 1/3/6/12 meses después, comparado con comprar cualquier día.
  4. Elige las acciones candidatas (radar), descarga sus fundamentales y escribe
     los JSON que lee la app (docs/data/).
"""
from __future__ import annotations

import io
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
CARTERA = RAIZ / "config" / "cartera.yaml"
SP500_CACHE = RAIZ / "config" / "sp500_cache.csv"
SALIDA = RAIZ / "docs" / "data"
SERIES = SALIDA / "series"

HORIZONTES = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
UMBRALES = [5, 10, 15, 20, 30, 40]          # % por debajo del P80
DESDE = "2000-01-01"
VENTANA = 252

MONEDA_SUFIJO = {".DE": "EUR", ".AS": "EUR", ".PA": "EUR", ".MC": "EUR", ".MI": "EUR",
                 ".BR": "EUR", ".HE": "EUR", ".SW": "CHF", ".CO": "DKK", ".ST": "SEK",
                 ".OL": "NOK", ".L": "GBp"}

GICS = {"Information Technology": "Tecnología", "Communication Services": "Comunicación",
        "Consumer Discretionary": "Consumo discrecional", "Consumer Staples": "Consumo básico",
        "Health Care": "Salud", "Financials": "Financiero", "Industrials": "Industria",
        "Energy": "Energía", "Materials": "Materiales", "Utilities": "Utilities",
        "Real Estate": "Inmobiliario"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def moneda(ticker: str, override: str | None = None) -> str:
    if override:
        return override
    if ticker.startswith("^"):
        return "pts"
    for suf, cur in MONEDA_SUFIJO.items():
        if ticker.endswith(suf):
            return cur
    return "USD"


def region(ticker: str) -> str:
    return "EE. UU." if "." not in ticker else "Europa"


def nombre_fichero(ticker: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in ticker)


def r(x, nd=2):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, nd)


def sig(x):
    """Redondeo a 5 cifras significativas para las series (ficheros más ligeros)."""
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.5g}")


def var(v: np.ndarray, n: int):
    if len(v) <= n:
        return None
    return r((v[-1] / v[-1 - n] - 1) * 100)


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------
def sp500() -> list[dict]:
    try:
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0 RadarBolsa"}, timeout=30).text
        df = pd.read_html(io.StringIO(html))[0]
        df = df.rename(columns={"Symbol": "ticker", "Security": "nombre",
                                "GICS Sector": "gics", "GICS Sub-Industry": "sub"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df[["ticker", "nombre", "gics", "sub"]].to_csv(SP500_CACHE, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo leer el S&P 500 de Wikipedia ({e}); uso la copia guardada.")
        if not SP500_CACHE.exists():
            return []
        df = pd.read_csv(SP500_CACHE)
    out = []
    for _, f in df.iterrows():
        sector = GICS.get(f["gics"], f["gics"])
        if "Semiconductor" in str(f.get("sub", "")):
            sector = "Semiconductores"
        out.append({"ticker": f["ticker"], "nombre": f["nombre"], "sector": sector})
    return out


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def descargar(tickers: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    import yfinance as yf

    series, errores = {}, []
    tickers = list(dict.fromkeys(t for t in tickers if t))
    lote_n = 40
    for i in range(0, len(tickers), lote_n):
        lote = tickers[i:i + lote_n]
        df = None
        for intento in range(3):
            try:
                df = yf.download(lote, start=DESDE, auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001
                print(f"Error descargando lote {i} (intento {intento+1}): {e}")
                time.sleep(5 * (intento + 1))
        if df is None:
            errores.extend(lote)
            continue
        for t in lote:
            try:
                s = df[t]["Close"].dropna()
            except Exception:  # noqa: BLE001
                s = pd.Series(dtype=float)
            if len(s) < 260:
                errores.append(t)
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")]
            series[t] = s.astype(float)
        time.sleep(1)
    return series, errores


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
def p80_movil(v: np.ndarray, q: float) -> np.ndarray:
    """Percentil q de los últimos 252 cierres, día a día (NaN al principio)."""
    out = np.full(len(v), np.nan)
    if len(v) < VENTANA:
        return out
    win = np.lib.stride_tricks.sliding_window_view(v, VENTANA)
    out[VENTANA - 1:] = np.percentile(win, q, axis=1)
    return out


def episodios(desc: np.ndarray, umbral: float) -> list[int]:
    """Días en que el descuento cruza -umbral. Para contar otro episodio, el
    precio tiene que haberse recuperado antes hasta -umbral/2 (una misma
    crisis cuenta una sola vez)."""
    entradas, armado = [], True
    for i, x in enumerate(desc):
        if np.isnan(x):
            continue
        if armado and x <= -umbral:
            entradas.append(i)
            armado = False
        elif not armado and x > -umbral / 2:
            armado = True
    return entradas


def tasas_historicas(s: pd.Series, desc: np.ndarray) -> dict:
    v = s.values
    out = {}
    for u in UMBRALES:
        idx = episodios(desc, u)
        res = {"episodios": len(idx), "ultimo": s.index[idx[-1]].strftime("%Y-%m-%d") if idx else None}
        for h, n in HORIZONTES.items():
            rets = [v[i + n] / v[i] - 1 for i in idx if i + n < len(v)]
            if rets:
                a = np.array(rets) * 100
                res[h] = {"n": len(a), "pct": r((a > 0).mean() * 100, 0), "med": r(np.median(a), 1),
                          "malo": r(np.percentile(a, 10), 1), "bueno": r(np.percentile(a, 90), 1)}
            else:
                res[h] = {"n": 0}
        out[str(u)] = res
    base = {}
    for h, n in HORIZONTES.items():
        if len(v) > n + 50:
            a = (v[n:] / v[:-n] - 1) * 100
            base[h] = {"pct": r((a > 0).mean() * 100, 0), "med": r(np.median(a), 1)}
    out["cualquier_dia"] = base
    return out


def lectura(desc_hoy, tasas, min_casos) -> dict:
    """Lectura histórica del descuento actual (no es una recomendación)."""
    tramo = None
    for u in UMBRALES:
        if desc_hoy is not None and desc_hoy <= -u:
            tramo = u
    if tramo is None:
        return {"nivel": "neutro", "tramo": None,
                "texto": "Cotiza en su nivel habitual del último año: no hay descuento relevante."}
    base = tasas["cualquier_dia"].get("6m", {})
    usado = next((u for u in sorted([u for u in UMBRALES if u <= tramo], reverse=True)
                  if tasas[str(u)]["6m"].get("n", 0) >= min_casos), None)
    if usado is None:
        n = tasas[str(tramo)]["6m"].get("n", 0)
        return {"nivel": "gris", "tramo": tramo,
                "texto": f"Solo {n} precedentes con un descuento así: la estadística no es fiable."}
    t = tasas[str(usado)]["6m"]
    ref = base.get("pct") or 0
    ventaja = (t["pct"] or 0) - ref
    if t["pct"] >= 70 and ventaja >= 3:
        nivel = "verde"
    elif t["pct"] >= 55:
        nivel = "ambar"
    else:
        nivel = "rojo"
    nota = f" (con ≥{tramo}% hay pocos casos; se usa ≥{usado}%)" if usado < tramo else ""
    texto = (f"Tras descuentos ≥{usado}%{nota}, a 6 meses estaba más alto en {t['pct']:.0f}% "
             f"de {t['n']} casos (cualquier día: {ref:.0f}%). Mediana {t['med']:+.1f}%, "
             f"escenario malo {t['malo']:+.1f}%.")
    return {"nivel": nivel, "tramo": tramo, "usado": usado, "pct": t["pct"], "base": ref,
            "ventaja": r(ventaja, 0), "n": t["n"], "med": t["med"], "malo": t["malo"], "texto": texto}


def metricas(s: pd.Series, p80: np.ndarray, hoy: str) -> dict:
    v = s.values
    ultimo = v[-1]
    f = s.index[-1].strftime("%Y-%m-%d")
    prev = s[s.index.year < s.index[-1].year]
    ult = v[-VENTANA:]
    rets = np.diff(ult) / ult[:-1]
    return {
        "precio": r(ultimo, 4 if ultimo < 10 else 2), "fecha": f, "en_sesion": f == hoy,
        "var_1d": var(v, 1), "var_5d": var(v, 5), "var_1m": var(v, 21), "var_3m": var(v, 63),
        "var_1a": var(v, 252), "var_ytd": r((ultimo / prev.iloc[-1] - 1) * 100) if len(prev) else None,
        "p80": r(p80[-1]), "desc": r((ultimo / p80[-1] - 1) * 100),
        "max_1a": r(ult.max()), "min_1a": r(ult.min()),
        "volatilidad": r(rets.std() * math.sqrt(252) * 100, 1),
        "historia_desde": s.index[0].strftime("%Y"),
    }


def guardar_serie(ticker: str, s: pd.Series, p80: np.ndarray):
    n = 5 * 252 + 5
    d = s.iloc[-n:]
    pp = p80[-n:]
    mensual = s.resample("ME").last().dropna()
    datos = {"t": ticker,
             "d": [[k.strftime("%Y-%m-%d"), sig(v), sig(q)] for (k, v), q in zip(d.items(), pp)],
             "m": [[k.strftime("%Y-%m"), sig(v)] for k, v in mensual.items()]}
    (SERIES / f"{nombre_fichero(ticker)}.json").write_text(json.dumps(datos, separators=(",", ":")),
                                                           encoding="utf-8")


def fundamentales(ticker: str) -> dict:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001
        return {}
    k = {"marketCap": "capitalizacion", "forwardPE": "per_futuro", "trailingPE": "per",
         "revenueGrowth": "crec_ventas", "earningsGrowth": "crec_beneficio",
         "profitMargins": "margen", "debtToEquity": "deuda_fp", "dividendYield": "dividendo",
         "recommendationKey": "consenso", "targetMeanPrice": "precio_objetivo",
         "numberOfAnalystOpinions": "n_analistas", "industry": "industria"}
    out = {v: info.get(kk) for kk, v in k.items() if info.get(kk) is not None}
    for p in ("crec_ventas", "crec_beneficio", "margen"):
        if p in out:
            out[p] = r(out[p] * 100, 1)
    for p in ("per_futuro", "per", "deuda_fp", "precio_objetivo"):
        if p in out:
            out[p] = r(out[p], 1)
    if "dividendo" in out:
        d = out["dividendo"]
        out["dividendo"] = r(d if d > 1 else d * 100, 2)
    resumen = info.get("longBusinessSummary")
    if resumen:
        out["descripcion_en"] = resumen[:700]
    return out


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    radar = cfg.get("radar", {})
    min_casos = radar.get("min_casos", 8)
    q = cfg.get("referencia_percentil", 80)
    SERIES.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Madrid"))
    hoy = ahora.strftime("%Y-%m-%d")

    cartera = []
    if CARTERA.exists():
        cartera = [c for c in (yaml.safe_load(CARTERA.read_text(encoding="utf-8")) or {}).get("tickers", []) if c]

    acciones = sp500() + cfg.get("extra_us", []) + cfg.get("europa", [])
    vistos, acciones_u = set(), []
    for a in acciones:
        if a["ticker"] not in vistos:
            vistos.add(a["ticker"])
            acciones_u.append(a)
    print(f"Universo: {len(acciones_u)} acciones")

    fijos = [("indice", e) for e in cfg["indices"]] + [("contexto", e) for e in cfg["contexto"]] + \
            [("sector", e) for e in cfg["sectores"]]
    tickers = [e["ticker"] for _, e in fijos] + [e.get("ticker_ucits") for e in cfg["sectores"]] + \
              [a["ticker"] for a in acciones_u] + cartera
    series, errores = descargar(tickers)
    print(f"Series: {len(series)} · errores: {len(errores)}")
    if len(series) < len(set(tickers)) * 0.6:
        print("Demasiados errores de descarga; no se sobrescriben los datos.")
        sys.exit(1)

    def calcular(t: str, base: dict, tipo: str) -> dict | None:
        if t not in series:
            return None
        s = series[t]
        p80 = p80_movil(s.values, q)
        desc = (s.values / p80 - 1) * 100
        it = {"ticker": t, "nombre": base.get("nombre", t), "tipo": tipo,
              "moneda": moneda(t, base.get("moneda")), "region": region(t),
              "sector": base.get("sector"), **metricas(s, p80, hoy)}
        if tipo != "contexto":
            it["tasas"] = tasas_historicas(s, desc)
            it["lectura"] = lectura(it["desc"], it["tasas"], min_casos)
        it["_s"], it["_p80"] = s, p80
        return it

    # --- Mercado ---------------------------------------------------------
    mercado = {"indices": [], "contexto": [], "sectores": []}
    for tipo, e in fijos:
        it = calcular(e["ticker"], e, tipo)
        if not it:
            continue
        it["comprable"], it["isin"] = e.get("comprable"), e.get("isin") or None
        if tipo == "sector":
            it["sector"] = e["nombre"]
            tu = e.get("ticker_ucits")
            iu = calcular(tu, {"nombre": e.get("comprable") or f"Xtrackers MSCI World {e['nombre']}"}, "ucits") if tu else None
            if iu:
                it["ucits"] = {k: iu[k] for k in ("ticker", "nombre", "moneda", "precio", "fecha", "var_1d",
                                                  "var_5d", "var_1m", "var_1a", "desc", "p80", "historia_desde")}
                it["ucits"]["isin"] = e.get("isin") or None
                guardar_serie(tu, iu["_s"], iu["_p80"])
        guardar_serie(it["ticker"], it.pop("_s"), it.pop("_p80"))
        mercado[{"indice": "indices", "contexto": "contexto", "sector": "sectores"}[tipo]].append(it)

    # --- Acciones ----------------------------------------------------------
    todas = []
    for a in acciones_u:
        it = calcular(a["ticker"], a, "accion")
        if it:
            todas.append(it)
    print(f"Acciones calculadas: {len(todas)}")

    def elegible(it):
        l = it["lectura"]
        return (it["desc"] is not None and it["desc"] <= radar.get("accion_min_descuento", -12)
                and l["nivel"] in ("verde", "ambar")
                and (it["volatilidad"] or 0) <= radar.get("max_volatilidad", 60)
                and (l.get("malo") if l.get("malo") is not None else -99) >= radar.get("peor_escenario_min", -30))

    cand = sorted([x for x in todas if elegible(x)],
                  key=lambda x: (x["lectura"].get("ventaja") or 0, x["lectura"].get("med") or 0), reverse=True)
    elegidas, por_sector = [], {}
    for x in cand:
        sct = x.get("sector") or "?"
        if por_sector.get(sct, 0) >= radar.get("max_por_sector", 3):
            continue
        por_sector[sct] = por_sector.get(sct, 0) + 1
        elegidas.append(x)
        if len(elegidas) >= radar.get("n_acciones", 30):
            break
    reserva = [x for x in cand if x not in elegidas][:30]
    print(f"Radar: {len(cand)} elegibles · {len(elegidas)} en radar")

    detalle = {x["ticker"] for x in elegidas + reserva} | set(cartera)
    oportunidades, en_cartera = [], []
    for x in todas:
        if x["ticker"] not in detalle:
            continue
        x["radar"] = "principal" if x in elegidas else ("reserva" if x in reserva else None)
        x["fund"] = fundamentales(x["ticker"])
        guardar_serie(x["ticker"], x.pop("_s"), x.pop("_p80"))
        if x["radar"]:
            oportunidades.append(x)
        if x["ticker"] in cartera:
            en_cartera.append(x["ticker"])
    oportunidades.sort(key=lambda x: (x["radar"] != "principal", -(x["lectura"].get("ventaja") or 0)))
    # posiciones fuera del universo (p. ej. un ETF concreto)
    extra = []
    for t in cartera:
        if t in detalle and any(x["ticker"] == t for x in todas):
            continue
        if any(i["ticker"] == t or i.get("ucits", {}).get("ticker") == t
               for g in mercado.values() for i in g):
            continue
        it = calcular(t, {"nombre": t}, "accion")
        if it:
            guardar_serie(t, it.pop("_s"), it.pop("_p80"))
            it["fund"] = fundamentales(t)
            extra.append(it)

    universo = [[x["ticker"], x["nombre"], x.get("sector"), x["region"], x["moneda"], x["precio"],
                 x["desc"], x["var_5d"], x["var_1m"], x["var_1a"], x["lectura"]["nivel"],
                 x["lectura"].get("pct"), x["lectura"].get("base"), x["lectura"].get("n")]
                for x in todas]
    for x in todas:
        x.pop("_s", None)
        x.pop("_p80", None)

    # --- Alertas -----------------------------------------------------------
    a = cfg.get("alertas", {})
    alertas = []
    for it in mercado["indices"]:
        if it["var_5d"] is not None and it["var_5d"] <= a.get("indice_5d", -5):
            alertas.append({"clave": f"{it['ticker']}|5d", "ticker": it["ticker"], "tipo": "indice",
                            "texto": f"{it['nombre']}: {it['var_5d']:+.1f}% en 5 sesiones"})
        if it["desc"] is not None and it["desc"] <= a.get("indice_vs_p80", -8):
            alertas.append({"clave": f"{it['ticker']}|p80", "ticker": it["ticker"], "tipo": "indice",
                            "texto": f"{it['nombre']}: {it['desc']:+.1f}% bajo su nivel habitual"})
    for it in mercado["sectores"]:
        if it["desc"] is not None and it["desc"] <= a.get("sector_vs_p80", -12):
            alertas.append({"clave": f"{it['ticker']}|p80", "ticker": it["ticker"], "tipo": "sector",
                            "texto": f"Sector {it['nombre']}: {it['desc']:+.1f}% bajo su nivel habitual"})
    for x in [o for o in oportunidades if o["radar"] == "principal"][:10]:
        alertas.append({"clave": f"{x['ticker']}|radar", "ticker": x["ticker"], "tipo": "accion",
                        "texto": f"{x['nombre']} entra en el radar ({x['desc']:+.1f}% bajo su nivel habitual)"})

    estado_f = SALIDA / "alertas_estado.json"
    previas = set(json.loads(estado_f.read_text(encoding="utf-8")).get("claves", [])) if estado_f.exists() else set()
    nuevas = [al for al in alertas if al["clave"] not in previas]
    estado_f.write_text(json.dumps({"fecha": hoy, "claves": sorted(al["clave"] for al in alertas)}), encoding="utf-8")
    nuevas_f = RAIZ / "alertas_nuevas.md"
    if nuevas:
        lin = [f"Novedades del {hoy}:", ""] + [f"- {al['texto']}" for al in nuevas]
        lin += ["", f"Abre RadarBolsa: {cfg.get('panel_url', '')}", "",
                "_Lectura histórica, no recomendación de inversión._"]
        nuevas_f.write_text("\n".join(lin), encoding="utf-8")
    elif nuevas_f.exists():
        nuevas_f.unlink()

    resumen = {
        "actualizado": ahora.strftime("%Y-%m-%d %H:%M"), "hoy": hoy,
        "mercado": mercado, "oportunidades": oportunidades, "cartera_extra": extra,
        "alertas": alertas, "alertas_nuevas": [al["clave"] for al in nuevas],
        "errores": errores[:50], "n_errores": len(errores), "n_universo": len(todas),
        "n_elegibles": len(cand),
        "metodo": {"horizontes": HORIZONTES, "umbrales": UMBRALES, "desde": DESDE,
                   "min_casos": min_casos, "percentil": q, "radar": radar},
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, separators=(",", ":")),
                                         encoding="utf-8")
    (SALIDA / "universo.json").write_text(json.dumps(universo, ensure_ascii=False, separators=(",", ":")),
                                          encoding="utf-8")
    print(f"OK · universo {len(todas)} · radar {len(oportunidades)} · alertas {len(alertas)} ({len(nuevas)} nuevas)")


if __name__ == "__main__":
    main()
