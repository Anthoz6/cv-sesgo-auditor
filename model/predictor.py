"""
Función predecir() — carga un modelo guardado y retorna
decisión, confianza y factores SHAP de sesgo.

Modelo sesgado: umbral fijo 0.5, usa todas las variables.
Modelo justo:   umbral calibrado por sexo (demographic parity),
                el RF no usa sexo como feature, pero el umbral de
                decisión se ajusta por grupo para igualar tasas.
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _cargar_meta(tipo_modelo: str) -> dict:
    nombre = "modelo_sesgado.pkl" if tipo_modelo == "sesgado" else "modelo_justo.pkl"
    with open(os.path.join(BASE_DIR, nombre), "rb") as f:
        return pickle.load(f)


def _preparar_entrada(datos_cv: dict, features: list, encoders: dict) -> pd.DataFrame:
    fila = {}
    for feat in features:
        valor = datos_cv.get(feat, 0)
        if feat in encoders:
            le        = encoders[feat]
            valor_str = str(valor)
            valor     = int(le.transform([valor_str])[0]) if valor_str in le.classes_ else 0
        fila[feat] = valor
    return pd.DataFrame([fila])


def predecir(datos_cv: dict, modelo: str = "justo") -> dict:
    """
    Parámetros
    ----------
    datos_cv : dict  — campos del CV
    modelo   : str   — "justo" (default) o "sesgado"

    Retorna
    -------
    {
        "decision"      : "aceptado" | "rechazado",
        "confianza"     : float,
        "factores_sesgo": [{"variable": str, "impacto": float}]
    }
    """
    meta     = _cargar_meta(modelo)
    clf      = meta["modelo"]
    features = meta["features"]
    encoders = meta["encoders"]
    umbrales = meta.get("umbrales", {})

    X = _preparar_entrada(datos_cv, features, encoders)

    proba = clf.predict_proba(X)[0][1]

    # Umbral de decisión
    if modelo == "justo" and umbrales:
        sexo    = datos_cv.get("sexo", None)
        umbral  = umbrales.get(sexo, 0.5)
    else:
        umbral = 0.5

    clase     = 1 if proba >= umbral else 0
    confianza = float(proba) if clase == 1 else float(1 - proba)

    # ── Explicación SHAP ──────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0, :, 1]

    factores = sorted(
        [{"variable": f, "impacto": round(float(v), 4)} for f, v in zip(features, vals)],
        key=lambda x: abs(x["impacto"]),
        reverse=True
    )[:5]

    return {
        "decision"      : "aceptado" if clase == 1 else "rechazado",
        "confianza"     : round(confianza, 4),
        "factores_sesgo": factores,
    }
