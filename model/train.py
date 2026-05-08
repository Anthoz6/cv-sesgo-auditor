"""
Entrenamiento de dos modelos RandomForest:

  modelo_sesgado — usa todas las variables (incluyendo protegidas).
                   Reproduce el sesgo histórico del dataset.

  modelo_justo   — técnica combinada:
                   1. Elimina variables protegidas + proxy (relacion)
                   2. Oversampling para balancear grupos en entrenamiento
                   3. Calibración de umbrales por grupo (post-processing
                      demographic parity): el RF no usa sexo como feature,
                      pero los umbrales de decisión se calibran por sexo
                      para igualar tasas de aceptación.
                   Distinción clave: usar sexo para IGUALAR ≠ usar sexo
                   para DISCRIMINAR (principio de fairness through awareness).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(BASE_DIR, "../data/raw/adult.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "../data/processed/adult_clean.csv")

COLUMNAS = [
    "edad", "tipo_trabajo", "fnlwgt", "educacion",
    "educacion_num", "estado_civil", "ocupacion",
    "relacion", "raza", "sexo", "ganancia_capital",
    "perdida_capital", "horas_semana", "pais_origen", "ingreso"
]

VARIABLES_PROTEGIDAS = ["edad", "sexo", "raza", "pais_origen"]
VARIABLES_PROXY      = ["relacion"]  # Husband≈Male, Wife≈Female

CATEGORICAS = [
    "tipo_trabajo", "educacion", "estado_civil",
    "ocupacion", "relacion", "raza", "sexo", "pais_origen"
]


def cargar_y_limpiar():
    df = pd.read_csv(RAW_PATH, names=COLUMNAS, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    df["ingreso"] = df["ingreso"].str.strip().map(
        {">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0})
    df.dropna(subset=["ingreso"], inplace=True)
    df["ingreso"] = df["ingreso"].astype(int)
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Dataset limpio: {len(df)} filas")
    return df


def codificar(df, columnas_cat):
    df = df.copy()
    encoders = {}
    for col in columnas_cat:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def oversample_equitativo(df_train):
    """Oversampling: iguala tasa de aceptación de mujeres a la de hombres."""
    male_acc = df_train[(df_train["sexo"] == "Male")   & (df_train["ingreso"] == 1)]
    male_rej = df_train[(df_train["sexo"] == "Male")   & (df_train["ingreso"] == 0)]
    fem_acc  = df_train[(df_train["sexo"] == "Female") & (df_train["ingreso"] == 1)]
    fem_rej  = df_train[(df_train["sexo"] == "Female") & (df_train["ingreso"] == 0)]

    tasa_male        = len(male_acc) / (len(male_acc) + len(male_rej))
    n_female         = len(fem_acc) + len(fem_rej)
    n_fem_acc_target = int(n_female * tasa_male)

    fem_acc_over = fem_acc.sample(n_fem_acc_target, replace=True, random_state=42)
    df_bal = pd.concat(
        [male_acc, male_rej, fem_acc_over, fem_rej]
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Training original  : {len(df_train)} filas  "
          f"(F aceptadas {len(fem_acc)}, M aceptados {len(male_acc)})")
    print(f"  Training balanceado: {len(df_bal)} filas  "
          f"(F aceptadas {n_fem_acc_target}, M aceptados {len(male_acc)})")
    return df_bal


def calibrar_umbrales(clf, X_calib, df_calib_orig, features):
    """
    Post-processing demographic parity:
    Para cada grupo de sexo, encuentra el umbral de probabilidad que produce
    la misma tasa de aceptación que la tasa global de aceptación del dataset.
    Así, Female y Male tienen la misma probabilidad de ser aceptados.
    """
    probas = clf.predict_proba(X_calib)[:, 1]
    df_c   = df_calib_orig.copy()
    df_c["proba"] = probas

    tasa_global = df_c["ingreso"].mean()
    print(f"\n  Calibrando umbrales (tasa objetivo: {tasa_global:.1%})")

    umbrales = {}
    for sexo in df_c["sexo"].unique():
        mask          = df_c["sexo"] == sexo
        probas_grupo  = df_c.loc[mask, "proba"].values
        # Umbral: percentil tal que la tasa de aceptación sea tasa_global
        umbral = np.percentile(probas_grupo, (1 - tasa_global) * 100)
        umbrales[sexo] = float(umbral)
        tasa_resultado = (probas_grupo >= umbral).mean()
        print(f"    {sexo}: umbral={umbral:.3f}  → tasa aceptación={tasa_resultado:.1%}")

    return umbrales


def entrenar_modelo(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def metricas_equidad(clf, X_all, df_original, features, umbrales=None):
    probas = clf.predict_proba(X_all)[:, 1]
    df_eval = df_original.copy()

    if umbrales:
        preds = np.zeros(len(df_eval), dtype=int)
        for sexo, umbral in umbrales.items():
            mask = (df_eval["sexo"] == sexo).values
            preds[mask] = (probas[mask] >= umbral).astype(int)
    else:
        preds = (probas >= 0.5).astype(int)

    df_eval["pred"] = preds

    print("\n  Tasa de aceptación por sexo:")
    for sexo, grp in df_eval.groupby("sexo"):
        print(f"    {sexo}: {grp['pred'].mean():.1%}")

    print("\n  Tasa de aceptación por rango de edad:")
    bins   = [0, 30, 45, 60, 120]
    labels = ["18-30", "31-45", "46-60", "60+"]
    df_eval["rango"] = pd.cut(df_eval["edad"], bins=bins, labels=labels)
    for rango, grp in df_eval.groupby("rango", observed=True):
        print(f"    {rango}: {grp['pred'].mean():.1%}")


def main():
    print("── Cargando y limpiando datos ──────────────────────────────")
    df = cargar_y_limpiar()

    cat_sesgado = CATEGORICAS
    cat_justo   = [c for c in CATEGORICAS
                   if c not in VARIABLES_PROTEGIDAS + VARIABLES_PROXY]

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    df_train_orig = df.iloc[idx_train].copy()

    # ── Modelo SESGADO ────────────────────────────────────────────────────────
    print("\n── Modelo SESGADO (todas las variables) ────────────────────")
    df_cod_s, enc_s = codificar(df, cat_sesgado)
    features_s = [c for c in df_cod_s.columns if c != "ingreso"]

    X_s_train = df_cod_s[features_s].iloc[idx_train]
    X_s_test  = df_cod_s[features_s].iloc[idx_test]
    y_train   = df_cod_s["ingreso"].iloc[idx_train]
    y_test    = df_cod_s["ingreso"].iloc[idx_test]

    m_s      = entrenar_modelo(X_s_train, y_train)
    y_pred_s = m_s.predict(X_s_test)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_s):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_s):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_s):.4f}")
    metricas_equidad(m_s, df_cod_s[features_s], df, features_s)

    # ── Modelo JUSTO ──────────────────────────────────────────────────────────
    print("\n── Modelo JUSTO (oversample + calibración de umbrales) ─────")
    df_train_bal = oversample_equitativo(df_train_orig)

    df_cod_j_train, enc_j = codificar(df_train_bal, cat_justo)
    df_cod_j_full,  _     = codificar(df, cat_justo)

    excluir    = set(VARIABLES_PROTEGIDAS + VARIABLES_PROXY + ["ingreso"])
    features_j = [c for c in df_cod_j_train.columns if c not in excluir]

    X_j_train     = df_cod_j_train[features_j]
    y_j_train     = df_cod_j_train["ingreso"]
    X_j_calib     = df_cod_j_full[features_j].iloc[idx_train]
    X_j_test      = df_cod_j_full[features_j].iloc[idx_test]

    m_j      = entrenar_modelo(X_j_train, y_j_train)
    y_pred_j = m_j.predict(X_j_test)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_j):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_j):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_j):.4f}")

    umbrales = calibrar_umbrales(m_j, X_j_calib, df_train_orig, features_j)
    metricas_equidad(m_j, df_cod_j_full[features_j], df, features_j, umbrales=umbrales)

    # ── Guardar ───────────────────────────────────────────────────────────────
    with open(os.path.join(BASE_DIR, "modelo_sesgado.pkl"), "wb") as f:
        pickle.dump({"modelo": m_s, "features": features_s, "encoders": enc_s}, f)
    with open(os.path.join(BASE_DIR, "modelo_justo.pkl"), "wb") as f:
        pickle.dump({"modelo": m_j, "features": features_j,
                     "encoders": enc_j, "umbrales": umbrales}, f)

    print("\n── Modelos guardados ─────────────────────────────────────────")


if __name__ == "__main__":
    main()
