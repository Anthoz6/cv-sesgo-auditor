"""
API Flask — dos endpoints versionados bajo /api/v1:
  POST /api/v1/analizar       → predicción individual
  POST /api/v1/analizar-lote  → predicción por lotes
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, Blueprint, request, jsonify
from model.predictor import predecir

app = Flask(__name__)
v1  = Blueprint("v1", __name__, url_prefix="/api/v1")


@v1.route("/analizar", methods=["POST"])
def analizar():
    datos = request.get_json(force=True)
    if not datos:
        return jsonify({"error": "Body JSON requerido"}), 400

    modelo    = datos.pop("modelo", "justo")
    resultado = predecir(datos, modelo=modelo)
    return jsonify(resultado)


@v1.route("/analizar-lote", methods=["POST"])
def analizar_lote():
    body = request.get_json(force=True)
    if not body or "cvs" not in body:
        return jsonify({"error": "Campo 'cvs' requerido"}), 400

    cvs    = body["cvs"]
    modelo = body.get("modelo", "justo")

    resultados = []
    for cv in cvs:
        resultados.append(predecir(dict(cv), modelo=modelo))

    aceptados  = sum(1 for r in resultados if r["decision"] == "aceptado")
    rechazados = len(resultados) - aceptados

    return jsonify({
        "resultados" : resultados,
        "total"      : len(resultados),
        "aceptados"  : aceptados,
        "rechazados" : rechazados,
    })


app.register_blueprint(v1)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
