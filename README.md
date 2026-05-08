# Auditor de Sesgo en CVs — MVP

**Caso 2:** Sesgo en selección de CVs | Introducción a la Ingeniería 2026-I — UNINÚÑEZ

## Requisitos

```bash
python -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
```

## Pasos para ejecutar

### 1. Descargar el dataset
```bash
mkdir -p data/raw
curl -o data/raw/adult.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

### 2. Entrenar los modelos
```bash
python model/train.py
```
Genera `model/modelo_sesgado.pkl` y `model/modelo_justo.pkl`.

### 3. Explorar el sesgo (notebook)
```bash
jupyter notebook notebooks/exploracion.ipynb
```
Ejecutar todas las celdas — guarda gráficas en `data/processed/graficas/`.

### 4. Iniciar la API
```bash
cd api && python app.py
# Disponible en http://localhost:5000
```

## Endpoints

### `POST /analizar`
```bash
curl -X POST http://localhost:5000/api/v1/analizar \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 45, "tipo_trabajo": "Private",
    "educacion_num": 13, "estado_civil": "Married-civ-spouse",
    "ocupacion": "Prof-specialty", "sexo": "Male",
    "horas_semana": 40, "pais_origen": "United-States",
    "modelo": "justo"
  }'
```

### `POST /analizar-lote`
```bash
curl -X POST http://localhost:5000/api/v1/analizar-lote \
  -H "Content-Type: application/json" \
  -d '{"cvs": [{...}, {...}], "modelo": "justo"}'
```

## Estructura
```
cv-sesgo-auditor/
├── data/
│   ├── raw/               ← adult.csv (descargar manualmente)
│   └── processed/graficas/← gráficas .png
├── notebooks/exploracion.ipynb
├── model/
│   ├── train.py
│   ├── predictor.py
│   ├── modelo_sesgado.pkl (generado)
│   └── modelo_justo.pkl   (generado)
├── api/
│   ├── app.py
│   └── requirements.txt
└── README.md
```
