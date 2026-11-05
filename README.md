# 🩺 Estimación del riesgo de diabetes — Prototipo académico

Proyecto de **Minería de Datos** · Universidad de la Costa (CUC) · Tercer corte, Actividad 1: *Integración y despliegue del modelo con Streamlit*.

> ⚠️ \*\*Aviso:\*\* prototipo académico. \*\*No sustituye el diagnóstico ni la decisión de un profesional de la salud.\*\*

**Aplicación pública:** `https://diabetes-app-mineria-szbknhebcvpppm2hjs27mf.streamlit.app/`

## Integrantes

|Nombre completo|Código estudiantil|
|-|-|
|Jesús Alberto Vargas España|jvargas72@cuc.edu.co|
|Santiago Patiño Arzayus|spatino2@cuc.edu.co|
|Duván Andrés Arzuza Carrascal|darzuza@cuc.edu.co|

## Objetivo

Estimar si una persona presenta un perfil compatible con **diabetes o prediabetes** a partir de 21 indicadores clínicos, de estilo de vida y sociodemográficos autorreportados, priorizando la **sensibilidad** (no pasar por alto casos) como criterio de tamizaje.

## Datos

*Diabetes Health Indicators Dataset* (BRFSS 2015, CDC), versión balanceada 50/50 (`diabetes\_binary\_5050split\_health\_indicators\_BRFSS2015.csv`): [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
70 692 filas → **69 057** tras eliminar 1 635 duplicados exactos; sin valores nulos. La app **no** usa el CSV; solo el modelo serializado y datos agregados del EDA (`artefactos/`).

## Estructura del repositorio

```
app.py                               # aplicación Streamlit (solo inferencia; no entrena)
requirements.txt                     # dependencias con versiones exactas
README.md
Proyecto\_Mineria\_Corte3.ipynb        # notebook actualizado (limpieza, CV, optimización, umbral, joblib)
artefactos/
  modelo\_diabetes\_pipeline.joblib    # pipeline completo: StandardScaler + modelo
  metadata\_modelo.json               # umbral, métricas, matriz de confusión, rangos, versiones
  eda\_stats.json, eda\_muestra.csv    # datos agregados para los gráficos de la app
  registros\_conocidos.json           # registros de X\_test para verificar app ↔ notebook
tests/test\_app.py                    # pruebas automáticas (caso normal, entrada inválida, registros conocidos)
```

## Ejecución local

```bash
python -m venv .venv \&\& source .venv/bin/activate      # en Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

El `.joblib` fue creado con **scikit-learn 1.8.0** (Python 3.12.3); `requirements.txt` fija esas versiones. En Streamlit Cloud elegir **Python 3.12** en *Advanced settings*.

## Reproducir el modelo

Abrir `Proyecto\_Mineria\_Corte3.ipynb` (Colab o local), subir el CSV y ejecutar *Entorno de ejecución → Ejecutar todo*. Regenera `artefactos/` y `requirements.txt`. Si se reentrena en otro entorno, **usar el `requirements.txt` generado** para que las versiones coincidan con las del `.joblib`.

## Modelo y resultados

Modelo final: **HistGradientBoosting (optimizado)** dentro de un `Pipeline` (estandarización de variables numéricas/ordinales + clasificador). Hiperparámetros: `{'min\_samples\_leaf': 20, 'max\_leaf\_nodes': 15, 'max\_depth': 4, 'learning\_rate': 0.03, 'l2\_regularization': 0.1}`.

|Etapa|Métrica|Gradient Boosting (corte 2)|Modelo final|
|-|-|-|-|
|Validación cruzada 5 folds|ROC-AUC|0.8259|0.8259 ± 0.0042|
|Validación cruzada 5 folds|Recall|0.7953|0.7974 ± 0.0057|

Evaluación en el conjunto de prueba (n = 13812), usado una sola vez:

|Umbral|Accuracy|Precisión|**Recall**|Especificidad|F1|FN|FP|
|-|-|-|-|-|-|-|-|
|0.50|0.748|0.731|0.800|0.695|0.764|1406|2071|
|**0.43 (elegido)**|0.748|0.708|**0.858**|0.635|0.776|996|2479|

**Por qué se prioriza el Recall:** en tamizaje, un falso negativo retrasa el diagnóstico; un falso positivo implica una prueba de confirmación. El umbral se eligió **solo con validación cruzada sobre entrenamiento**: el más alto que alcanza una sensibilidad ≥ 85%.
La optimización de hiperparámetros **no cambió de forma apreciable el ROC-AUC**; la ganancia de sensibilidad proviene de la decisión del umbral.

## Pruebas de calidad

Ejecutar desde la raíz: `python tests/test\_app.py` (o `pytest -q tests/test\_app.py`).

|Prueba|Entrada|Resultado esperado|
|-|-|-|
|1. Caso normal|valores por defecto del formulario|probabilidad en \[0, 1] y clase 0/1|
|2. Entrada inválida|peso 25 kg y talla 230 cm (IMC ≈ 5, imposible)|la app muestra un error y **no** predice|
|3. Registros conocidos de `X\_test`|los 3 registros siguientes|la app reproduce la probabilidad y la clase del notebook|

|Caso|Índice original|Real|Probabilidad (notebook)|Clase|
|-|-|-|-|-|
|verdadero positivo|48151|1|0.719744|1|
|verdadero negativo|19201|0|0.333070|0|
|falso negativo|56380|1|0.419943|0|

Para la prueba manual en el navegador: IMC = peso ÷ talla²; usar talla 170 cm y peso = IMC × 2.89 kg para reproducir el IMC entero del registro (los valores completos están en `artefactos/registros\_conocidos.json`).

## Limitaciones

1. Datos **balanceados artificialmente**: la probabilidad no es el riesgo real; con prevalencia ≈ 13.9% el valor predictivo positivo estimado sería \~28%.
2. El objetivo mezcla diabetes y prediabetes.
3. Encuesta autorreportada de EE. UU. (2015); sin validación en población colombiana; ingresos en dólares.
4. Sin variables de laboratorio (glucosa, HbA1c).
5. Con el umbral elegido el modelo deja sin detectar \~14% de los casos reales.

## Despliegue en Streamlit Community Cloud

1. Subir todo el contenido de este repositorio a GitHub (repositorio **público**).
2. En [https://share.streamlit.io](https://share.streamlit.io) → *Create app* → elegir el repositorio, rama principal y archivo `app.py`; en *Advanced settings* elegir Python 3.12.
3. Verificar la URL pública desde una ventana de incógnito y pegarla arriba.

## Evaluación del uso de Inteligencia Artificial

> \*\*El equipo debe revisar este apartado y ajustarlo a lo que realmente hizo.\*\* Todos los integrantes deben comprender el código entregado.

* **Herramientas:** Gemini (borradores del informe y del código del corte 2) y Claude (auditoría del notebook del corte 2, reescritura del notebook, `app.py`, pruebas y documentación del tercer corte).
* **Propósito:** revisar el pipeline, corregir problemas de reproducibilidad, optimizar y validar el modelo, construir la aplicación y documentar.
* **Prompt principal:** se pidió a Claude, adjuntando el PDF de la actividad, el informe y el notebook del corte 2, que actuara como profesor de Minería de Datos para auditar errores, optimizar el modelo con validación cruzada, serializarlo con `joblib` y construir `app.py`, `requirements.txt`, `README.md` y un protocolo de pruebas.
* **Cómo se verificó:** ejecución completa del notebook de arriba hacia abajo con el CSV original; reproducción de las métricas del corte 2; prueba de ida y vuelta del `joblib`; pruebas automáticas que comparan las predicciones de la app con las del notebook; revisión de las cifras del informe contra los datos.
* **Problemas encontrados:** notebook del corte 2 con celdas ejecutadas en desorden y salidas obsoletas (columna `Cluster`); `drop\_duplicates` solo impreso y no aplicado; sugerencia `df = df.drop\_duplicates(inplace=True)` (deja `df = None`); *macro-recall* presentado como sensibilidad; escalado solo de `bmi`; silueta de K-Means mencionada sin cálculo; cifras del informe (matriz de confusión y prevalencias de `highbp`/`highchol`) que no coincidían con los datos.
* **Ajustes realizados:** limpieza reproducible, `Pipeline` con escalado completo, validación cruzada, umbral elegido sin usar la prueba, cifras de las conclusiones generadas por el propio notebook, serialización verificada y corrección del informe.

## Seguridad

El repositorio no contiene contraseñas, tokens ni datos personales; los datos son respuestas de encuesta anonimizadas.

