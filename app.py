"""
Prototipo académico · Riesgo de diabetes (BRFSS 2015)
Minería de Datos · Universidad de la Costa (CUC) · Tercer corte
La app NO entrena: solo carga el pipeline serializado (preprocesamiento + modelo) y hace inferencia.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Riesgo de diabetes · Prototipo académico", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent / "artefactos"

# >>> COMPLETAR los códigos estudiantiles antes de desplegar <<<
INTEGRANTES = [
    ("Jesús Alberto Vargas España", "jvargas72@cuc.edu.co"),
    ("Santiago Patiño Arzayus", "spatino2@cuc.edu.co"),
    ("Duván Andrés Arzuza Carrascal", "darzuza@cuc.edu.co"),
]
FUENTE = ("Diabetes Health Indicators Dataset (BRFSS 2015, CDC), versión balanceada 50/50 — Kaggle: "
          "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")

EDADES = {1: "18–24 años", 2: "25–29 años", 3: "30–34 años", 4: "35–39 años", 5: "40–44 años", 6: "45–49 años",
          7: "50–54 años", 8: "55–59 años", 9: "60–64 años", 10: "65–69 años", 11: "70–74 años", 12: "75–79 años", 13: "80 años o más"}
SALUD = {1: "1 · Excelente", 2: "2 · Muy buena", 3: "3 · Buena", 4: "4 · Regular", 5: "5 · Mala"}
EDUCACION = {1: "Nunca asistió / solo preescolar", 2: "Primaria (grados 1–8)", 3: "Secundaria incompleta (grados 9–11)",
             4: "Bachiller (grado 12 / GED)", 5: "Universidad incompleta o técnica (1–3 años)", 6: "Universitario (4 años o más)"}
INGRESOS = {1: "Menos de 10 000 USD/año", 2: "10 000–15 000", 3: "15 000–20 000", 4: "20 000–25 000",
            5: "25 000–35 000", 6: "35 000–50 000", 7: "50 000–75 000", 8: "75 000 USD/año o más"}
SINO = ["No", "Sí"]


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_activos():
    modelo = joblib.load(BASE / "modelo_diabetes_pipeline.joblib")
    meta = json.loads((BASE / "metadata_modelo.json").read_text(encoding="utf-8"))
    eda = json.loads((BASE / "eda_stats.json").read_text(encoding="utf-8"))
    muestra = pd.read_csv(BASE / "eda_muestra.csv")
    return modelo, meta, eda, muestra


def validar_entradas(v: dict, rangos: dict) -> list:
    """Validación de tipos, rangos y categorías. Devuelve la lista de errores (vacía si todo es válido)."""
    errores = []
    categorias = {"age": set(EDADES), "genhlth": set(SALUD), "education": set(EDUCACION), "income": set(INGRESOS)}
    for campo, valor in v.items():
        if isinstance(valor, bool) or not isinstance(valor, (int, float, np.integer, np.floating)) or not np.isfinite(valor):
            errores.append(f"«{campo}» debe ser un número válido.")
            continue
        if campo in categorias and int(valor) not in categorias[campo]:
            errores.append(f"«{campo}» tiene una categoría no permitida ({valor}).")
        lo, hi = rangos[campo]
        if not lo <= valor <= hi:
            errores.append(f"«{campo}» = {valor:g} está fuera del rango observado en los datos de entrenamiento ({lo:g}–{hi:g}).")
    return errores


modelo, meta, eda, muestra = cargar_activos()
UMBRAL = meta["umbral"]
mt = meta["metricas_test_umbral_elegido"]
m05 = meta["metricas_test_umbral_05"]

# ───────────────────────── Encabezado ─────────────────────────
st.title("🩺 Estimación del riesgo de diabetes con Minería de Datos")
st.warning("**Prototipo académico.** Esta aplicación fue desarrollada con fines educativos y **no sustituye el diagnóstico, "
           "la valoración ni la decisión de un profesional de la salud.** Ante cualquier duda sobre su salud, consulte a un médico.", icon="⚠️")

tab_inicio, tab_eda, tab_pred, tab_eval = st.tabs(["🏠 Proyecto", "📊 Análisis exploratorio", "🩺 Predicción", "🔬 Evaluación técnica"])

# ───────────────────────── Proyecto ─────────────────────────
with tab_inicio:
    st.subheader("Presentación del proyecto")
    st.markdown(
        "**Objetivo.** Desarrollar y desplegar un modelo de clasificación que, a partir de indicadores clínicos, de estilo de vida y "
        "sociodemográficos autorreportados, estime si una persona presenta un perfil compatible con **diabetes o prediabetes**, "
        "priorizando la **sensibilidad** (no pasar por alto casos) como criterio de tamizaje.")
    st.markdown("**Asignatura:** Minería de Datos · Universidad de la Costa (CUC)")
    st.markdown("**Integrantes**")
    st.table(pd.DataFrame(INTEGRANTES, columns=["Nombre completo", "Código estudiantil"]))
    st.markdown(f"**Fuente de los datos:** {FUENTE}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros usados (tras limpieza)", f"{eda['n_registros']:,}".replace(",", " "))
    c2.metric("Variables predictoras", len(meta["features"]))
    c3.metric("Modelo final", meta["modelo"])
    st.caption("Metodología CRISP-DM: limpieza → EDA → modelado con validación cruzada → selección por sensibilidad → serialización con joblib → despliegue.")

# ───────────────────────── EDA ─────────────────────────
with tab_eda:
    st.subheader("Tres hallazgos del análisis exploratorio")
    nombres = {0: "Sin diabetes", 1: "Diabetes / prediabetes"}

    st.markdown("**1 · Distribución del IMC (BMI) según el grupo**")
    m = muestra.assign(Grupo=muestra["diabetes_binary"].map(nombres))
    f1 = px.histogram(m[m.bmi <= 60], x="bmi", color="Grupo", barmode="overlay", opacity=0.65, nbins=45, histnorm="probability density",
                      color_discrete_sequence=["#4C78A8", "#E45756"], labels={"bmi": "IMC (kg/m²)"})
    f1.update_layout(height=340, yaxis_title="Densidad")
    st.plotly_chart(f1, width="stretch")
    st.info(f"Interpretación: la mediana del IMC es **{eda['bmi_mediana']['sin_diabetes']:.0f}** en el grupo sin diabetes y "
            f"**{eda['bmi_mediana']['con_diabetes']:.0f}** en el grupo con diabetes/prediabetes: la distribución se desplaza hacia valores más altos. "
            "Hay solapamiento amplio: el IMC por sí solo no basta para clasificar. (Gráfico sobre una muestra de 6 000 registros, IMC ≤ 60.)")

    st.markdown("**2 · Prevalencia de factores de riesgo en cada grupo**")
    prev = pd.DataFrame(eda["prevalencia"]).T.reset_index().rename(columns={"index": "Factor"})
    prev = prev.melt(id_vars="Factor", var_name="Grupo", value_name="Proporción")
    prev["Grupo"] = prev["Grupo"].map({"sin_diabetes": "Sin diabetes", "con_diabetes": "Diabetes / prediabetes"})
    f2 = px.bar(prev, y="Factor", x="Proporción", color="Grupo", barmode="group", orientation="h",
                color_discrete_sequence=["#4C78A8", "#E45756"])
    f2.update_layout(height=420, xaxis_tickformat=".0%")
    st.plotly_chart(f2, width="stretch")
    pr = eda["prevalencia"]
    st.info(f"Interpretación: la presión alta está presente en **{pr['Presión alta']['con_diabetes']:.1%}** del grupo con diabetes/prediabetes frente a "
            f"**{pr['Presión alta']['sin_diabetes']:.1%}** del otro grupo; el colesterol alto, en **{pr['Colesterol alto']['con_diabetes']:.1%}** vs "
            f"**{pr['Colesterol alto']['sin_diabetes']:.1%}**. La actividad física es **más frecuente** en el grupo sin diabetes.")

    st.markdown("**3 · Correlación de Spearman de cada variable con el diagnóstico**")
    cor = pd.Series(eda["correlacion_spearman"]).sort_values()
    f3 = go.Figure(go.Bar(x=cor.values, y=cor.index, orientation="h", marker_color=["#E45756" if v > 0 else "#4C78A8" for v in cor.values]))
    f3.update_layout(height=520, xaxis_title="Coeficiente de Spearman")
    st.plotly_chart(f3, width="stretch")
    top = cor.sort_values(ascending=False).head(3)
    st.info("Interpretación: las asociaciones positivas más altas son " + ", ".join(f"**{k}** ({v:+.2f})" for k, v in top.items()) +
            ". Son **moderadas** (ninguna supera 0.5 en valor absoluto): por eso se usa un modelo multivariable. Correlación no implica causalidad.")

# ───────────────────────── Predicción ─────────────────────────
with tab_pred:
    st.subheader("Ingrese los datos de la persona")
    st.caption("Todos los campos son obligatorios. Los datos no se almacenan.")
    with st.form("form_prediccion"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Datos básicos**")
            peso = st.number_input("Peso (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.1, key="peso")
            talla = st.number_input("Talla (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="talla")
            sexo = st.selectbox("Sexo biológico", ["Femenino", "Masculino"], key="sex")
            edad = st.selectbox("Rango de edad", list(EDADES), format_func=EDADES.get, index=6, key="age")
            educ = st.selectbox("Nivel educativo", list(EDUCACION), format_func=EDUCACION.get, index=3, key="education")
            ingr = st.selectbox("Ingreso anual del hogar (USD, escala del estudio)", list(INGRESOS), format_func=INGRESOS.get, index=4, key="income")
        with b:
            st.markdown("**Salud**")
            genh = st.selectbox("Salud general percibida", list(SALUD), format_func=SALUD.get, index=2, key="genhlth")
            ment = st.number_input("Días de mala salud mental (últimos 30 días)", min_value=0, max_value=30, value=0, step=1, key="menthlth")
            phys = st.number_input("Días de mala salud física (últimos 30 días)", min_value=0, max_value=30, value=0, step=1, key="physhlth")
            hbp = st.selectbox("¿Le han dicho que tiene la presión arterial alta?", SINO, key="highbp")
            hch = st.selectbox("¿Le han dicho que tiene el colesterol alto?", SINO, key="highchol")
            chk = st.selectbox("¿Se midió el colesterol en los últimos 5 años?", SINO, index=1, key="cholcheck")
            dwk = st.selectbox("¿Tiene dificultad seria para caminar o subir escaleras?", SINO, key="diffwalk")
        with c:
            st.markdown("**Antecedentes y hábitos**")
            stk = st.selectbox("¿Ha tenido un accidente cerebrovascular (ACV)?", SINO, key="stroke")
            hrt = st.selectbox("¿Enfermedad coronaria o infarto?", SINO, key="heartdiseaseorattack")
            smk = st.selectbox("¿Ha fumado 100 cigarrillos o más en su vida?", SINO, key="smoker")
            act = st.selectbox("¿Hizo actividad física en los últimos 30 días (fuera del trabajo)?", SINO, index=1, key="physactivity")
            fru = st.selectbox("¿Come fruta al menos una vez al día?", SINO, index=1, key="fruits")
            veg = st.selectbox("¿Come verduras al menos una vez al día?", SINO, index=1, key="veggies")
            alc = st.selectbox("¿Consumo elevado de alcohol? (hombres >14 y mujeres >7 tragos/semana)", SINO, key="hvyalcoholconsump")
            cov = st.selectbox("¿Tiene algún tipo de cobertura médica?", SINO, index=1, key="anyhealthcare")
            ncs = st.selectbox("En el último año, ¿no fue al médico por su costo?", SINO, key="nodocbccost")
        enviar = st.form_submit_button("Calcular estimación", type="primary", key="enviar")

    imc = float(round(peso / (talla / 100) ** 2))
    st.caption(f"IMC calculado: **{imc:.0f} kg/m²** (peso ÷ talla², redondeado al entero, como en los datos de entrenamiento).")

    if enviar:
        yn = lambda s: 1.0 if s == "Sí" else 0.0
        valores = {"highbp": yn(hbp), "highchol": yn(hch), "cholcheck": yn(chk), "bmi": imc, "smoker": yn(smk), "stroke": yn(stk),
                   "heartdiseaseorattack": yn(hrt), "physactivity": yn(act), "fruits": yn(fru), "veggies": yn(veg),
                   "hvyalcoholconsump": yn(alc), "anyhealthcare": yn(cov), "nodocbccost": yn(ncs), "genhlth": float(genh),
                   "menthlth": float(ment), "physhlth": float(phys), "diffwalk": yn(dwk), "sex": 1.0 if sexo == "Masculino" else 0.0,
                   "age": float(edad), "education": float(educ), "income": float(ingr)}
        errores = validar_entradas(valores, meta["rangos"])
        if errores:
            st.session_state["ultima_prediccion"] = None
            st.error("No se puede calcular la estimación. Corrija lo siguiente:\n\n" + "\n".join(f"- {e}" for e in errores))
        else:
            fila = pd.DataFrame([valores])[meta["features"]]
            proba = float(modelo.predict_proba(fila)[0, 1])
            clase = int(proba >= UMBRAL)
            st.session_state["ultima_prediccion"] = {"probabilidad": proba, "clase": clase}
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Probabilidad estimada por el modelo", f"{proba:.1%}")
            r2.metric("Umbral de decisión", f"{UMBRAL:.0%}")
            r3.metric("Resultado", "Perfil de riesgo elevado" if clase else "Por debajo del umbral")
            st.progress(min(max(proba, 0.0), 1.0))
            if clase:
                st.error("El perfil ingresado se parece al de personas con diabetes o prediabetes en los datos de entrenamiento. "
                         "**Esto no es un diagnóstico**: se recomienda consultar a un profesional de la salud para una prueba de glucosa.")
            else:
                st.success("El perfil ingresado no supera el umbral del modelo. **Esto no descarta la enfermedad**: el modelo deja pasar "
                           f"aproximadamente {1 - mt['Recall']:.0%} de los casos reales en las pruebas.")
            st.caption("La probabilidad es una **puntuación relativa** obtenida con datos balanceados 50/50; no equivale al riesgo real de "
                       "la persona en la población general (ver «Evaluación técnica» → Limitaciones).")

# ───────────────────────── Evaluación técnica ─────────────────────────
with tab_eval:
    st.subheader("Métricas finales del modelo (conjunto de prueba)")
    st.caption(f"Modelo: **{meta['modelo']}** · n entrenamiento = {meta['n_train']:,} · n prueba = {meta['n_test']:,} · umbral = {UMBRAL:.2f}".replace(",", " "))
    tabla = pd.DataFrame({"Umbral 0.50": m05, f"Umbral {UMBRAL:.2f} (elegido)": mt}).loc[
        ["Accuracy", "Precision", "Recall", "Especificidad", "F1", "ROC_AUC"]].round(3)
    st.dataframe(tabla, width="stretch")
    st.markdown(
        f"- **Recall (sensibilidad) = {mt['Recall']:.3f}**: proporción de personas con diabetes/prediabetes que el modelo detecta.\n"
        f"- **Especificidad = {mt['Especificidad']:.3f}**: proporción de personas sin la condición que el modelo clasifica correctamente.\n"
        f"- **Precisión = {mt['Precision']:.3f}**: de cada 100 alertas, ~{mt['Precision']*100:.0f} son casos reales (en un conjunto balanceado).\n"
        f"- **ROC-AUC = {m05['ROC_AUC']:.3f}**: capacidad de ordenar a los pacientes por riesgo, independiente del umbral.")

    st.markdown("#### Matriz de confusión")
    opcion = st.radio("Umbral", [f"{UMBRAL:.2f} (elegido)", "0.50"], horizontal=True, key="umbral_cm")
    mm = mt if "elegido" in opcion else m05
    cm = np.array([[mm["TN"], mm["FP"]], [mm["FN"], mm["TP"]]])
    fcm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Predice sano", "Predice diabetes"], y=["Real: sano", "Real: diabetes"])
    fcm.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fcm, width="stretch")
    st.markdown(
        f"**Cómo leerla:** {mm['TN']:,} personas sanas fueron clasificadas como sanas (verdaderos negativos); {mm['FP']:,} personas sanas recibieron una alerta "
        f"(**falsos positivos**); {mm['FN']:,} personas con diabetes/prediabetes **no fueron detectadas** (**falsos negativos**, el error más costoso en tamizaje); "
        f"y {mm['TP']:,} fueron detectadas (verdaderos positivos). Mover el umbral cambia falsos negativos por falsos positivos.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Curva ROC")
        fr = go.Figure()
        fr.add_scatter(x=meta["roc"]["fpr"], y=meta["roc"]["tpr"], name=f"AUC = {meta['roc']['auc']:.3f}")
        fr.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash"))
        fr.update_layout(height=340, xaxis_title="1 − Especificidad", yaxis_title="Sensibilidad")
        st.plotly_chart(fr, width="stretch")
    with col2:
        st.markdown("#### Variables más influyentes")
        imp = pd.Series(meta["importancia_permutacion"]).sort_values()
        fi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#4C78A8"))
        fi.update_layout(height=340, xaxis_title="Caída del ROC-AUC al permutar la variable")
        st.plotly_chart(fi, width="stretch")

    st.markdown("#### Comparación antes vs. después de la optimización (validación cruzada, 5 folds)")
    ant = pd.DataFrame(meta["cv_antes"]).set_index("Modelo")[["CV_recall", "CV_roc_auc"]].add_suffix(" (antes)")
    des = pd.DataFrame(meta["cv_despues"]).set_index("Modelo")[["CV_recall", "CV_roc_auc"]].add_suffix(" (después)")
    st.dataframe(ant.join(des, how="outer").round(4), width="stretch")

    st.markdown("#### Limitaciones del modelo")
    st.markdown(
        f"1. **Datos balanceados artificialmente (50/50).** En la población general la prevalencia es mucho menor (≈ {meta['prevalencia_real_supuesta']:.0%} en el archivo sin balancear de la misma fuente); "
        f"con esa prevalencia, el valor predictivo positivo sería de ~{meta['vpp_poblacion_real']:.0%}. La probabilidad mostrada **no es el riesgo real**.\n"
        "2. **Variable objetivo mezclada:** agrupa diabetes y prediabetes; el modelo no distingue entre ambas ni entre tipos de diabetes.\n"
        "3. **Datos de encuesta autorreportados (EE. UU., 2015):** pueden contener sesgo de memoria y no se validaron en población colombiana; "
        "la escala de ingresos está en dólares estadounidenses.\n"
        "4. **Sin variables de laboratorio** (glucosa, HbA1c): solo una prueba médica confirma el diagnóstico.\n"
        f"5. **Errores esperables:** con el umbral elegido el modelo deja sin detectar ~{1 - mt['Recall']:.0%} de los casos reales y genera alertas innecesarias en ~{1 - mt['Especificidad']:.0%} de las personas sanas.\n"
        "6. **Asociación, no causalidad:** las variables influyentes indican patrones estadísticos, no causas.\n"
        "7. **Prototipo académico:** no sustituye la valoración de un profesional de la salud.")
