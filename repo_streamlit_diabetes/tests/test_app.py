"""
Pruebas de calidad de la app (Actividad 1 · Tercer corte). Ejecutar desde la raíz del repositorio:
    python tests/test_app.py        (o:  pytest -q tests/test_app.py)
Prueba 1: caso normal · Prueba 2: entrada inválida · Prueba 3: registros conocidos del conjunto de prueba (coinciden con el notebook).
"""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent
TALLA_REF = 170.0
BINARIAS = ["highbp", "highchol", "cholcheck", "smoker", "stroke", "heartdiseaseorattack", "physactivity", "fruits", "veggies",
            "hvyalcoholconsump", "anyhealthcare", "nodocbccost", "diffwalk"]


def nueva_app():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def enviar(at):
    next(b for b in at.button if b.key == "enviar").click()
    at.run()
    assert not at.exception, at.exception


def test_metadatos():
    meta = json.loads((RAIZ / "artefactos" / "metadata_modelo.json").read_text(encoding="utf-8"))
    assert isinstance(meta["importancia_permutacion"], dict) and 0 < meta["umbral"] < 1
    assert len(meta["features"]) == 21
    print("[OK] Metadatos del modelo íntegros")


def test_caso_normal():
    at = nueva_app()
    enviar(at)
    r = at.session_state["ultima_prediccion"]
    assert r is not None and 0.0 <= r["probabilidad"] <= 1.0 and r["clase"] in (0, 1)
    print(f"[OK] Caso normal → probabilidad {r['probabilidad']:.4f}, clase {r['clase']}")


def test_entrada_invalida():
    at = nueva_app()
    at.number_input(key="peso").set_value(25.0)     # cada campo está en rango, pero el IMC resultante (≈ 4.7) es imposible
    at.number_input(key="talla").set_value(230.0)
    enviar(at)
    assert at.session_state["ultima_prediccion"] is None, "La app no debe predecir con entradas inválidas"
    assert len(at.error) >= 1 and "fuera del rango" in at.error[0].value
    print("[OK] Entrada inválida → bloqueada con mensaje de error")


def test_registros_conocidos():
    registros = json.loads((RAIZ / "artefactos" / "registros_conocidos.json").read_text(encoding="utf-8"))
    for reg in registros:
        e = reg["entradas"]
        at = nueva_app()
        at.number_input(key="talla").set_value(TALLA_REF)
        at.number_input(key="peso").set_value(e["bmi"] * (TALLA_REF / 100) ** 2)
        at.selectbox(key="sex").set_value("Masculino" if e["sex"] == 1 else "Femenino")
        at.selectbox(key="age").set_value(int(e["age"]))
        at.selectbox(key="genhlth").set_value(int(e["genhlth"]))
        at.selectbox(key="education").set_value(int(e["education"]))
        at.selectbox(key="income").set_value(int(e["income"]))
        at.number_input(key="menthlth").set_value(int(e["menthlth"]))
        at.number_input(key="physhlth").set_value(int(e["physhlth"]))
        for c in BINARIAS:
            at.selectbox(key=c).set_value("Sí" if e[c] == 1 else "No")
        enviar(at)
        r = at.session_state["ultima_prediccion"]
        assert abs(r["probabilidad"] - reg["probabilidad_notebook"]) < 1e-9, (reg["caso"], r, reg["probabilidad_notebook"])
        assert r["clase"] == reg["prediccion_notebook"]
        print(f"[OK] {reg['caso']:<20} app={r['probabilidad']:.6f} notebook={reg['probabilidad_notebook']:.6f} clase={r['clase']}")


if __name__ == "__main__":
    test_metadatos(); test_caso_normal(); test_entrada_invalida(); test_registros_conocidos()
    print("\nTodas las pruebas pasaron.")
