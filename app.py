import os
import time
import hmac
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# 1. Configuración de página
st.set_page_config(
    page_title="Buscador RAG | Isla de Maipo",
    page_icon="🔍",
    layout="wide"
)

CARPETA_PROYECTO = Path(__file__).parent
ARCHIVO_ENV = CARPETA_PROYECTO / ".env"
NOMBRE_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
# Modelos de Gemini que se prueban en orden (puedes fijar uno con el secreto GEMINI_MODEL)
MODELOS_GEMINI_POR_DEFECTO = ["gemini-3.8-flash", "gemini-3.6-flash"]


# 2. Cargar modelo de embeddings en caché (para no recargarlo en cada búsqueda)
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(NOMBRE_MODELO)


# 3. Leer secretos (primero Streamlit, luego .env en tu computador)
def leer_secreto(nombre: str):
    try:
        valor = st.secrets[nombre]
        if valor:
            return str(valor)
    except Exception:
        pass
    load_dotenv(dotenv_path=ARCHIVO_ENV)
    return os.getenv(nombre)


def obtener_conexion():
    db_url = leer_secreto("DATABASE_URL")
    if not db_url:
        st.error("No se encontró DATABASE_URL (ni en los secretos de Streamlit ni en el archivo .env).")
        st.stop()
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


# 4. Búsqueda vectorial en Neon (la parte "Retrieval")
def buscar(pregunta: str, top_k: int = 5):
    modelo = cargar_modelo()
    vector_pregunta = modelo.encode(pregunta, normalize_embeddings=True).tolist()

    conn = obtener_conexion()
    cur = conn.cursor()
    query = """
        SELECT
            documento,
            pagina,
            indice_en_pagina,
            texto,
            1 - (embedding <=> %s::vector) AS similitud
        FROM trozos_investigacion
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """
    cur.execute(query, (vector_pregunta, vector_pregunta, top_k))
    resultados = cur.fetchall()
    cur.close()
    conn.close()
    return resultados


# 5. Generación de la respuesta con Gemini (la parte "Generation")
def generar_respuesta(pregunta: str, resultados):
    api_key = leer_secreto("GEMINI_API_KEY")
    if not api_key:
        return None, "No hay clave de Gemini configurada (GEMINI_API_KEY)."

    fuentes = "\n\n".join(
        f"[{i}] ({doc}, pág. {pag})\n{texto}"
        for i, (doc, pag, _idx, texto, _sim) in enumerate(resultados, start=1)
    )
    prompt = (
        "Eres un asistente que ayuda a una periodista a encontrar información en sus propios "
        "documentos de investigación. Responde la pregunta usando SOLO los fragmentos numerados "
        "que aparecen abajo. Si la respuesta no está en los fragmentos, di claramente que no la "
        "encontraste. Cita las fuentes entre corchetes, por ejemplo [1] o [2]. Responde en español, "
        "de forma clara y breve (máximo 6 oraciones). No inventes datos ni cifras.\n\n"
        f"PREGUNTA: {pregunta}\n\nFRAGMENTOS:\n{fuentes}"
    )

    modelo_fijo = leer_secreto("GEMINI_MODEL")
    modelos = [modelo_fijo] if modelo_fijo else MODELOS_GEMINI_POR_DEFECTO

    errores = []
    for modelo in modelos:
        for intento in range(2):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2},
                    },
                    timeout=60,
                )
                if r.status_code == 200:
                    partes = r.json()["candidates"][0]["content"]["parts"]
                    texto = "".join(p.get("text", "") for p in partes).strip()
                    if texto:
                        return texto, None
                    errores.append(f"{modelo}: respuesta vacía")
                    break
                # 500/503 = servicio ocupado: se reintenta una vez (con 429, que es límite de cuota, no se insiste)
                if r.status_code in (500, 503) and intento == 0:
                    time.sleep(6)
                    continue
                try:
                    detalle = r.json().get("error", {}).get("message", "")[:120]
                except Exception:
                    detalle = ""
                errores.append(f"{modelo}: código {r.status_code} {detalle}".strip())
                break
            except Exception as e:
                errores.append(f"{modelo}: {type(e).__name__}")
                break
    return None, " | ".join(errores)


def leer_respuesta_guardada(pregunta: str, top_k: int):
    """Busca en Neon si esta pregunta ya fue respondida antes (para no gastar la cuota gratuita)."""
    try:
        conn = obtener_conexion()
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS respuestas_guardadas ("
            "pregunta TEXT, top_k INT, respuesta TEXT, PRIMARY KEY (pregunta, top_k))"
        )
        conn.commit()
        cur.execute(
            "SELECT respuesta FROM respuestas_guardadas WHERE pregunta = %s AND top_k = %s",
            (pregunta, top_k),
        )
        fila = cur.fetchone()
        cur.close()
        conn.close()
        return fila[0] if fila else None
    except Exception:
        return None


def guardar_respuesta(pregunta: str, top_k: int, respuesta: str):
    try:
        conn = obtener_conexion()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO respuestas_guardadas (pregunta, top_k, respuesta) VALUES (%s, %s, %s) "
            "ON CONFLICT (pregunta, top_k) DO NOTHING",
            (pregunta, top_k, respuesta),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def generar_respuesta_cacheada(pregunta: str, resultados):
    """Primero mira si la respuesta ya está guardada en Neon; si no, la genera con Gemini y la guarda.
    Si falla, lanza una excepción (y así no se guarda el fallo)."""
    top_k = len(resultados)
    guardada = leer_respuesta_guardada(pregunta, top_k)
    if guardada:
        return guardada
    respuesta, error = generar_respuesta(pregunta, resultados)
    if not respuesta:
        raise RuntimeError(error or "desconocido")
    guardar_respuesta(pregunta, top_k, respuesta)
    return respuesta


# 6. Puerta de acceso con clave (solo se activa si existe APP_PASSWORD en los secretos)
def pedir_clave():
    try:
        clave_correcta = st.secrets["APP_PASSWORD"]
    except Exception:
        clave_correcta = None
    if not clave_correcta:
        return
    if st.session_state.get("acceso_ok"):
        return
    st.title("🔒 Acceso restringido")
    st.write("Este buscador contiene material de una investigación periodística (incluye entrevistas). "
             "Ingresa la clave que envió la autora.")
    with st.form("form_clave"):
        intento = st.text_input("Clave de acceso", type="password")
        enviar = st.form_submit_button("Entrar")
    if enviar:
        if hmac.compare_digest(intento.encode("utf-8"), str(clave_correcta).encode("utf-8")):
            st.session_state["acceso_ok"] = True
            st.rerun()
        else:
            st.error("Clave incorrecta.")
    st.stop()


pedir_clave()

# 7. Barra Lateral (Sidebar)
with st.sidebar:
    st.title("📚 Sobre el Proyecto")
    st.markdown("""
    **Reportaje de Investigación Periodística**  
    *Irregularidad Municipal: El destino de las cotizaciones previsionales descontadas a los profesores de Isla de Maipo.*

    **Autora:** Constanza Melo Gaete  
    **Curso:** Periodismo de Investigación II  
    **Universidad de Chile**

    ---
    **Cómo funciona este RAG:**
    1. 🔎 **Recuperación:** la pregunta se convierte en vector y se buscan en Neon (PostgreSQL + `pgvector`) los trozos más parecidos.
    2. 🧠 **Generación:** una IA (Gemini) redacta la respuesta usando solo esos trozos y cita las fuentes.

    **Base de Conocimiento:**
    * 📑 **5 documentos PDF** originales (63 páginas).
    * 🧩 **213 trozos (chunks)** de ~800 caracteres.
    * 🧠 **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, local y gratuito).
    """)
    st.info("🔒 Las claves de acceso y la conexión a la base de datos se guardan como secretos y no se muestran en pantalla.")

# 8. Encabezado principal
st.title("🔍 Buscador Semántico RAG")
st.caption("Pregunta sobre los documentos de la investigación sobre la deuda previsional docente en Isla de Maipo: el sistema busca los trozos relevantes y redacta una respuesta con ellos.")

# 9. Preguntas de ejemplo (usan session_state para que la pregunta no se pierda al recargar)
if "pregunta" not in st.session_state:
    st.session_state["pregunta"] = ""


def elegir_pregunta(texto: str):
    st.session_state["pregunta"] = texto


EJEMPLOS = [
    ("¿A cuánto asciende la deuda y desde cuándo?",
     "¿A cuánto asciende la deuda previsional y desde cuándo?"),
    ("¿Qué dijo Erika Valdés sobre salud?",
     "¿Qué dijo Erika Valdés sobre la cobertura de salud?"),
    ("¿Qué respondió el alcalde a Contraloría?",
     "¿Qué respondió el alcalde a Contraloría en el Ordinario N°1100?"),
    ("¿Qué es la bicicleta financiera según las trabajadoras?",
     "¿Qué es la bicicleta financiera según las trabajadoras?"),
    ("¿Qué pasó con los fondos FAEP y la Resolución 6670?",
     "¿Qué pasó con los fondos FAEP y la Resolución Exenta 6670?"),
]

st.markdown("##### 💡 Preguntas de ejemplo (haz clic para probar):")
columnas = st.columns(3)
for i, (etiqueta, pregunta_completa) in enumerate(EJEMPLOS):
    columnas[i % 3].button(
        etiqueta,
        key=f"ejemplo_{i}",
        on_click=elegir_pregunta,
        args=(pregunta_completa,),
    )

# 10. Formulario de Búsqueda
pregunta = st.text_input(
    label="Escribe tu pregunta sobre la investigación:",
    key="pregunta",
    placeholder="Ej: ¿Qué consecuencias laborales denuncian los docentes?",
    help="El sistema busca los párrafos más cercanos en los 5 PDFs y redacta una respuesta con ellos."
)

top_k = st.slider("Cantidad de trozos a recuperar:", min_value=1, max_value=10, value=5)

# 11. Ejecución y presentación de resultados
if pregunta.strip():
    pregunta_limpia = pregunta.strip()
    try:
        with st.spinner("Buscando los trozos más relevantes en Neon..."):
            resultados = buscar(pregunta_limpia, top_k=top_k)

        # Respuesta generada (las preguntas repetidas se leen de memoria y no gastan cuota)
        respuesta = None
        error = None
        try:
            with st.spinner("Redactando la respuesta con los trozos encontrados..."):
                respuesta = generar_respuesta_cacheada(pregunta_limpia, resultados)
        except Exception as e:
            error = str(e)

        st.markdown("### 🧠 Respuesta generada")
        if respuesta:
            st.success(respuesta.replace("$", "\\$"))
            st.caption("Respuesta redactada por una IA usando solo los trozos de abajo. Verifica siempre en el documento original.")
        else:
            st.warning("No se pudo generar la respuesta automática (puede ser el límite de uso gratuito de la IA, "
                       "que se renueva cada día). Los trozos encontrados aparecen abajo: la búsqueda en Neon funciona igual. "
                       f"Detalle: {error or 'desconocido'}")

        st.markdown(f"### 📋 Fuentes: los {len(resultados)} trozos más relevantes")

        for i, (doc, pag, idx_pag, texto, similitud) in enumerate(resultados, start=1):
            similitud_pct = max(0.0, float(similitud)) * 100

            # Etiqueta de similitud visual
            if similitud_pct >= 65:
                badge = f"🟢 Muy alta coincidencia ({similitud_pct:.1f}%)"
            elif similitud_pct >= 50:
                badge = f"🟡 Alta coincidencia ({similitud_pct:.1f}%)"
            else:
                badge = f"⚪ Coincidencia moderada ({similitud_pct:.1f}%)"

            with st.expander(f"[{i}] {badge} | 📄 {doc} (Pág. {pag})", expanded=False):
                st.markdown(f"**Documento:** `{doc}` &nbsp;|&nbsp; **Página:** `{pag}` &nbsp;|&nbsp; **Similitud:** `{similitud_pct:.2f}%`")
                st.info(texto.replace("$", "\\$"))

    except Exception as e:
        st.error(f"Error al realizar la consulta: {e}")
