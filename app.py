import os
import hmac
from pathlib import Path
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


# 2. Cargar modelo en caché (para no recargarlo en cada búsqueda)
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(NOMBRE_MODELO)


# 3. Conexión a Neon PostgreSQL
def leer_url_base_de_datos():
    """Primero busca en los secretos de Streamlit (publicada en internet);
    si no existen, busca en el archivo .env (en tu computador)."""
    try:
        url = st.secrets["DATABASE_URL"]
        if url:
            return url
    except Exception:
        pass
    load_dotenv(dotenv_path=ARCHIVO_ENV)
    return os.getenv("DATABASE_URL")


def obtener_conexion():
    db_url = leer_url_base_de_datos()
    if not db_url:
        st.error("No se encontró DATABASE_URL (ni en los secretos de Streamlit ni en el archivo .env).")
        st.stop()
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


# 4. Función de búsqueda vectorial
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


# 4b. Puerta de acceso con clave (solo se activa si existe APP_PASSWORD en los secretos)
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

# 5. Barra Lateral (Sidebar)
with st.sidebar:
    st.title("📚 Sobre el Proyecto")
    st.markdown("""
    **Reportaje de Investigación Periodística**  
    *Irregularidad Municipal: El destino de las cotizaciones previsionales descontadas a los profesores de Isla de Maipo.*

    **Autora:** Constanza Melo Gaete  
    **Curso:** Periodismo de Investigación II  
    **Universidad de Chile**

    ---
    **Base de Conocimiento:**
    * 📑 **5 documentos PDF** originales (63 páginas).
    * 🧩 **213 trozos (chunks)** de ~800 caracteres.
    * 🗄️ **Base de datos:** Neon PostgreSQL + extensión `pgvector`.
    * 🧠 **Modelo de Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, local y gratuito).
    """)
    st.info("🔒 La conexión a la base de datos se guarda como secreto y no se muestra en pantalla.")

# 6. Encabezado principal
st.title("🔍 Buscador Semántico RAG")
st.caption("Búsqueda inteligente por significado en los documentos de investigación sobre la deuda previsional docente en Isla de Maipo.")

# 7. Preguntas de ejemplo (usan session_state para que la pregunta no se pierda al recargar)
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
     "¿Qué respondió el alcalde a Contraloría?"),
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

# 8. Formulario de Búsqueda
pregunta = st.text_input(
    label="Escribe tu pregunta sobre la investigación:",
    key="pregunta",
    placeholder="Ej: ¿Qué consecuencias laborales denuncian los docentes?",
    help="El sistema buscará los párrafos con el significado más cercano en los 5 PDFs."
)

top_k = st.slider("Cantidad de trozos más parecidos a mostrar:", min_value=1, max_value=10, value=5)

# 9. Ejecución y presentación de resultados
if pregunta.strip():
    with st.spinner("Buscando los trozos más relevantes en Neon..."):
        try:
            resultados = buscar(pregunta.strip(), top_k=top_k)

            st.markdown(f"### 📋 Se encontraron los {len(resultados)} trozos más relevantes:")

            for i, (doc, pag, idx_pag, texto, similitud) in enumerate(resultados, start=1):
                similitud_pct = max(0.0, float(similitud)) * 100

                # Etiqueta de similitud visual
                if similitud_pct >= 65:
                    badge = f"🟢 Muy alta coincidencia ({similitud_pct:.1f}%)"
                elif similitud_pct >= 50:
                    badge = f"🟡 Alta coincidencia ({similitud_pct:.1f}%)"
                else:
                    badge = f"⚪ Coincidencia moderada ({similitud_pct:.1f}%)"

                with st.expander(f"Resultado #{i} — {badge} | 📄 {doc} (Pág. {pag})", expanded=(i <= 2)):
                    st.markdown(f"**Documento:** `{doc}` &nbsp;|&nbsp; **Página:** `{pag}` &nbsp;|&nbsp; **Similitud:** `{similitud_pct:.2f}%`")
                    st.info(texto)

        except Exception as e:
            st.error(f"Error al realizar la consulta: {e}")
