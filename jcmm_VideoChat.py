import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable

# Helper de YouTube (sigue disponible para el modo URL manual)
from yt_transcript_compat import (
    get_transcript_text,
    NoTranscriptAvailable,
    extract_video_id,
)

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="VideoChat bot", page_icon="🎬")
st.subheader("jcmm VideoChat")
st.title("YouTube Video Content Chatbot")
st.markdown("With this app you can audit a Youtube video:")
st.markdown("1. a summary of the video,")
st.markdown("2. the topics that are discussed in the video,")

BASE_DIR = Path(__file__).parent
TRANS_DIR = BASE_DIR / "Transcriptions"

# --- Catálogo L35PILLS (edita las URLs si lo necesitas) ---
# Si la URL está vacía, la app intentará leerla de la primera línea del .txt (cualquier http).
L35PILLS_VIDEOS: List[Dict[str, str]] = [
    {
        "title": "8º L35 Pills Cómo usar la nueva plantilla corporativa de L35",
        "url": "https://youtu.be/y9tRN5acyBU?si=PbplUr5HTRjTXC4Z",
    },
    {
        "title": "7º L35 Pills IA aplicada a imágenes y videos",
        "url": "https://youtu.be/jCKR4N7Okds?si=Bkm97Hps-2MTg5cg",
    },
    {
        "title": "6º L35 Pills Desarrollo de detalles 2D en Revit",
        "url": "https://youtu.be/XRJEpRo84d0?si=uNrXbVciyQNU1viA",
    },
    {
        "title": "5º L35 Pills Representación Gráfica de planos y vistas BIM en fase Schematics",
        "url": "https://youtu.be/SimcsTSWrag?si=rbAIQta_NOJjQOzM",
    },
    {
        "title": "4º L35 Pills Base de datos de modelos BIM y su explotación",
        "url": "https://youtu.be/ghCI19r6BiU?si=ckPs9R26_P0wZQ0m",
    },
    {
        "title": "3º L35 PILLS Planos de venta y comercialización",
        "url": "https://youtu.be/CkcuOtJDobU?si=liHPHm50Y_r6-08o",
    },
    {
        "title": "2º L35 PILLS Concursos y primeras fases de proyectos con Revit",
        "url": "https://youtu.be/H4SGUxgcG7g?si=akhbrIMSkDXlvHMl",
    },
    {
        "title": "1º L35 PILLS Introducción a la Metodología BIM",
        "url": "https://youtu.be/bUxAw8AmZXo?si=kd7PAU8G7m-uIy0K",
    },
]

# ---------------------- UTILIDADES ----------------------
def youtube_thumb(url: str) -> Optional[str]:
    vid = extract_video_id(url or "")
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def normalize(s: str) -> str:
    """Normaliza para comparar nombres de archivo (case/acentos/espacios)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w]+", "", s)  # quita separadores y signos
    return s

def read_transcription_by_title(title: str) -> tuple[Optional[str], Optional[Path], Optional[str]]:
    """
    Devuelve (texto, ruta_usada, url_encontrada).
    Busca primero el archivo exacto <title>.txt y luego por coincidencia normalizada.
    Intenta extraer una URL de la cabecera si existe.
    """
    candidates: List[Path] = []
    exact = TRANS_DIR / f"{title}.txt"
    if exact.exists():
        candidates.append(exact)
    else:
        if TRANS_DIR.exists():
            norm_title = normalize(title)
            for p in TRANS_DIR.glob("*.txt"):
                if normalize(p.stem) == norm_title:
                    candidates.append(p)
                    break

    if not candidates:
        return None, None, None

    path = candidates[0]
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = path.read_text(errors="ignore")

    # Intenta extraer URL en la cabecera
    m = re.search(r"https?://\S+", text.splitlines()[0] if text else "")
    url_in_text = m.group(0) if m else None
    return text, path, url_in_text

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.sidebar.subheader("User Autentication")
    st.sidebar.button("Login:")

    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
    if not OPENAI_API_KEY:
        st.sidebar.warning("Please enter your OpenAI API key")

    # Entrada manual por URL (como antes)
    video_url_input = st.sidebar.text_input("Youtube video URL")

    st.markdown("---")
    st.markdown("### L35PILLS")

    # Grid 2 columnas con miniaturas + título + botón "Usar"
    grid_cols = st.columns(2, gap="small")
    for i, item in enumerate(L35PILLS_VIDEOS):
        col = grid_cols[i % 2]
        with col:
            thumb = youtube_thumb(item.get("url", ""))
            if thumb:
                st.image(thumb, use_column_width=True)
            else:
                st.empty()
            st.caption(item["title"])
            if st.button("Usar este vídeo", key=f"use_{i}"):
                # URL: la del listado o la encontrada en el .txt
                text, path_used, url_from_txt = read_transcription_by_title(item["title"])
                chosen_url = item.get("url") or url_from_txt or ""
                if not chosen_url:
                    st.warning(
                        "No pude determinar la URL. Añádela en el diccionario L35PILLS_VIDEOS "
                        "o ponla en la primera línea del .txt."
                    )
                st.session_state.video_title = item["title"]
                st.session_state.video_url = chosen_url
                if text:
                    st.session_state.transcription_y = text
                    st.session_state.show_transcription = False
                    st.success(
                        f"Transcripción local cargada"
                        + (f" desde '{path_used.name}'." if path_used else ".")
                    )
                else:
                    st.warning(
                        "No se encontró el archivo de transcripción local en 'Transcriptions'."
                    )

# ---------------------- LLM ----------------------
def get_response(user_query: str, chat_history: list[dict]) -> str:
    template = """
    You are a helpful assistant. Answer the following questions considering only the history of the conversation
    and the Chat history.
    Do not search the Internet for information unless the user specifically requests it.

    Chat history:
    {chat_history}

    User query:
    {user_query}
    """
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)

    history_lines = [f"{m['role']}: {m['content']}" for m in chat_history if isinstance(m, dict)]
    response = chain.run({
        "chat_history": "\n".join(history_lines),
        "user_query": user_query,
    })
    return response

def get_summary(transcription_text: str, video_url_value: str) -> str:
    template = """
    You are a helpful assistant.
    Create a detailed summary of the provided text.
    Provide an extensive and detailed summary, including all important points, topics discussed, people speaking, main ideas, and any relevant facts.
    Be sure to include enough detail so that someone who has not seen the video can fully understand the content.
    Use the same language as the transcript_text to generate the summary.
    For each important point, provide a link to access that specific part of the video.
    You can do this by taking the URL of the video from the variable 'video_url' and adding the parameter t=XXs with XX being the seconds from the start of the video.
    To calculate the seconds from the beginning, you can calculate that the number of words per minute of video is approximately 170.

    Video URL: {video_url}

    Transcription text:
    {transcription_t}
    """
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run({
        "transcription_t": transcription_text,
        "video_url": video_url_value,
    })

# ---------------------- SESSION STATE ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "transcription_y" not in st.session_state:
    st.session_state.transcription_y = ""
if "show_transcription" not in st.session_state:
    st.session_state.show_transcription = False

# ---------------------- ACTIONS ----------------------
def load_video_from_url(url: str | None):
    """Modo clásico: URL manual -> usa API para transcribir."""
    vid = extract_video_id(url or "")
    if not vid:
        return
    st.session_state.video_url = url or ""
    st.session_state.video_title = ""
    try:
        transcription_y = get_transcript_text(vid, pref_langs=["es", "es-ES", "es-419", "en", "en-US"])
        if transcription_y:
            st.session_state.transcription_y = transcription_y
            st.success("Transcripción cargada correctamente (API YouTube).")
        else:
            st.warning("No se obtuvo texto de transcripción.")
    except NoTranscriptAvailable as e:
        st.error(f"No existe una transcripción disponible para este vídeo en los idiomas probados. Detalle: {e}")
    except TranscriptsDisabled:
        st.error("El autor del video ha deshabilitado las transcripciones.")
    except VideoUnavailable:
        st.error("El video no está disponible.")
    except Exception as e:
        st.error(f"No se pudo cargar la transcripción: {e}")

def reset_conversation():
    st.session_state.chat_history = []
    st.session_state.video_url = ""
    st.session_state.video_title = ""
    st.session_state.transcription_y = ""
    st.session_state["summary"] = ""
    st.session_state.show_transcription = False

# ---------------------- SIDEBAR BUTTONS ----------------------
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load video (URL)"):
            load_video_from_url(video_url_input)
        if st.button("Summary"):
            if st.session_state.transcription_y:
                summary = get_summary(st.session_state.transcription_y, st.session_state.video_url)
                st.session_state["summary"] = summary
                st.session_state.chat_history.append({"role": "assistant", "content": summary})
            else:
                st.sidebar.warning("No se ha cargado ninguna transcripción aún.")
    with col2:
        if st.button("Transcription"):
            if st.session_state.transcription_y:
                st.session_state.show_transcription = True
            else:
                st.warning("No se ha cargado ninguna transcripción aún.")
        if st.button("New Conversation"):
            reset_conversation()

# ---------------------- MAIN UI ----------------------
if st.session_state.video_url:
    st.video(st.session_state.video_url)

if st.session_state.get("show_transcription"):
    st.subheader("Transcripción del Video:")
    st.write(st.session_state.transcription_y)

for message in st.session_state.chat_history:
    role = message.get("role", "")
    content = message.get("content", "")
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.write(content)

# ---------------------- CHAT INPUT ----------------------
user_query = st.chat_input("Escribe tu mensaje aquí...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
