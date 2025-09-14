import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable

# Modo URL manual (sigue existiendo)
from yt_transcript_compat import (
    get_transcript_text,
    NoTranscriptAvailable,
    extract_video_id,
)

# ---------------------- CONFIG BÁSICA ----------------------
st.set_page_config(page_title="VideoChat bot", page_icon="🎬")
st.subheader("jcmm VideoChat")
st.title("YouTube Video Content Chatbot")
st.markdown("With this app you can audit a Youtube video:")
st.markdown("1. a summary of the video,")
st.markdown("2. the topics that are discussed in the video,")

BASE_DIR = Path(__file__).parent
TRANS_DIR = BASE_DIR / "Transcriptions"
TRANS_DIR.mkdir(parents=True, exist_ok=True)  # garantiza carpeta

# ---------------------- CATÁLOGO L35PILLS ----------------------
# Si la URL está vacía, la app intentará leerla del .txt (primera línea que contenga http).
L35PILLS_VIDEOS: List[Dict[str, str]] = [
    {"title": "8º L35 Pills Cómo usar la nueva plantilla corporativa de L35", 
     "url": "https://youtu.be/y9tRN5acyBU?si=PbplUr5HTRjTXC4Z"},
    {"title": "7º L35 Pills IA aplicada a imágenes y videos",
     "url": "https://youtu.be/jCKR4N7Okds?si=Bkm97Hps-2MTg5cg"},
    {"title": "6º L35 Pills Desarrollo de detalles 2D en Revit",
     "url": "https://youtu.be/XRJEpRo84d0?si=uNrXbVciyQNU1viA"},
    {"title": "5º L35 Pills Representación Gráfica de planos y vistas BIM en fase Schematics",
     "url": "https://youtu.be/SimcsTSWrag?si=rbAIQta_NOJjQOzM"},
    {"title": "4º L35 Pills Base de datos de modelos BIM y su explotación",
     "url": "https://youtu.be/ghCI19r6BiU?si=ckPs9R26_P0wZQ0m"},
    {"title": "3º L35 PILLS Planos de venta y comercialización",
     "url": "https://youtu.be/CkcuOtJDobU?si=liHPHm50Y_r6-08o"},
    {"title": "2º L35 PILLS Concursos y primeras fases de proyectos con Revit",
     "url": "https://youtu.be/H4SGUxgcG7g?si=akhbrIMSkDXlvHMl"},
    {"title": "1º L35 PILLS Introducción a la Metodología BIM",
     "url": "https://youtu.be/bUxAw8AmZXo?si=kd7PAU8G7m-uIy0K"},
]

# ---------------------- UTILIDADES ----------------------
def normalize(s: str) -> str:
    """Normaliza para emparejar nombres de archivo: quita acentos, espacios y signos, y pasa a minúsculas."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w]+", "", s)
    return s

def find_first_url(s: str) -> Optional[str]:
    m = re.search(r"https?://\S+", s or "")
    return m.group(0) if m else None

def youtube_thumb(url: str) -> Optional[str]:
    vid = extract_video_id(url or "")
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def read_transcription_by_title(title: str) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
    """
    Lee la transcripción local por título. Devuelve (texto, ruta, url_encontrada_en_txt).
    Busca fichero exacto '<title>.txt' y, si no existe, por coincidencia 'normalizada'.
    Extrae una URL de la primera línea si existe.
    """
    # 1) exacto
    exact = TRANS_DIR / f"{title}.txt"
    if exact.exists():
        text = exact.read_text(encoding="utf-8", errors="ignore")
        return text, exact, find_first_url(text.splitlines()[0] if text else "")
    # 2) normalizado
    target = normalize(title)
    for p in TRANS_DIR.glob("*.txt"):
        if normalize(p.stem) == target:
            text = p.read_text(encoding="utf-8", errors="ignore")
            return text, p, find_first_url(text.splitlines()[0] if text else "")
    return None, None, None

# ---- Helpers tiempo/URL/capítulos ----
def _hms_to_seconds(hms: str) -> int:
    parts = hms.strip().split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    h, m, s = parts
    return h * 3600 + m * 60 + s

def _seconds_to_hms(total_s: int) -> str:
    total_s = max(0, int(total_s))
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def add_ts_to_url(url: str, seconds: int) -> str:
    """Inserta/actualiza ?t=XXs preservando el resto de parámetros."""
    if not url:
        return ""
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["t"] = f"{int(seconds)}s"
    new_q = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))

_TS_BLOCK_RE = re.compile(
    r"\[\s*(?P<start>(?:\d+:)?\d{1,2}:\d{2})\s*-\s*(?P<end>(?:\d+:)?\d{1,2}:\d{2})\s*\]\s*",
    flags=re.UNICODE
)

def parse_blocks_from_text(txt: str) -> List[Dict]:
    """
    Extrae bloques con timestamps tipo:
    [0:00:00 - 0:04:00] ...contenido...
    Devuelve: [{"start": seg, "end": seg, "text": "..."}]
    """
    blocks = []
    if not txt:
        return blocks
    matches = list(_TS_BLOCK_RE.finditer(txt))
    if not matches:
        return blocks
    for i, m in enumerate(matches):
        start_s = _hms_to_seconds(m.group("start"))
        end_s = _hms_to_seconds(m.group("end"))
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        content = txt[content_start:content_end].strip()
        if content:
            blocks.append({"start": start_s, "end": end_s, "text": content})
    return blocks

def build_blocks_from_segments(segments: List[Dict], window_s: int = 240) -> List[Dict]:
    """
    Agrupa los segmentos API en ventanas temporales (4 min por defecto)
    para generar capítulos aproximados cuando no hay marcas explícitas.
    """
    if not segments:
        return []
    segs = sorted([s for s in segments if s.get("start") is not None], key=lambda x: x["start"])
    t0 = int(segs[0]["start"]) if segs else 0
    last = segs[-1]
    t_last = int(last["start"] + (last.get("duration") or 0))
    blocks = []
    cursor = t0
    while cursor <= t_last + 1:
        window_end = cursor + window_s
        chunk = [s for s in segs if cursor <= (s["start"] or 0) < window_end]
        text = " ".join((s.get("text") or "").strip() for s in chunk if s.get("text"))
        if text.strip():
            blocks.append({"start": cursor, "end": window_end, "text": text.strip()})
        cursor = window_end
    return blocks

# ---------------------- LLM HELPERS ----------------------
def get_response(user_query: str, chat_history: List[Dict]) -> str:
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
        openai_api_key=st.session_state.get("OPENAI_API_KEY", ""),
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)
    history_lines = [f"{m['role']}: {m['content']}" for m in chat_history if isinstance(m, dict)]
    response = chain.run({"chat_history": "\n".join(history_lines), "user_query": user_query})
    return response

def get_summary(transcription_text: str, video_url_value: str) -> str:
    template = """
    You are a helpful assistant.
    Create a detailed summary of the provided text.
    Provide an extensive and detailed summary, including all important points discussed, people speaking, main ideas, and any relevant facts.
    Be sure to include enough detail so that someone who has not seen the video can fully understand the content.
    Use the same language as the transcript_text to generate the summary.
    
    Video URL: {video_url}

    Transcription text:
    {transcription_t}
    """
    model = ChatOpenAI(
        openai_api_key=st.session_state.get("OPENAI_API_KEY", ""),
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run({"transcription_t": transcription_text, "video_url": video_url_value})

def generate_linked_summary(blocks: List[Dict], video_url_value: str) -> str:
    """
    Genera un resumen 'por capítulos' con enlace al instante de inicio de cada bloque.
    Si no hay bloques, devuelve aviso.
    """
    if not blocks:
        return "No hay capítulos detectados en la transcripción."
    model = ChatOpenAI(
        openai_api_key=st.session_state.get("OPENAI_API_KEY", ""),
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=400,
    )
    title_prompt_tpl = ChatPromptTemplate.from_template(
        "Escribe un título conciso (máx 12 palabras) en español que resuma este fragmento:\n\n{frag}\n\nDevuelve solo el título, sin comillas."
    )
    title_chain = LLMChain(llm=model, prompt=title_prompt_tpl)

    lines = ["## Resumen por capítulos\n"]
    for b in blocks:
        start = int(b["start"]) if b.get("start") is not None else 0
        hms = _seconds_to_hms(start)
        link = add_ts_to_url(video_url_value or "", start)
        try:
            title = title_chain.run({"frag": b["text"][:2000]}).strip()
        except Exception:
            title = (b.get("text", "")[:80] + "…").replace("\n", " ")
        if link:
            lines.append(f"- [{hms}] [{title}]({link})")
        else:
            lines.append(f"- [{hms}] {title}")
    return "\n".join(lines)

# ---------------------- SESSION STATE ----------------------
ss = st.session_state
ss.setdefault("chat_history", [])
ss.setdefault("video_url", "")
ss.setdefault("video_title", "")
ss.setdefault("summary", "")
ss.setdefault("transcription_y", "")
ss.setdefault("show_transcription", False)
ss.setdefault("OPENAI_API_KEY", "")
ss.setdefault("blocks", [])

# ---------------------- ACCIONES ----------------------
def load_video_from_url(url: str | None):
    """Modo clásico: URL manual -> usa API YouTube para transcribir."""
    vid = extract_video_id(url or "")
    if not vid:
        st.sidebar.warning("Pon una URL válida de YouTube.")
        return
    ss.video_url = url or ""
    ss.video_title = ""
    try:
        segments = []
        try:
            from yt_transcript_compat import get_segments
            segments = get_segments(vid, pref_langs=["es", "es-ES", "es-419", "en", "en-US"])
        except Exception:
            segments = []

        if segments:
            ss.transcription_y = " ".join(s.get("text", "") for s in segments if s.get("text")).strip()
            ss.blocks = build_blocks_from_segments(segments, window_s=240)
        else:
            transcription_y = get_transcript_text(vid, pref_langs=["es", "es-ES", "es-419", "en", "en-US"])
            ss.transcription_y = transcription_y or ""
            ss.blocks = parse_blocks_from_text(ss.transcription_y)

        ss.show_transcription = False if ss.transcription_y else False
        if ss.transcription_y:
            st.sidebar.success("Transcripción cargada (API YouTube)." if segments else "Transcripción cargada.")
        else:
            st.sidebar.warning("No se obtuvo texto de transcripción.")
    except NoTranscriptAvailable as e:
        st.sidebar.error(f"No existe una transcripción disponible para este vídeo. Detalle: {e}")
    except TranscriptsDisabled:
        st.sidebar.error("El autor del video ha deshabilitado las transcripciones.")
    except VideoUnavailable:
        st.sidebar.error("El video no está disponible.")
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar la transcripción: {e}")

def load_video_from_catalog(item: Dict):
    """Nuevo modo: selección en cuadrícula -> lee transcripción local, no usa API."""
    title = item["title"]
    txt, path_used, url_in_txt = read_transcription_by_title(title)
    if not txt:
        st.sidebar.error(f"No se encontró 'Transcriptions/{title}.txt' (o equivalente normalizado).")
        return
    # Fija URL (prioridad: lista -> .txt -> vacío)
    url = item.get("url") or url_in_txt or ""
    ss.video_title = title
    ss.video_url = url
    ss.transcription_y = txt
    ss.blocks = parse_blocks_from_text(ss.transcription_y) or []
    ss.show_transcription = False  # No mostrar de inmediato
    if path_used:
        st.sidebar.success(f"Transcripción local cargada: {path_used.name}")
    if not ss.video_url:
        st.sidebar.warning("No tengo URL del vídeo. Añádela en el diccionario o en la primera línea del .txt.")

def reset_conversation():
    ss.chat_history = []
    ss.video_url = ""
    ss.video_title = ""
    ss.transcription_y = ""
    ss.summary = ""
    ss.show_transcription = False
    ss.blocks = []



# --- LOGIN (bloquea el resto de la app si no hay acceso) ---
from simple_login import require_login

if not require_login(
    app_name="jcmmVideoChat",
    password_key="APP_PASSWORD",   # st.secrets["APP_PASSWORD"] o env APP_PASSWORD
    session_flag="__is_auth",      # nombre del flag de sesión
):
    st.stop()
# --- FIN LOGIN ---



# ---------------------- SIDEBAR (BOTONES ORIGINALES ARRIBA) ----------------------
with st.sidebar:
    st.sidebar.subheader("User Autentication")
    st.sidebar.button("Login:")
    ss.OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password", value=ss.get("OPENAI_API_KEY", ""))

    video_url_input = st.text_input("Youtube video URL", value=ss.get("video_url", ""))

    # === Botones en su sitio original ===
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load video (URL)"):
            load_video_from_url(video_url_input)

        # Resumen por capítulos con enlaces (capítulos)
        if st.button("Chapters"):
            if ss.transcription_y:
                if not ss.blocks:
                    ss.blocks = parse_blocks_from_text(ss.transcription_y)
                ss.summary = generate_linked_summary(ss.blocks, ss.video_url)
                ss.chat_history.append({"role": "assistant", "content": ss.summary})
            else:
                st.warning("No se ha cargado ninguna transcripción aún.")

        # >>> NUEVO: Resumen narrativo usando get_summary(...)
        if st.button("Summary"):
            if ss.transcription_y:
                ss.summary = get_summary(ss.transcription_y, ss.video_url)
                ss.chat_history.append({"role": "assistant", "content": ss.summary})
            else:
                st.warning("No se ha cargado ninguna transcripción aún.")
        # <<< FIN NUEVO

    with col2:
        if st.button("Transcription"):
            if ss.transcription_y:
                ss.show_transcription = True
            else:
                st.warning("No se ha cargado ninguna transcripción aún.")
        if st.button("New Conversation"):
            reset_conversation()

    st.markdown("---")

    # === Nueva selección L35PILLS (debajo, en expander, no desplaza lo anterior) ===
    with st.expander("Elegir vídeo de L35PILLS (sin usar API)"):
        grid_cols = st.columns(2, gap="small")
        for i, item in enumerate(L35PILLS_VIDEOS):
            col = grid_cols[i % 2]
            with col:
                thumb = youtube_thumb(item.get("url", ""))
                if thumb:
                    st.image(thumb, width='content')
                st.caption(item["title"])
                if st.button("Usar este video", key=f"use_{i}"):
                    load_video_from_catalog(item)

# ---------------------- MAIN ----------------------
if ss.video_title:
    st.markdown(f"### {ss.video_title}")

if ss.video_url:
    st.video(ss.video_url)

if ss.show_transcription and ss.transcription_y:
    st.subheader("Transcripción del Video")
    st.write(ss.transcription_y)

# Historial de chat
for message in ss.chat_history:
    role = message.get("role", "")
    content = message.get("content", "")
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.write(content)

# Entrada de chat
user_query = st.chat_input("Escribe tu mensaje aquí...")
if user_query:
    ss.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    response = get_response(user_query, ss.chat_history)
    ss.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
