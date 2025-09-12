import streamlit as st
# from langchain.schema import AIMessage, HumanMessage
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI  # Mantengo este import para compatibilidad de tu entorno
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    VideoUnavailable,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import re
import time  # Añadido 2025-01-13

# --- CONFIG ---
st.set_page_config(page_title="VideoChat bot", page_icon="")
st.subheader("jcmm VideoChat")
st.title("YouTube Video Content Chatbot")
st.markdown("With this app you can audit a Youtube video:")
st.markdown("1. a summary of the video,")
st.markdown("2. the topics that are discussed in the video,")

# --- SIDEBAR ---
with st.sidebar:
    st.sidebar.subheader("User Autentication")
    st.sidebar.button("Login:")

    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
    if not OPENAI_API_KEY:
        st.sidebar.warning("Please enter your OpenAI API key")

    video_url = st.sidebar.text_input("Youtube video URL")


# --- HELPERS ---
def extract_video_id(url: str | None) -> str | None:
    if not url:
        return None
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",  # URL estándar
        r"(?:https?://)?youtu\.be/([^?]+)",  # URL corta
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)",  # URL de embed
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1).split("?")[0]
    st.error("Invalid video URL")
    return None


def _snippets_to_text(snippets) -> str:
    """Convierte una lista de *dicts* o *FetchedTranscriptSnippet* en texto plano.

    En versiones recientes de youtube-transcript-api, Transcript.fetch() puede
    devolver objetos FetchedTranscriptSnippet (no subscriptables). Este helper
    se asegura de soportar ambos formatos.
    """
    parts: list[str] = []
    for s in snippets:
        # Formato clásico (lista de dicts)
        if isinstance(s, dict):
            parts.append(s.get("text", ""))
        else:
            # Formato nuevo (objetos con atributos)
            text = getattr(s, "text", "")
            parts.append(text)
    return " ".join(parts).strip()


# Decorador para reintentar la función en caso de fallo
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
def fetch_transcript_with_retry(video_id: str, languages: list[str] | None = None) -> str:
    """Obtiene la transcripción con fallback compatible entre versiones.

    1) Intenta el camino clásico: get_transcript(video_id, languages)
       -> devuelve lista de dicts.
    2) Si falla, usa list_transcripts().find_transcript(languages).fetch()
       -> en versiones nuevas puede devolver objetos FetchedTranscriptSnippet,
          que convertimos a texto con _snippets_to_text.
    """
    languages = languages or ["es", "en"]

    # Camino 1: clásico y más estable
    try:
        snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return _snippets_to_text(snippets)
    except NoTranscriptFound:
        # Pasamos al camino 2 (fallback)
        pass
    except (TranscriptsDisabled, VideoUnavailable) as e:
        # Errores definitivos
        raise e
    except Exception:
        # Cualquier otro error inesperado -> probamos el camino 2
        pass

    # Camino 2: API de alto nivel con objetos Transcript
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # Intento directo con todas las lenguas preferidas
        try:
            t = transcripts.find_transcript(languages)
        except Exception:
            # Búsqueda individual por si alguna coincide
            t = None
            for lang in languages:
                try:
                    t = transcripts.find_transcript([lang])
                    if t:
                        break
                except Exception:
                    continue
        if not t:
            raise NoTranscriptFound("No transcript matching preferred languages.")

        snippets = t.fetch()  # dicts o FetchedTranscriptSnippet (según versión)
        return _snippets_to_text(snippets)
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        raise
    except Exception as e:
        # Propagamos para que tenacity haga reintentos y Streamlit muestre el error
        raise RuntimeError(f"Unexpected transcript fetch error: {e}")


# --- LLM UTILS ---
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
    model_name = "gpt-4o-mini"
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)

    history_lines = [f"{msg['role']}: {msg['content']}" for msg in chat_history if isinstance(msg, dict)]
    chat_history_text = "\n".join(history_lines)
    response = chain.run({
        "chat_history": chat_history_text,
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

    model_name = "gpt-4o-mini"
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run({
        "transcription_t": transcription_text,
        "video_url": video_url_value,
    })
    return response


# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "transcription_y" not in st.session_state:
    st.session_state.transcription_y = ""
if "show_transcription" not in st.session_state:
    st.session_state.show_transcription = False


# --- ACTIONS ---
def load_video(url: str | None):
    vid = extract_video_id(url)
    if not vid:
        return
    st.session_state.video_url = url or ""
    try:
        transcription_y = fetch_transcript_with_retry(vid)
        if transcription_y:
            st.session_state.transcription_y = transcription_y
            st.success("Transcripción cargada correctamente.")
        else:
            st.warning("No se obtuvo texto de transcripción.")
    except NoTranscriptFound:
        st.error("No existe una transcripción en los idiomas especificados.")
    except TranscriptsDisabled:
        st.error("El autor del video ha deshabilitado las transcripciones.")
    except VideoUnavailable:
        st.error("El video no está disponible.")
    except Exception as e:
        st.error(f"No se pudo cargar la transcripción: {str(e)}")


def reset_conversation():
    st.session_state.chat_history = []
    st.session_state.video_url = ""
    st.session_state.transcription_y = ""
    st.session_state["summary"] = ""
    st.session_state.show_transcription = False


# --- SIDEBAR BUTTONS ---
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load video"):
            load_video(video_url)
        if st.button("Summary"):
            if st.session_state.transcription_y:
                summary = get_summary(st.session_state.transcription_y, st.session_state.video_url)
                st.session_state["summary"] = summary
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": summary,
                })
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

# --- MAIN UI ---
if st.session_state.video_url:
    st.video(st.session_state.video_url)

if st.session_state.get("show_transcription"):
    st.subheader("Transcripción del Video:")
    st.write(st.session_state.transcription_y)

for message in st.session_state.chat_history:
    role = message.get("role", "")
    content = message.get("content", "")
    if role == "assistant":
        with st.chat_message("assistant"):
            st.write(content)
    elif role == "user":
        with st.chat_message("user"):
            st.write(content)

# --- CHAT INPUT ---
user_query = st.chat_input("Escribe tu mensaje aquí...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
