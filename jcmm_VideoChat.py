import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable

# Importa el helper unificado (creado en el canvas)
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

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.sidebar.subheader("User Autentication")
    st.sidebar.button("Login:")

    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
    if not OPENAI_API_KEY:
        st.sidebar.warning("Please enter your OpenAI API key")

    video_url = st.sidebar.text_input("Youtube video URL")

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
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "transcription_y" not in st.session_state:
    st.session_state.transcription_y = ""
if "show_transcription" not in st.session_state:
    st.session_state.show_transcription = False

# ---------------------- ACTIONS ----------------------

def load_video(url: str | None):
    vid = extract_video_id(url or "")
    if not vid:
        return
    st.session_state.video_url = url or ""
    try:
        # Pref-langs ajustables
        transcription_y = get_transcript_text(vid, pref_langs=["es", "es-ES", "es-419", "en", "en-US"])
        if transcription_y:
            st.session_state.transcription_y = transcription_y
            st.success("Transcripción cargada correctamente.")
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
    st.session_state.transcription_y = ""
    st.session_state["summary"] = ""
    st.session_state.show_transcription = False

# ---------------------- SIDEBAR BUTTONS ----------------------
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load video"):
            load_video(video_url)
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
