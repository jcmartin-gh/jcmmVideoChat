import streamlit as st
# from langchain.schema import AIMessage, HumanMessage
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI  # Actualizado para la versión recomendada de LangChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, TranscriptsDisabled, NoTranscriptFound
from tenacity import retry, stop_after_attempt, wait_fixed
# from youtube_transcript_api._errors import NoTranscriptAvailable
import os
import re
import time # Añadido 2025-01-13

# Load environment variables
# load_dotenv()

# Configuración de la API de OpenAI
# OPENAI_API_KEY = st.secrets["api"]["key"]

# Configuración de la aplicación
st.set_page_config(page_title="VideoChat bot", page_icon="")
st.subheader("jcmm VideoChat")
st.title("YouTube Video Content Chatbot")
st.markdown("With this app you can audit a Youtube video:")
st.markdown("1. a summary of the video,")
st.markdown("2. the topics that are discussed in the video,")

# Barra lateral para autenticación
with st.sidebar:
    # Placeholder para autenticación (OAuth sugerido para mejorar seguridad)
    st.sidebar.subheader('User Autentication')
    st.sidebar.button('Login:')

    # Input para la API Key de OpenAI
    OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
    if not OPENAI_API_KEY:
        st.sidebar.warning("Please enter your OpenAI API key")

    # Input para la URL del video
    video_url = st.sidebar.text_input('Youtube video URL')

# Función para extraer el ID del video de la URL
def extract_video_id(url):
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)',  # URL estándar
        r'(?:https?://)?youtu\.be/([^?]+)',                        # URL corta
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)'      # URL de embed
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1).split('?')[0]  # Elimina cualquier parámetro adicional
    st.error("Invalid video URL")
    return None

# Agragar Log de Depuración
import requests
from youtube_transcript_api.formatters import JSONFormatter

from tenacity import retry, stop_after_attempt, wait_fixed


# Decorador para reintentar la función en caso de fallo
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
def fetch_transcript_with_retry(video_id, languages=['es', 'en']):
    # retrieve the available transcripts
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list('video_id')
    st.success(transcript_list)
    

# Función para obtener una respuesta del chatbot
def get_response(user_query, chat_history):
    template = """
    You are a helpful assistant. Answer the following questions considering only the history of the conversation
    and the Chat history.
    Do not search the Internet for information unless the user specifically requests it.

    Chat history:
    {chat_history}

    User query:
    {user_query}
    """
    # Inicialización del modelo de Langchain
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

# Función para obtener un resumen detallado del video
def get_summary(transcription_text):
    template = """
    You are a helpful assistant.
    Create a detailed summary of the provided text.
    Provide an extensive and detailed summary, including all important points, topics discussed, people speaking, main ideas, and any relevant facts.
    Be sure to include enough detail so that someone who has not seen the video can fully understand the content.
    Use the same language as the transcript_text to generate the summary.
    For each important point, provide a link to access that specific part of the video.
    You can do this by taking the URL of the video from the variable 'video_url' and adding the parameter t=XXs with XX being the seconds from the start of the video.
    To calculate the seconds from the beginning, you can calculate that the number of words per minute of video is approximately 170.
    This is an Example:
    (((
    The video features a conversation between Steven from Show It Better and Olly Thomas, a design technology specialist at Bjarke Ingels Group (BIG) in London.
    The discussion revolves around the evolving role of AI in architecture, the future of architectural visualization, and how architects can adapt to these changes.

    Key Points Discussed:

    1. The Role of AI in Architecture:

        Olly discusses the current state of AI in architecture, emphasizing that while AI has not taken over the industry, it is becoming an integral part of the design process.
        He notes that AI tools are currently seen as "tools" but may evolve into "collaborators" in the future. [Link to this point]

    2. Olly's Transition to AI:

        Olly shares his journey from being a BIM specialist to a design technology manager,
        highlighting the lack of formal training in AI and the importance of self-learning and experimentation.
        He mentions that his role now encompasses BIM, computation, AI, AR, and VR. [Link to this point]

    Conclusion:
        The conversation between Steven and Olly provides valuable insights into the intersection of AI and architecture,
        highlighting both the challenges and opportunities that lie ahead."
    )))

    Transcription text:
    {transcription_t}
    """
    
    # Inicialización del modelo de Langchain
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
        "transcription_t": transcription_text
    })
    return response

# Inicialización del estado de la sesión
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'summary' not in st.session_state:
    st.session_state['summary'] = ""
if 'transcription_y' not in st.session_state:
    st.session_state.transcription_y = ""

# Función para cargar y mostrar el video
def load_video(video_url):
    video_id = extract_video_id(video_url)
    if video_id:
        st.session_state.video_url = video_url
        try:
            transcription_y = fetch_transcript_with_retry(video_id)
            if transcription_y:
                st.session_state.transcription_y = transcription_y
                st.success("Transcripción cargada correctamente.")
        except Exception as e:
            st.error(f"No se pudo cargar la transcripción: {str(e)}")
            st.success(transcript_list)
# Función para reiniciar la conversación
def reset_conversation():
    st.session_state.chat_history = []
    st.session_state.video_url = ""
    st.session_state.transcription_y = ""
    st.session_state['summary'] = ""
    st.session_state.show_transcription = False

# Botones de la barra lateral
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Load video'):
            load_video(video_url)
        if st.button("Summary"):
            if st.session_state.transcription_y:
                summary = get_summary(st.session_state.transcription_y)
                st.session_state['summary'] = summary
                #st.session_state['summary'] = get_summary(st.session_state.transcription_y)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': summary,
                })           
            else:
                st.sidebar.warning("No se ha cargado ninguna transcripción aún.")
    with col2:
        if st.button("Transcription"):
            if st.session_state.transcription_y:
                st.session_state.show_transcription = True #Añadido 2025-01-13
                #st.write("**Transcripción del Video:**")
                #st.write(st.session_state.transcription_y)
            else:
                st.warning("No se ha cargado ninguna transcripción aún.")
        if st.button('New Conversation'):
            reset_conversation()

# Mostrar el video si ya está cargado
if st.session_state.video_url:
    st.video(st.session_state.video_url)

# Mostrar la transcripción en el cuerpo principal. Añadido 2025-01-13
if st.session_state.get('show_transcription'):
    st.subheader("Transcripción del Video:")
    st.write(st.session_state.transcription_y)

# Mostrar el historial de la conversación
for message in st.session_state.chat_history:
    role = message.get('role', '')
    content = message.get('content', '')
    if role == 'assistant':
        with st.chat_message("assistant"):
            st.write(content)
    elif role == 'user':
        with st.chat_message("user"):
            st.write(content)

# Entrada del usuario
user_query = st.chat_input("Escribe tu mensaje aquí...")
if user_query:
    st.session_state.chat_history.append({'role': 'user', 'content': user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    with st.chat_message("assistant"):
        st.write(response)
