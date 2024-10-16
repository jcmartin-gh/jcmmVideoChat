

import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI  # Importación actualizada
# from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import re
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

# Load environment variables
# load_dotenv()

# App config
st.set_page_config(page_title="VideoChat bot", page_icon="")
st.subheader("jcmm VideoChat")
st.title("YouTube Video Content Chatbot")
st.markdown("With this app you can audit a Youtube video:")
st.markdown("1. a summary of the video,")
st.markdown("2. the topics that are discussed in the video,")

# Create the sidebar
with st.sidebar:
    # Placeholder for authentication (code needed)
    st.sidebar.button('Iniciar sesión')

    # Input for OpenAI API Key
    OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
    if not OPENAI_API_KEY:
        st.sidebar.warning("Please enter your OpenAI API key")

    # Input for video URL
    video_url = st.sidebar.text_input('Ingresa la URL del video')

# Function to extract the video ID from the URL
def extract_video_id(url):
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)',  # Standard URL
        r'(?:https?://)?youtu\.be/([^?]+)'                         # Short URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid video URL")

# Function to transcribe the video from its ID
import time

def get_transcript(video_id, retries=3, delay=5):
    for attempt in range(retries):
        try:
            # Intenta obtener la transcripción
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es', 'fr', 'de'])
            transcript_text = "\n".join([entry['text'] for entry in transcript])
            return transcript_text
        except CouldNotRetrieveTranscript:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error("No se pudo recuperar la transcripción para este video. Intente con otro video.")
                st.stop()
        except Exception as e:
            st.error(f"Error al obtener la transcripción: {str(e)}")
            st.stop()

# Function to get chatbot response
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

    prompt = ChatPromptTemplate.from_template(template)

    model_name = "gpt-4o-mini"
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=4000,
    )

    chain = LLMChain(llm=model, prompt=prompt)

    # Convert chat history to text
    history_lines = []
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
        else:
            role = msg.role
            content = msg.content
        history_lines.append(f"{role}: {content}")

    chat_history_text = "\n".join(history_lines)

    response = chain.run({
        "chat_history": chat_history_text,
        "user_query": user_query,
    })

    return response

# Function to get summary
def get_summary(transcription_t):
    template = """
    You are a helpful assistant.
    Create a detailed summary of the provided text.
    Provide an extensive and detailed summary, including all important points, topics discussed, people speaking, main ideas, and any relevant facts.
    Be sure to include enough detail so that someone who has not seen the video can fully understand the content.

    For each important point, provide a link to access that specific part of the video.
    You can do this by taking the URL of the video from the beginning of the transcription and adding the parameter t=XXs with XX being the seconds from the start of the video.
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

    prompt = ChatPromptTemplate.from_template(template)

    model_name = "gpt-4o-mini"
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=4000,
    )

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run({
        "transcription_t": transcription_t
    })

    return response

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'video_url' not in st.session_state:
    st.session_state.video_url = ""

if 'summary' not in st.session_state:
    st.session_state['summary'] = ""

if 'summary_displayed' not in st.session_state:
    st.session_state['summary_displayed'] = False

if 'transcription_y' not in st.session_state:
    st.session_state.transcription_y = ""

# Function to load and display the video
def load_video(video_url):
    if video_url:
        try:
            st.session_state.video_url = video_url
            video_id = extract_video_id(video_url)
            transcription_y = get_transcript(video_id)
            if transcription_y:
                st.session_state.transcription_y = transcription_y
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "Transcripción del video cargada.",
                    'display': False  # Indicador para no mostrar el texto
                })
        except ValueError as ve:
            st.error("Error al extraer el ID del video: " + str(ve))

# Function to reset conversation
def reset_conversation():
    st.session_state.chat_history = []
    st.session_state.video_url = ""
    st.session_state.transcription_y = ""
    st.session_state['summary'] = ""
    st.session_state['summary_displayed'] = False

# Sidebar buttons
with st.sidebar:
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Load video'):
            load_video(video_url)
        if st.button("Summary"):
            if 'transcription_y' in st.session_state and st.session_state.transcription_y:
                if st.session_state['summary'] == "":
                  st.write("Video Summary:")
                  st.session_state['summary'] = get_summary(st.session_state.transcription_y)
                  st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': st.session_state['summary'],
                   })
                  st.session_state['summary_displayed'] = True

            else:
                st.write("No transcription has been loaded yet.")

    with col2:
        if st.button("Show Transcript"):
            if 'transcription_y' in st.session_state and st.session_state.transcription_y:
                st.session_state.chat_history.append({'role': 'assistant', 'content': st.session_state.transcription_y})
                st.session_state['summary_displayed'] = False
            else:
                st.write("No transcription has been loaded yet.")
        if st.button('Reset Conversation'):
            reset_conversation()

# Display the video if it's already loaded
if st.session_state.video_url:
    with st.container():
        st.video(st.session_state.video_url)

# Display summary when the summary button is clicked
if 'summary' in st.session_state and st.session_state['summary'] and not st.session_state['summary_displayed']:
    st.write("Video Summary:")
    st.write(st.session_state['summary'])
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': st.session_state['summary'],
    })
    st.session_state['summary_displayed'] = True

# Conversation display
for message in st.session_state.chat_history:
    if isinstance(message, dict):
        role = message.get('role', '')
        content = message.get('content', '')
    else:
        role = message.role
        content = message.content

    if role == 'assistant' or role == 'AI':
        with st.chat_message("assistant"):
            st.write(content)
    elif role == 'user' or role == 'Human':
        with st.chat_message("user"):
            st.write(content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append({'role': 'user', 'content': user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})

    with st.chat_message("assistant"):
        st.write(response)
