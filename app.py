import streamlit as st
import datetime
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM Models
def initialize_models():
    models = {}
    groq_api_key = os.getenv("GROQ_API_KEY")
    try:
        models["Llama-3.3-70B"] = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=None,
            api_key=groq_api_key
        )
        models["Gemma-2-9B"] = ChatGroq(
            model="gemma2-9b-it",
            temperature=0.7,
            max_tokens=None,
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"⚠️ Error initializing models: {str(e)}")
        return {}
    return models

# Get model response
def get_model_response(question, model):
    prompt = f"""You are a knowledgeable assistant who can provide information on a wide range of topics.  
USER QUESTION: {question}  

Your task:  
1. Analyze the user's question  
2. Provide a helpful, informative response
3. Be concise and factual  
4. If the question relates to financial advice, add appropriate disclaimers
"""
    try:
        response = model.invoke(prompt)
        return {"content": response.content}
    except Exception as e:
        return {"content": f"Error: {str(e)}"}
    
# --- Streamlit UI ---
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main, .stApp {background-color: #1e1e1e; color: #e0e0e0;}
.stChatMessage {background-color: #2d2d2d; border-radius: 10px; padding: 10px; margin: 10px 0;}
.stButton>button {background-color: #1f77b4; color: white; border-radius: 20px;}
.info-card {background-color: #2d2d2d; border-radius: 10px; padding: 20px; margin: 10px 0;}
h1, h2, h3, h4, h5, h6 {color: #e0e0e0;}
</style>
""", unsafe_allow_html=True)

available_models = initialize_models()

# Sidebar
with st.sidebar:
    st.title("🤖 AI Assistant")
    if available_models:
        selected_model = st.selectbox("Choose the AI model:", list(available_models.keys()))
    else:
        selected_model = None
    
    st.markdown("### Features")
    st.markdown("""
    - Answers to general knowledge questions
    - Explanations of complex topics
    - Creative content generation
    - Information about various subjects
    - Problem solving assistance
    """)
    
    st.markdown("### Sample Questions")
    st.markdown("""
    - What is quantum computing?
    - Explain how solar panels work
    - Tell me about the history of Rome
    - What are the benefits of regular exercise?
    - How does blockchain technology work?
    """)

# Main content
st.title("🤖 AI Assistant")
st.markdown(f"""
<div class="info-card">
    <h3>Welcome to your AI Assistant!</h3>
    <p>Ask me anything! I can help you with:</p>
    <ul>
        <li>General knowledge and information</li>
        <li>Explanations of complex topics</li>
        <li>Creative content and ideas</li>
        <li>Problem solving and advice</li>
    </ul>
    <p><strong>Current Model:</strong> {selected_model if selected_model else "No model available"}</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        if selected_model:
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            with st.spinner("Generating response..."):
                response_data = get_model_response(prompt, available_models[selected_model])
            
            message_placeholder.markdown(response_data["content"])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_data["content"]})
        else:
            st.markdown("Please select an AI model from the sidebar.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()