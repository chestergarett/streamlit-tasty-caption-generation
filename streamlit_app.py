import streamlit as st
import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv 

load_dotenv() 

def login_page():
    """Display login page and handle authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        return False
    return True

def initialize_api_credentials():
    """Initialize API credentials after successful login"""
    # Define the required scope
    scope = "https://www.googleapis.com/auth/cloud-platform"
    service_account_info = st.secrets["credentials"]

    # Load the service account credentials
    vertex_credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=[scope]
    )

    # Refresh the credentials to get an access token
    vertex_credentials.refresh(Request())
    return vertex_credentials.token

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main Streamlit App
def main():
    # First check login
    if not login_page():
        st.stop()  # Stop execution if not logged in
    
    # Load CSS file
    load_css("styles.css")
    
    # Initialize API credentials after successful login
    access_token = initialize_api_credentials()
    
    # Streamlit App Title
    st.markdown(
        "<h1 style='text-align: center;'>🫦 Tasty Caption Generation 💦</h1>",
        unsafe_allow_html=True
    )
    
    # Welcome message with username
    st.markdown(f"Welcome, {st.session_state.username}!")
    
    # Initialize session state for storing captions if it doesn't exist
    if 'generated_captions' not in st.session_state:
        st.session_state.generated_captions = []

    # Main content: Input fields and caption generation
    instruction = st.text_input("Enter Instruction:", placeholder="Generate a *Category* Caption")
    input_text = st.text_area("Enter Context:", placeholder="Describe the Caption")
    
    if st.button("Generate Captions"):
        if not instruction.strip() or not input_text.strip():
            st.error("Instruction and context cannot be empty.")
        else:
            st.write(f"Generating {st.session_state['num_captions']} captions...")
            # Clear previous captions
            st.session_state.generated_captions = []

            # Generate captions logic
            for i in range(st.session_state["num_captions"]):
                response = generate_caption_from_api(
                    instruction,
                    input_text,
                    st.session_state["max_length"],
                    st.session_state["temperature"],
                    st.session_state["top_k"],
                    st.session_state["top_p"],
                    access_token
                )
                
                if response:
                    # Store caption in session state
                    st.session_state.generated_captions.append(response)

    # Display stored captions
    for i, caption in enumerate(st.session_state.generated_captions):
        st.write(f"**Caption {i + 1}:** {caption}")

    # Generation Settings in Sidebar
    with st.sidebar:
        st.header("Generation Settings")
        default_settings = {
            "num_captions": 1,
            "max_length": 1024,
            "temperature": 0.90,
            "top_k": 50,
            "top_p": 0.90
        }

        # Initialize session state for sliders
        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Sliders and inputs for settings
        st.slider("Number of Captions", min_value=1, max_value=100, key="num_captions")
        st.select_slider("Max Tokens", options=[256, 512, 1024], key="max_length")
        st.slider("Temperature", min_value=0.0, max_value=1.5, step=0.10, key="temperature")
        st.slider("Top-K", min_value=0, max_value=100, step=10, key="top_k")
        st.slider("Top-P", min_value=0.0, max_value=1.0, step=0.10, key="top_p")

def generate_caption_from_api(instruction, input_text, max_length, temperature, top_k, top_p, access_token):
    """Function to generate caption using API call to Vertex AI."""
    # Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    # Define the endpoint URL using the endpoints section
    url = f"https://{st.secrets.endpoints.ENDPOINT_DNS}/v1beta1/{st.secrets.endpoints.ENDPOINT_RESOURCE_NAME}/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": alpaca_prompt.format(instruction, input_text, "")}],
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Send the POST request to the API
    response = requests.post(
        url, headers={"Authorization": f"Bearer {access_token}"}, json=payload, stream=True
    )
    
    if not response.ok:
        raise ValueError(response.text)

    result = []  # List to accumulate the chunks
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
        if chunk:
            chunk = chunk.decode("utf-8").removeprefix("data:").strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if type(data) is not dict or "error" in data:
                raise ValueError(data)
            delta = data["choices"][0]["delta"].get("content")
            if delta:
                result.append(delta)  # Accumulate the chunks
    full_result = ''.join(result)
    return full_result

# Start the Streamlit app
if __name__ == "__main__":
    main()