import streamlit as st
import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv 
from constants import context_sample_guide

load_dotenv() 

# --- Authentication Functions ---
def check_credentials(username, password):
    """Check if username/password combination exists in authorized users"""
    try:
        # Debug: Print all available secrets keys
        st.write("Available secrets:", st.secrets.keys())
        
        # Debug: Print authorized users section
        if 'authorized_users' in st.secrets:
            st.write("Authorized users:", st.secrets.authorized_users)
        
        # Check if username exists and password matches
        if username in st.secrets.authorized_users:
            return st.secrets.authorized_users[username] == password
        return False
    except Exception as e:
        st.error(f"Debug - Error in check_credentials: {str(e)}")
        return False

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

def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        "authenticated": False,
        "username": None,
        "generated_captions": [],
        "num_captions": 1,
        "max_length": 1024,
        "temperature": 0.90,
        "top_k": 50,
        "top_p": 0.90
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    return defaults
            
    

def add_logout_button():
    """Add a logout button to the sidebar"""
    if st.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# Main Streamlit App
def main():
    # Initialize session state
    defaults = initialize_session_state()
    
    default_settings = {
            "num_captions": defaults['num_captions'],
            "max_length": defaults['max_length'],
            "temperature": defaults['temperature'],
            "top_k": defaults['top_k'],
            "top_p": defaults['top_p']
        }
    
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

    if 'show_markdown' not in st.session_state:
        st.session_state.show_markdown = False

    toggle_button = st.button("Show/Hide Sample Guide")
    if toggle_button:
        st.session_state.show_markdown = not st.session_state.show_markdown

    # Display markdown content if the button is clicked
    if st.session_state.show_markdown:
        st.markdown(context_sample_guide)

    # Main content: Input fields and caption generation
    instruction_options = [
        "Tip Me",
        "Winner",
        "Holiday",
        "Bundle",
        "Descriptive",
        "Spin the Wheel",
        "Girlfriend",
        "List",
        "Short",
        "Sub Promo",
        "VIP",
    ]
    # Main content: Input fields and caption generation
    selected_option = st.selectbox("Select Category:", options=instruction_options, )
    instruction = f"Generate a {selected_option} caption"
    input_text = st.text_area("Enter Context:", placeholder="Describe the Caption")
    
    if st.button("Generate Captions"):
        is_valid, error_message = validate_inputs(instruction, input_text)
        if not is_valid:
            st.error(error_message)
        else:
            with st.spinner("Generating captions..."):
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

        # Generation Settings in Sidebar
        col1, col2 = st.columns(2)
    
        with col1:
            if st.button("Reset"):
                for key, value in default_settings.items():
                    st.session_state[key] = value

            # Add logout button to sidebar
        with col2:
            add_logout_button()

        # Initialize session state for sliders
        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Sliders and inputs for settings
        st.slider("Number of Captions", min_value=1, max_value=100, key="num_captions", help="The number of captions to generate.")
        st.select_slider("Max Tokens", options=[256, 512, 1024], key="max_length", help="The maximum number of tokens (words or pieces of words) to include in the generated text.")
        st.slider("Temperature", min_value=0.1, max_value=1.5, step=0.10, key="temperature",  help="Controls the randomness of predictions.")
        st.slider("Top-K", min_value=10, max_value=100, step=10, key="top_k", help="Lower values limit options to the most common words, higher values include less common words for variety.")
        st.slider("Top-P", min_value=0.1, max_value=1.0, step=0.10, key="top_p", help="Lower values stick to safer word choices, higher values add more variety.")

def generate_caption_from_api(
    instruction: str,
    input_text: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    access_token: str
) -> str:
    """
    Generate caption using API call to Vertex AI.
    
    Args:
        instruction: The instruction for caption generation
        input_text: The context for caption generation
        max_length: Maximum length of generated text
        temperature: Temperature for text generation
        top_k: Top-k parameter for sampling
        top_p: Top-p parameter for sampling
        access_token: Authentication token for API access
    
    Returns:
        str: Generated caption text
    
    Raises:
        ValueError: If API request fails or returns error
    """
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

def validate_inputs(instruction: str, input_text: str) -> tuple[bool, str]:
    """Validate user inputs before generation"""
    if not instruction.strip():
        return False, "Instruction cannot be empty"
    if not input_text.strip():
        return False, "Context cannot be empty"
    if len(input_text) > 1000:  # Example limit
        return False, "Context is too long (max 1000 characters)"
    return True, ""

# Start the Streamlit app
if __name__ == "__main__":
    main()
