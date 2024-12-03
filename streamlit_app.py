import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers.generation.streamers import BaseStreamer
from peft import PeftModel

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
load_css("styles.css")

# --- Constants ---
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL = "Llama-3-8B-Caption-RAG"

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- GPU Check ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    st.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    st.error("No GPU found. Using CPU instead.")

# --- Utility Functions ---
@st.cache_resource
def load_model():
    """
    Load the tokenizer and model, and ensure both are on the same device.
    """

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Load the Peft model (adapter) on top of the base model
    model = PeftModel.from_pretrained(model, NEW_MODEL)

    # Merge the adapter layers into the base model and unload them
    model = model.merge_and_unload()

    model = model.to(device)  # Move the model to the correct device

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# --- Custom Streamlit Streamer ---
class StreamlitTextStreamer(BaseStreamer):
    def __init__(self, tokenizer, output_placeholder):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_placeholder = output_placeholder
        self.current_output = ""

    def put(self, value):
        """Called when a token is generated."""
        # Ensure value is a tensor, convert it to a list, and decode
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        token = self.tokenizer.decode(value, skip_special_tokens=True)
        self.current_output += token
        # Update the Streamlit placeholder in real-time
        self.output_placeholder.markdown(f"**{self.current_output}**")

    def end(self):
        """Called when generation is finished."""
        pass  # Optional cleanup

## --- Streamlit App ---
st.markdown(
    "<h1 style='text-align: center;'>🫦 Tasty Caption Generation 💦</h1>",
    unsafe_allow_html=True
)

# Input fields in the main section
instruction = st.text_input("Enter Instruction:", placeholder="Generate a *Category* Caption")
input_text = st.text_area("Enter Context:", placeholder="Describe the Caption")

# Floating "Contact Support" Button
st.markdown("""
    <div class="floating-support", '_blank')">Contact Support</div>
""", unsafe_allow_html=True)

# Function to reset all sliders to default values
def reset_to_defaults():
    for key, value in default_settings.items():
        st.session_state[key] = value

# Sidebar for settings
with st.sidebar:
    st.header("Generation Settings")

    # Default values
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

    # Number of Captions Slider
    st.markdown(
        "<div class='inline-label'>Total Captions<div class='tooltip'>❔"
        "<span class='tooltiptext'>Select the number of captions to generate.</span>"
        "</div></div>", unsafe_allow_html=True)
    st.slider(
        "", min_value=1, max_value=20,
        key="num_captions"
    )

    # Maximum Output Length Slider
    st.markdown(
        "<div class='inline-label'>Max Tokens<div class='tooltip'>❔"
        "<span class='tooltiptext'>Determines the maximum length of the generated caption.</span>"
        "</div></div>", unsafe_allow_html=True)
    st.select_slider(
        "", options=[256, 512, 1024],
        key="max_length"
    )

    # Temperature Slider
    st.markdown(
        "<div class='inline-label'>Temperature<div class='tooltip'>❔"
        "<span class='tooltiptext'>Lower values makes output more focused/predictable, higher values make it more creative/unique.</span>"
        "</div></div>", unsafe_allow_html=True)
    st.slider(
        "", min_value=0.0, max_value=1.5,
        step=0.10, key="temperature"
    )

    # Top-K Sampling Slider
    st.markdown(
        "<div class='inline-label'>Top-K<div class='tooltip'>❔"
        "<span class='tooltiptext'>Lower values limit options to the most common words, higher values include less common words for variety.</span>"
        "</div></div>", unsafe_allow_html=True)
    st.slider(
        "", min_value=0, max_value=100,
        step=10,key="top_k"
    )

    # Top-P Sampling Slider
    st.markdown(
        "<div class='inline-label'>Top-P<div class='tooltip'>❔"
        "<span class='tooltiptext'>Lower values stick to safer word choices, higher values add mores variety.</span>"
        "</div></div>", unsafe_allow_html=True)
    st.slider(
        "", min_value=0.0, max_value=1.0,
        step=0.10, key="top_p"
    )

    # Add spacer for better organization
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Default Settings Button at the Bottom
    st.button("Default Settings", on_click=reset_to_defaults)

# Initialize session state for storing generated captions
if "generated_captions" not in st.session_state:
    st.session_state["generated_captions"] = []

# Generate button in the main section
if st.button("Generate Captions"):
    if not instruction.strip() or not input_text.strip():
        st.error("Instruction and context cannot be empty.")
    else:
        st.write(f"Generating {st.session_state['num_captions']} captions...")
        prompt = alpaca_prompt.format(instruction, input_text, "")

        # Initialize generator and ensure it runs on the GPU
        generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=0  # Explicitly set to GPU device 0
        )

        # Clear the stored captions before generating new ones
        st.session_state["generated_captions"] = []

        # Generate multiple captions
        for i in range(st.session_state["num_captions"]):
            current_output = ""
            
            # Generate caption for each iteration
            for token in generator(
                prompt,
                max_length=st.session_state["max_length"],
                temperature=st.session_state["temperature"],
                top_k=st.session_state["top_k"],
                top_p=st.session_state["top_p"],
                do_sample=True,
                return_full_text=False,  # Stream partial output
            ):
                # Extract the text generated so far
                current_output += token["generated_text"]
            
            # Add the generated caption to session state
            st.session_state["generated_captions"].append(f"**Caption {i + 1}:** {current_output}")

# Display the captions stored in session state
if st.session_state["generated_captions"]:
    st.write("Generated Captions:")
    for caption in st.session_state["generated_captions"]:
        st.markdown(caption)
