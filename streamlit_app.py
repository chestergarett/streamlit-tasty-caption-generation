import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers.generation.streamers import BaseStreamer
from peft import PeftModel

# --- Constants ---
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL = "Benchoonngg/Tasty-Unsloth-Llama-3.1-8B-v4"
access_token = "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"

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

# --- Streamlit App ---
st.title("Llama 3.1 Caption Generator with GPU Support")

instruction = st.text_input("Enter Instruction:", "Generate a Holiday Caption.")
input_text = st.text_area("Enter Context:", "Describe a Holiday Caption in a 4th of July content.")
max_length = st.slider("Max Output Length:", min_value=128, max_value=1024, value=256)

if st.button("Generate Caption"):
    if not instruction.strip() or not input_text.strip():
        st.error("Instruction and context cannot be empty.")
    else:
        prompt = alpaca_prompt.format(instruction, input_text, "")

        # Initialize generator and ensure it runs on the GPU
        generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=0  # Explicitly set to GPU device 0
        )
        output_placeholder = st.empty()  # Placeholder for dynamic updates

        # Custom real-time caption generation logic
        current_output = ""
        for token in generator(
            prompt,
            max_length=max_length,
            temperature=0.90,
            top_k=50,
            top_p=1,
            do_sample=True,
            return_full_text=False,  # Stream partial output
        ):
            # Extract the text generated so far
            current_output += token["generated_text"]
            # Update the Streamlit placeholder in real-time
            output_placeholder.markdown(f"**{current_output}**")
