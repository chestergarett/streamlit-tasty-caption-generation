# Tasty Caption Generation using Llama 3.1 8B

A Streamlit application for generating captions efficiently using a GPU-accelerated pipeline.

---

### Prerequisites

- Python 3.8 or later installed.
- A **dedicated GPU** is required for optimal performance.

---

### Installation Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Benchoonngg/streamlit-tasty-caption-generation.git
cd streamlit-tasty-caption-generation
```

### Step 2: Clone the Model from HuggingFace
```bash
git lfs install
git clone https://huggingface.co/Benchoonngg/Tasty-Unsloth-Llama-3.1-8B-v4
```

### Step 3: Create a virtual environment
```bash
python -m venv venv
```

### Activate the virtual environment
On Windows
```bash
./venv\Scripts\activate
```
On Mac/Linux
```bash
source venv/bin/activate
```

### Step 4: Install Streamlit
```bash
pip install streamlit
```

### Step 5: Install the Correct Dependencies
Choose your operating system and install the appropriate requirements file:
```bash
pip install -r requirements-windows.txt
```
```bash
pip install -r requirements-mac.txt
```

### Step 6: Run the Streamlit App
Run the application using the following command:
```bash
streamlit run streamlit_app.py
```

<br>## Note :
Ensure you have a dedicated GPU installed and configured to leverage CUDA capabilities for caption generation.
