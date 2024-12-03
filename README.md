# Caption Generation Streamlit

A Streamlit application for generating captions efficiently using a GPU-accelerated pipeline. Follow the steps below to set up and run the project.

---

## Prerequisites

- Python 3.8 or later installed.
- A **dedicated GPU** is required for optimal performance.

---

## Installation Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Benchoonngg/streamlit-tasty-caption-generation.git
cd streamlit-tasty-caption-generation
```


### Step 2: Create a virtual environment
```python -m venv venv```

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

### Step 3: Install Streamlit
pip install streamlit

### Step 4: Install the Correct Dependencies
Choose your operating system and install the appropriate requirements file:
pip install -r requirements-windows.txt
pip install -r requirements-mac.txt

### Step 5: Run the Streamlit App
Run the application using the following command:
streamlit run streamlit_app.py

# Note :
Ensure you have a dedicated GPU installed and configured to leverage CUDA capabilities for caption generation.
