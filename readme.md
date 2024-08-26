# Conversational Chat App using Streamlit

This is a Streamlit app that demonstrates a conversational chat interface powered by a language model and a retrieval-based system. The app allows you to have interactive conversations with the model about a given CSV dataset.

## LangChain + Streamlit🔥+ Llama

* https://ai.plainenglish.io/%EF%B8%8F-langchain-streamlit-llama-bringing-conversational-ai-to-your-local-machine-a1736252b172

LangChain + Streamlit🔥+ Llama 🦙: Bringing Conversational AI to Your Local Machine 🤯
Integrating Open Source LLMs and LangChain for Free Generative Question Answering (No API Key required)

Here is an overview of the blog’s structure, outlining the specific sections that will provide a detailed breakdown of the process:

- Setting up the virtual environment and creating file structure
- Getting LLM on your local machine
- Integrating LLM with LangChain and customizing PromptTemplate
- Document Retrieval and Answer Generation
- Building application using Streamlit
  
### User Note : Please 1st Download and place the LLAMA 2B model into native directory before running the program.

Download LLAMA 2b here : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin

(size of model is 6gbs so not possible to push if not in repo (may have automatically been rejected by repo management system)

## Prerequisites

- Python 3.7 or later
- Streamlit (`pip install streamlit`)

## Installation

1. Clone this repository to your local machine:


2. Install the required Python packages:
```
pip install -r requirements.txt
```


## Usage

1. Run the Streamlit app by executing the following command in your terminal:

```
streamlit run app.py
```


The app will open in your default web browser.

2. Upload a CSV file by using the file uploader in the sidebar.

3. Start a conversation by typing a query in the input box and clicking the "Send" button.

4. The app will display a chat history, showing both user inputs and the model's responses.

## Features

- Interactive conversation with a language model.
- Retrieval-based responses using embeddings and FAISS index.
- Easy integration with various language models (e.g., Llama, Vicuna, Alpaca).
- Preserves context in conversation history.

## Customization

You can customize the app by modifying the `app.py` script:

- Change the language model by updating the `load_llm` function.
- Modify the CSV loader and embeddings to suit your dataset and requirements.
- Adjust the UI layout, styling, and components.

## Acknowledgments

This app is built using Streamlit and several libraries from the LangChain project, including document loaders, embeddings, vector stores, and conversational chains. Special thanks to the LangChain team for their contributions.

### Demo
#### Our sample csv
![img_1.png](img_1.png)

#### Upload csv here :
![img.png](img.png)

#### Asking questions about the csv
![img_2.png](img_2.png)
![img_3.png](img_3.png)


Integrating **LangChain**, **Streamlit**, and **Llama** can create powerful applications that combine natural language processing, user-friendly interfaces, and advanced large language models. Here's how you could approach this integration:

### 1. **LangChain:**
   LangChain provides tools to build applications powered by language models. It includes functionality like chaining LLM prompts, managing memory, interacting with external APIs, and accessing external data sources such as databases.

### 2. **Streamlit:**
   Streamlit is a fast and easy way to create interactive web apps with Python. You can use it to create simple, intuitive interfaces for interacting with LangChain-powered backends.

### 3. **Llama (LLM):**
   Llama models are high-performance, open-source language models. Combining Llama with LangChain allows you to leverage Llama’s capabilities within the structured framework LangChain provides.

---

### Steps to Build a LangChain + Streamlit + Llama App

#### **1. Set up LangChain and Llama:**

You’ll need to install LangChain and a framework for accessing the Llama model, such as HuggingFace or any compatible API.

```bash
pip install langchain streamlit transformers
```

#### **2. Load Llama Model:**

You can use HuggingFace's `transformers` library to load the Llama model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Llama model and tokenizer
model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a method to generate text
def generate_llama_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### **3. Create a LangChain Chain:**

LangChain provides various modules to chain together prompts and handle logic.

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM

# Define a custom Llama model wrapper to integrate it with LangChain
class LlamaLLM(LLM):
    def _call(self, prompt, stop=None):
        return generate_llama_response(prompt)

# Setup LangChain LLM chain
llama_llm = LlamaLLM()

# Create a prompt template and chain it
template = "You are a helpful assistant. Answer the following question: {question}"
prompt = PromptTemplate(input_variables=["question"], template=template)
llm_chain = LLMChain(llm=llama_llm, prompt=prompt)
```

#### **4. Build the Streamlit Interface:**

Use Streamlit to build the front-end interface, where users can input questions or prompts.

```python
import streamlit as st

# Streamlit UI
st.title("LangChain + Llama + Streamlit")

# User input
user_question = st.text_input("Enter your question:")

# Display response
if st.button("Submit"):
    if user_question:
        response = llm_chain.run({"question": user_question})
        st.write(f"Response: {response}")
    else:
        st.write("Please enter a question!")
```

---

### How It Works:
- **LangChain**: Manages the LLM and constructs the prompts.
- **Llama**: Handles the language model's response generation.
- **Streamlit**: Provides an interactive front-end for users to input their queries.

### Benefits:
- **Modular Design**: You can easily extend the system by adding more complex chains, additional data sources, or even integrating with APIs.
- **User-Friendly**: Streamlit simplifies front-end development, making it easy to build web interfaces.
- **Scalable NLP**: Llama models, combined with LangChain, allow you to build scalable, advanced language-based applications.

This setup enables you to build a flexible and interactive application that leverages advanced LLM capabilities through a simple web interface!

