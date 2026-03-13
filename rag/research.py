import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Page Setup
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")
st.title("🔬 AI Research Tool")

# 3. Model Initialization
# temperature=0.3 is better for research to keep it factual
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 4. Define the Prompt Template
# Using {task_format} instead of {format} to avoid conflict with Python's format() function
template = """You are a highly capable research assistant. 
Your task is to provide a {task_format} of the research paper titled: '{paper}'.

Please provide the response in the following language: {language}.
Ensure the information is accurate and structured professionally."""

prompt = PromptTemplate.from_template(template)

# 5. Create the Chain (LCEL)
chain = prompt | model | StrOutputParser()

# 6. Streamlit UI Elements
col1, col2 = st.columns(2)

with col1:
    paper = st.selectbox("Select a research paper", [
        "Attention Is All You Need", 
        "LSTM (Long Short-Term Memory)", 
        "Transformer-XL", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Gemini: A Family of Highly Capable Multimodal Models"
    ])
    language = st.selectbox("Select target language", ["English", "Nepali", "Hindi", "French", "German"])

with col2:
    task_format = st.selectbox("Select a task", ["Summary", "Key Points", "Core Facts"])

# 7. Execution Logic
if st.button("Generate Analysis"):
    # Formatting and Invoking happens only when the button is clicked
    with st.spinner(f"Analyzing {paper}..."):
        try:
            response = chain.invoke({
                "paper": paper,
                "language": language,
                "task_format": task_format
            })
            
            st.markdown("---")
            st.subheader(f"Analysis: {paper}")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")