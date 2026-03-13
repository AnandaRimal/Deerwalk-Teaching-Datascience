import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Initialize the Gemini 2.5 Flash Model
# Setting temperature to 0 is better for classification tasks (consistency)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 3. Define the Prompt Template
template = """You are an expert email security analyst. 
Classify the following email content as either 'Spam' or 'Not Spam'. 
Provide a very brief reason for your decision.

Email Content:
{email_content}

Classification:"""

prompt = PromptTemplate.from_template(template)

# 4. Create a Chain (LCEL)
# This pipes the prompt into the model and parses the text output automatically
chain = prompt | model | StrOutputParser()

# 5. Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="🛡️")
st.title("📧 Email Spam Classifier")

email_input = st.text_area("Enter the email content:", height=200)

if st.button("Classify"):
    if email_input.strip():
        with st.spinner("Analyzing email..."):
            # Invoke the chain with the user input
            response = chain.invoke({"email_content": email_input})
            
            st.markdown("### Analysis Result:")
            st.write(response)
    else:
        st.warning("Please enter some text before clicking Classify.")