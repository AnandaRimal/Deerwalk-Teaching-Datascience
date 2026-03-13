import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Page Configuration
st.set_page_config(page_title="Movie Recs", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

# 3. Initialize Gemini 2.5 Flash
# temperature=0.7 is great for recommendations to get diverse results
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 4. Define the Prompt Template
template = """You are a movie recommendation system. 
Based on the user's preferences, recommend three movies along with a brief description for each.

User Preferences: {user_preferences}

Recommendations:"""

prompt = PromptTemplate.from_template(template)

# 5. Create the Chain (LCEL)
# Piping the prompt to the model and then to a string parser for clean output
chain = prompt | model | StrOutputParser()

# 6. Streamlit UI
user_preferences = st.text_area("Enter your movie preferences (e.g., 'I love 90s sci-fi and thrillers'):", height=150)

if st.button("Recommend Movies"):
    if user_preferences.strip():
        with st.spinner("Finding the perfect movies for you..."):
            # Using .invoke() instead of calling the model directly
            response = chain.invoke({"user_preferences": user_preferences})
            
            st.markdown("### Your Recommendations:")
            st.write(response)
    else:
        st.warning("Please tell me what kind of movies you like first!")