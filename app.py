# app.py (Final Deployment Version)

import streamlit as st
import os
import pandas as pd
import requests
import json
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Support Chat", page_icon="🛍️")

# --- HELPER FUNCTION TO CREATE THE CREW ---
@st.cache_resource
def create_chatbot_crew():
    """
    This function creates and caches the CrewAI crew.
    It's designed to be called once and reused.
    """
    load_dotenv()

    # Define the strict operational guidelines for the agent
    guidelines = """You are a highly-trained AI Support Specialist. Your primary goal is to assist users by accessing internal company data using the tools you've been given. You are professional, concise, and adhere strictly to your operational guidelines.

    **Operational Guidelines:**
    - **NEVER** make up answers or provide information that does not come directly from a tool's output. Your knowledge is limited to your tools.
    - If a tool returns no information, you must state that you could not find the information.
    - If a user query is ambiguous, vague, or missing information (e.g., "I need help", "the price is wrong"), you MUST ask a clarifying question to get the information you need. To do this, you will conclude your work by providing your question as the 'Final Answer'.
    - If a user query is off-topic or outside your scope (e.g., "tell me a joke," "what's the weather"), you MUST conclude your work with the 'Final Answer:' using the exact phrase: "I'm sorry, I can only assist with questions related to our products, store policies, and order statuses."
    """

    # --- LLM SETUP (Using Groq) ---
    llm = ChatGroq(
        model="llama3-8b-8192",
        verbose=True,
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # --- TOOLS DEFINITION ---
    product_df = pd.read_csv('data/products.csv')

    @tool("Product Search Tool")
    def search_products(query: str) -> str:
        """Searches the product database for a given query to find information like price and stock."""
        name_matches = product_df['product_name'].str.contains(query, case=False)
        category_matches = product_df['category'].str.contains(query, case=False)
        results = product_df[name_matches | category_matches]
        if not results.empty:
            return json.dumps(results.to_dict(orient='records'))
        return "No products found matching that description."

    @tool("FAQ Search Tool")
    def search_faq(query: str) -> str:
        """Searches the FAQ file for answers to common questions about company policies like shipping and returns."""
        with open('data/faq.txt', 'r') as f:
            faq_content = f.read()
        relevant_sections = []
        for section in faq_content.split('\n\n'):
            if query.lower() in section.lower():
                relevant_sections.append(section)
        if relevant_sections:
            return "\n\n".join(relevant_sections)
        return "I couldn't find an answer to that in the FAQ."

    # --- AGENT DEFINITION ---
    support_agent = Agent(
        role="E-Commerce Support Specialist",
        goal="Provide accurate, tool-based support to users. Follow your operational guidelines precisely.",
        backstory=guidelines,
        tools=[search_products, search_faq], # IMPORTANT: Order Status tool is removed for deployment
        llm=llm,
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    # --- TASK DEFINITION ---
    support_task = Task(
        description=f"""Analyze the user's query and provide a helpful response.
        **User Query:** {{query}}
        **Your Operational Guidelines:**\n{guidelines}""",
        expected_output="A helpful and accurate answer, a clarifying question, or a polite refusal based on your operational guidelines.",
        agent=support_agent
    )

    # --- CREW DEFINITION ---
    crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        process=Process.sequential,
        verbose=2
    )
    return crew

# --- MAIN APP LOGIC ---
st.title("🛍️ E-Commerce Customer Support")
st.write("Welcome! I can help you with product questions and our store policies. How can I assist you today?")

# Get the cached crew
try:
    crew = create_chatbot_crew()

    # --- CHAT HISTORY MANAGEMENT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- USER INPUT HANDLING ---
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = crew.kickoff(inputs={'query': prompt})
                st.markdown(result)
        
        st.session_state.messages.append({"role": "assistant", "content": result})

except Exception as e:
    st.error(f"An error occurred: {e}. Please check your API keys and environment setup.")