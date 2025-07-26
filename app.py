# app.py (Final Deployment Version)

import streamlit as st
import os
import json
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MEN-KID-WOMEN Support", page_icon="🛍️")

# --- HELPER FUNCTION TO CREATE THE CREW (CACHED FOR PERFORMANCE) ---
@st.cache_resource
def create_chatbot_crew():
    """
    This function creates and caches the CrewAI crew.
    It's designed to be called once and reused.
    """
    load_dotenv()

    # --- LLM SETUP ---
    llm = ChatGroq(
        model="llama3-8b-8192",
        verbose=True,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # --- LOAD KNOWLEDGE BASE ---
    product_data = []
    with open('data/products.jsonl', 'r') as f:
        for line in f:
            product_data.append(json.loads(line))
    
    faq_data = []
    with open('data/faq.jsonl', 'r') as f:
        for line in f:
            faq_data.append(json.loads(line))

    # --- TOOLS DEFINITION ---
    @tool("Store Information Search Tool")
    def search_store_knowledge(query: str) -> str:
        """
        Searches the MEN-KID-WOMEN store's product catalog and FAQ database.
        This is the only tool and should be used for every query.
        """
        query_lower = query.lower()

        # Intelligent Ambiguity Check
        ambiguous_terms = ['jeans', 'shirt', 't-shirt', 'trousers', 'shoes', 'jacket']
        genders_in_query = any(g in query_lower for g in ['men', 'woman', 'women', 'kid', 'kids', 'boy', 'girl'])
        
        for term in ambiguous_terms:
            if term == query_lower and not genders_in_query:
                return f"CLARIFICATION_NEEDED: Of course! Are you looking for {term} for men, women, or kids?"
        
        # Search FAQs first
        faq_results = [qa for qa in faq_data if query_lower in qa['question'].lower()]
        if faq_results:
            return json.dumps(faq_results)

        # Then search products
        product_results = []
        for item in product_data:
            if (query_lower in item['name'].lower() or
                query_lower in item['category'].lower() or
                (item.get('gender') and query_lower in item['gender'].lower()) or
                query_lower in item['style'].lower()):
                product_results.append(item)
        
        if not product_results:
            return "I'm sorry, I couldn't find any information about that in our store's catalog or FAQ."
            
        return json.dumps(product_results[:5])

    # --- AGENT DEFINITION ---
    store_expert = Agent(
        role="Efficient Store Data Retriever",
        goal="For every user query, use your search tool once and only once. Then, based on the tool's output, provide a final answer.",
        backstory="""You are a simple but powerful AI assistant for 'MEN-KID-WOMEN'. Your only job is to take a user's question, use your single search tool, and report the result. If the tool gives you a question to ask (prefixed with 'CLARIFICATION_NEEDED:'), your final answer is that exact question. Otherwise, your final answer is a summary of the data the tool returned. You must be decisive and never use the tool more than once per query.""",
        tools=[search_store_knowledge],
        llm=llm,
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    # --- TASK DEFINITION ---
    support_task = Task(
        description="Answer the user's query using your search tool, following your strict operational procedure. The user's query is: {query}",
        expected_output="A helpful and accurate answer based on the single output from your tool.",
        agent=store_expert
    )

    # --- CREW DEFINITION ---
    crew = Crew(
        agents=[store_expert],
        tasks=[support_task],
        process=Process.sequential,
        verbose=2
    )
    return crew

# --- MAIN APP LOGIC ---
st.title("🛍️ MEN-KID-WOMEN Customer Support")
st.write("Welcome! I am an expert on our store's products and policies. How can I assist you today?")

try:
    crew = create_chatbot_crew()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about our products or policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = crew.kickoff(inputs={'query': prompt})
                if "CLARIFICATION_NEEDED:" in result:
                    result = result.replace("CLARIFICATION_NEEDED:", "").strip()
                st.markdown(result)
        
        st.session_state.messages.append({"role": "assistant", "content": result})

except Exception as e:
    st.error(f"An error occurred: {e}. Please check your API keys and environment setup.")