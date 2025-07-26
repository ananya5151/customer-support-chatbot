# backend_api.py

import os
import json
from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Initialize Flask App
app = Flask(__name__)

# --- Global variable for the crew ---
chatbot_crew = None

def create_chatbot_crew():
    """This function creates and returns the specialized, single-agent CrewAI crew."""
    load_dotenv()

    llm = ChatGroq(
        model="llama3-8b-8192",
        verbose=True,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    product_data = []
    with open('data/products.jsonl', 'r') as f:
        for line in f:
            product_data.append(json.loads(line))
    
    faq_data = []
    with open('data/faq.jsonl', 'r') as f:
        for line in f:
            faq_data.append(json.loads(line))

    @tool("Store Information Search Tool")
    def search_store_knowledge(query_dict: dict) -> str: # <-- FIX: Expects a dictionary now
        """
        Searches the MEN-KID-WOMEN store's product catalog and FAQ database. 
        Use this for any question about products (price, stock, availability) or policies (shipping, returns).
        """
        # --- THE CRUCIAL FIX IS HERE ---
        # The agent sends a dictionary like {'query': 'user question'}. We need to extract the actual text.
        query = query_dict.get('query', '')
        query_lower = query.lower()
        # --- END OF FIX ---

        # --- INTELLIGENT AMBIGUITY CHECK ---
        ambiguous_terms = ['jeans', 'shirt', 't-shirt', 'trousers', 'shoes', 'jacket']
        genders_in_query = any(g in query_lower for g in ['men', 'woman', 'women', 'kid', 'kids', 'boy', 'girl'])
        
        for term in ambiguous_terms:
            if term == query_lower and not genders_in_query:
                # If the query is ambiguous, the TOOL ITSELF returns the clarifying question.
                return f"CLARIFICATION_NEEDED: Of course! Are you looking for {term} for men, women, or kids?"
        # --- END OF INTELLIGENT CHECK ---

        # Search FAQs first for policy questions
        faq_results = [qa for qa in faq_data if query_lower in qa['question'].lower()]
        if faq_results:
            return json.dumps(faq_results)

        # If not in FAQ, search products
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
        backstory="""You are a simple but powerful AI assistant. Your only job is to take a user's question, use your single search tool, and report the result. If the tool gives you a question to ask (prefixed with 'CLARIFICATION_NEEDED:'), your final answer is that exact question. Otherwise, your final answer is a summary of the data the tool returned. You must be decisive and never use the tool more than once per query.""",
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

# --- Main Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    global chatbot_crew
    if not chatbot_crew:
        chatbot_crew = create_chatbot_crew()
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        result = chatbot_crew.kickoff(inputs={'query': query})
        if "CLARIFICATION_NEEDED:" in result:
            result = result.replace("CLARIFICATION_NEEDED:", "").strip()
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Initializing Intelligent Assistant...")
    chatbot_crew = create_chatbot_crew()
    print("Assistant Initialized. Starting API server...")
    app.run(host='0.0.0.0', port=5000)