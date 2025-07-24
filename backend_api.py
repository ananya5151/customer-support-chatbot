# backend_api.py

import os
import pandas as pd
import requests
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

# --- Mock Order Database ---
mock_orders = {
    "ORD12345": {"status": "Shipped", "estimated_delivery": "2 days"},
    "ORD67890": {"status": "Processing", "estimated_delivery": "5 days"},
    "ORD54321": {"status": "Delivered", "estimated_delivery": "N/A"}
}

# --- API Endpoint for Mock Orders ---
@app.route('/order_status/<order_id>', methods=['GET'])
def get_order_status(order_id):
    order = mock_orders.get(order_id)
    if order:
        return jsonify(order)
    else:
        return jsonify({"error": "Order not found"}), 404

# --- This is the new, centralized set of instructions ---
guidelines = """You are a highly-trained AI Support Specialist. Your primary goal is to assist users by accessing internal company data using the tools you've been given. You are professional, concise, and adhere strictly to your operational guidelines.

**Operational Guidelines:**
- **NEVER** make up answers or provide information that does not come directly from a tool's output. Your knowledge is limited to your tools.
- If a tool returns no information, you must state that you could not find the information.
- If a user query is ambiguous, vague, or missing information (e.g., "I need help", "the price is wrong"), you MUST ask a clarifying question to get the information you need. To do this, you will conclude your work by providing your question as the 'Final Answer'.
- If a user query is off-topic or outside your scope (e.g., "tell me a joke," "what's the weather"), you MUST conclude your work with the 'Final Answer:' using the exact phrase: "I'm sorry, I can only assist with questions related to our products, store policies, and order statuses."
"""

def create_chatbot_crew():
    """This function creates and returns the CrewAI crew."""
    load_dotenv()

    llm = ChatGroq(
        model="llama3-8b-8192",
        verbose=True,
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

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

    @tool("Order Status Tool")
    def check_order_status(order_id: str) -> str:
        """Checks the status of a specific order by its ID by calling the order status API."""
        api_url = f"http://127.0.0.1:5000/order_status/{order_id}"
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                return f"Status for order {order_id}: {response.json()}"
            else:
                return f"Could not find order with ID {order_id}."
        except requests.exceptions.RequestException:
            return "The order status service is currently unavailable."

    # --- AGENT DEFINITION ---
    support_agent = Agent(
        role="E-Commerce Support Specialist",
        goal="Provide accurate, tool-based support to users. Follow your operational guidelines precisely.",
        backstory=guidelines, # The agent's long-term memory of the rules
        tools=[search_products, search_faq, check_order_status],
        llm=llm,
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    # --- TASK DEFINITION ---
    support_task = Task(
        # The task now includes the guidelines for immediate context
        description=f"""Analyze the user's query and provide a helpful response.
        
        **User Query:**
        {{query}}

        **Your Operational Guidelines:**
        {guidelines}
        """,
        expected_output="A helpful and accurate answer, a clarifying question, or a polite refusal based on your operational guidelines.",
        agent=support_agent
    )

    crew = Crew(
        agents=[support_agent],
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
        return jsonify({"error": "Chatbot crew not initialized"}), 500

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        result = chatbot_crew.kickoff(inputs={'query': query})
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Initializing CrewAI chatbot...")
    chatbot_crew = create_chatbot_crew()
    print("Chatbot Initialized. Starting API server...")
    app.run(host='0.0.0.0', port=5000)