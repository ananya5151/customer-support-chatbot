import requests
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. SET UP THE LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 2. DEFINE THE TOOLS USING THE @tool DECORATOR ---

# Load our data
product_df = pd.read_csv('data/products.csv')

@tool("Product Search Tool")
def search_products(query: str) -> str:
    """Searches the product database for a given query to find information like price and stock."""
    results = product_df[product_df['product_name'].str.contains(query, case=False)]
    if not results.empty:
        return results.to_string()
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
    """
    Checks the status of a specific order by its ID. Use this tool when a user asks for their order status.
    """
    # The URL of our running mock_api.py server
    api_url = f"http://127.0.0.1:5000/order_status/{order_id}"
    print(f"--- Contacting API at: {api_url} ---") # A small debug print
    try:
        response = requests.get(api_url)
        # Check if the API returned a successful response
        if response.status_code == 200:
            return f"Status for order {order_id}: {response.json()}"
        else:
            return f"Could not find order with ID {order_id}. Please double-check the ID."
    except requests.exceptions.ConnectionError:
        return "The order status service is currently unavailable. Please try again later."

# --- 3. CREATE AGENT ---
support_agent = Agent(
    role="Senior Customer Support Agent",
    goal="Be the most helpful and friendly customer support agent. Provide accurate information about products and company policies.",
    backstory=(
        "You are a seasoned customer support agent at a cutting-edge e-commerce company. "
        "You are an expert in the company's products and policies. "
        "You are known for your patience, clarity, and wit. "
        "You always strive to provide the best possible customer experience."
    ),
    tools=[search_products, search_faq, check_order_status],
    llm=llm,
    allow_delegation=False,
    verbose=True
    memory=True
)

# --- 4. DEFINE THE TASK ---
support_task = Task(
    description="Answer the user's query: '{query}'. Be helpful and concise.",
    expected_output="A clear and friendly answer to the user's question, based on the information found in the tools.",
    agent=support_agent
)

# --- 5. CREATE THE CREW ---
support_crew = Crew(
    agents=[support_agent],
    tasks=[support_task],
    process=Process.sequential,
    verbose=2
)


# --- 6. RUN THE CREW ---
if __name__ == "__main__":
    print("🤖 Chatbot initialized. How can I help you today?")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye! 👋")
            break
        
        result = support_crew.kickoff(inputs={'query': user_query})
        
        print("\nChatbot:", result)
        print("-" * 50)