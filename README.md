# Intelligent E-Commerce Chatbot

This project is a fully functional customer support chatbot designed to provide intelligent, automated assistance for a fictional e-commerce platform. It leverages a sophisticated AI agent framework (CrewAI) to understand user queries and interact with various tools to provide accurate information.

---

## 🚀 Live Demo

**[https://customer-support-chatbot-fdk67i9me648p7hr9ujnhx.streamlit.app/]**

---

## ✨ Key Features

* **Multi-Tool Agent:** The chatbot can dynamically choose between different tools to answer questions:
    * **Product Search:** Looks up product information (price, stock) from a CSV database.
    * **FAQ Search:** Retrieves answers to common questions about store policies (shipping, returns).
    * **Order Status:** Connects to a mock API to check the real-time status of orders.
* **Robust Error Handling:** Features advanced prompt engineering to gracefully handle ambiguous ("I need help") and off-topic ("tell me a joke") queries without crashing or looping.
* **Conversational Memory:** Remembers the context of the conversation to provide more natural and coherent interactions.
* **Web Interface:** A clean, user-friendly chat interface built with Streamlit.
* **Decoupled Architecture:** Built with a separate frontend (Streamlit) and backend (Flask), a standard practice for scalable web applications.

---

## 🛠️ Tech Stack

* **Backend & AI:** Python, CrewAI, LangChain, Flask
* **LLM Provider:** Groq (for high-speed Llama 3 inference)
* **Frontend:** Streamlit
* **Data Handling:** Pandas
* **Deployment:** Streamlit Community Cloud

---

## 📂 Architecture

The application runs as two separate processes that communicate with each other:
1.  **Backend API (`backend_api.py`):** A Flask server that initializes the CrewAI agent and exposes a `/chat` endpoint. This is the "brain" of the operation.
2.  **Frontend UI (`app.py`):** A Streamlit application that provides the user interface. It sends user queries to the Flask backend and displays the response.

---

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/customer-support-chatbot.git](https://github.com/YOUR_USERNAME/customer-support-chatbot.git)
    cd customer-support-chatbot
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On Mac/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your environment variables:**
    * Create a file named `.env`.
    * Add your Groq API key: `GROQ_API_KEY="YOUR_API_KEY_HERE"`
5.  **Run the application:**
    * **Terminal 1 (Run the Backend):** `python backend_api.py`
    * **Terminal 2 (Run the Frontend):** `streamlit run app.py`

---

## 📸 Screenshot

<img width="1902" height="875" alt="Screenshot 2025-07-24 194431" src="https://github.com/user-attachments/assets/8f28191e-8073-438a-ae99-cb21b03fdff3" />


