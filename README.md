# FinAgent: Multi-Agent Financial Insight Engine

FinAgent is a powerful multi-agent system designed to provide comprehensive financial insights and risk assessments from transaction data. It uses specialized AI agents for data processing, insight generation, and fraud detection, all exposed through a robust FastAPI backend and an interactive Streamlit frontend.

## Project Structure

```
. \
├── app.py
├── config.py
├── orchestrator.py
├── req.txt
├── streamlit_app.py
├── transactions.csv
├── agents/
│   ├── data_agent.py
│   ├── insight_agent.py
│   └── risk_agent.py
├── api/
│   └── main.py
├── chroma_db/ (runtime generated)
├── data/
└── models/
```

## Features

*   **Data Ingestion:** Upload transaction data in CSV or JSON format.
*   **Multi-Agent Analysis:** Orchestrates specialized agents for a holistic financial review.
    *   **Data Agent:** Processes raw data, generates statistics, and detects anomalies.
    *   **Insight Agent:** Answers natural language queries about financial data.
    *   **Risk Agent:** Performs fraud detection and risk assessment.
*   **Interactive Dashboard:** Streamlit frontend for easy interaction and visualization.
*   **RESTful API:** FastAPI backend for programmatic access to FinAgent capabilities.

## Setup and Installation

To get the FinAgent project up and running, follow these steps:

1.  **Clone the Repository (if you haven't already):**

    ```bash
    git clone https://github.com/wtfashwin/FinAgent-Multi-Agent-Financial-Insight-Engine.git
    cd FinAgent-Multi-Agent-Financial-Insight-Engine
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**

    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**

    Install all the necessary Python packages using the provided `req.txt` file:

    ```bash
    pip install -r req.txt
    ```

## Running the Application

FinAgent consists of two main components: a FastAPI backend and a Streamlit frontend. Both need to be running to use the full application.

### 1. Start the FastAPI Backend

Open your first terminal, navigate to the `api` directory, and run the `main.py` file:

```bash
cd api
python main.py
```

You should see output similar to this, indicating the FastAPI server is running (typically on `http://127.0.0.1:8000`):

```
INFO:     Will watch for changes in these directories: ['c:\Users\Admin\Documents\Resume\Projects\FinAgent\api']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [PID] using StatReload
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     🚀 Starting FinAgent API...
INFO:     ✓ FinAgent initialized successfully
INFO:     Application startup complete.
```

### 2. Start the Streamlit Frontend

Open a **new terminal** (keep the FastAPI backend running in the first terminal). Navigate back to the root directory of your project (`FinAgent`) and run the Streamlit application:

```bash
cd ..
streamlit run streamlit_app.py
```

This will open the Streamlit application in your default web browser, usually at `http://localhost:8501`.

## Usage

Once both the backend and frontend are running:

1.  **Home Page:** Check the API health and get an overview of the project.
2.  **Upload Data:** Navigate to the "Upload Data" section to upload your transaction data (CSV or JSON).
3.  **Run Analysis:** Initiate a comprehensive multi-agent analysis on the uploaded data.
4.  **Get Insights:** Ask natural language queries to get specific financial insights.
5.  **Risk Summary:** View a summary of the risk assessment.
6.  **Statistics:** Get detailed statistics about your dataset.
7.  **Anomalies:** See any detected anomalies in your data.
8.  **Sample Data:** View a sample of the currently loaded data.

Enjoy using FinAgent to gain deeper insights into your financial transactions!
