import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

# --- Import Google GenAI components ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="FinAgent v0.3 (Elite Fix)",
    page_icon="🤖",
    layout="wide"
)

# --- Define Gemini Model Names ---
GEMINI_LLM_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# --- Session State Initialization (CRITICAL for flow control) ---
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'df_hash' not in st.session_state:
    st.session_state.df_hash = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for API Key ---
st.sidebar.header("🔑 Configuration")
google_api_key = st.sidebar.text_input(
    "Enter your Google API Key", 
    type="password"
)

# --- LLM & Embedding Model Setup (Cached Resource) ---
@st.cache_resource
def get_llm(api_key):
    """Initializes the ChatGoogleGenerativeAI LLM (Model only runs once per key change)."""
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL, 
            google_api_key=api_key
        )
    except Exception as e:
        # Do not use st.error here, as it breaks the cache. Let the caller handle display.
        return f"Error initializing LLM: {e}"

@st.cache_resource
def get_embeddings(api_key):
    """Initializes the GoogleGenerativeAIEmbeddings (Model only runs once per key change)."""
    if not api_key:
        return None
    try:
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL, 
            google_api_key=api_key
        )
    except Exception as e:
        # Do not use st.error here, as it breaks the cache. Let the caller handle display.
        return f"Error initializing Embeddings: {e}"

# --- "Data Agent": Data Loading & Cleansing (Unchanged) ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            scaler = StandardScaler()
            df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
            df_processed = df.drop(['Time', 'Amount'], axis=1)
            
            # Elite Tip: Generate a hash of the processed data to use as a manual cache key
            data_hash = hash(df_processed.to_json())
            return df_processed, df, data_hash
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None
    return None, None, None

# --- "Risk Agent": Anomaly & Fraud Modeling (Unchanged) ---
@st.cache_resource
def train_risk_model(df):
    features = df.drop(['Class'], axis=1, errors='ignore')
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(features)
    return model

# Function remains outside cache as it relies on mutable state (the model and the input df)
def run_risk_analysis(model, df):
    features = df.drop(['Class'], axis=1, errors='ignore')
    df['anomaly_score'] = model.decision_function(features)
    df['is_anomaly'] = model.predict(features)
    df['is_anomaly'] = df['is_anomaly'] == -1
    return df

# --- "Insight Agent": RAG Pipeline - Setup function (NOT CACHED with st.cache_data/resource) ---
def get_anomaly_summary(df):
    """Creates a text summary of the anomalous transactions."""
    anomalies = df[df['is_anomaly']]
    if anomalies.empty:
        return "No anomalies detected in this batch."
        
    summary = f"""
    Anomaly Detection Report:
    - Total transactions processed: {len(df)}
    - Total anomalies detected: {len(anomalies)}
    
    Key observations from anomalous transactions:
    - Average 'Scaled_Amount' for anomalies: {anomalies['Scaled_Amount'].mean():.2f}
    - Average 'Scaled_Amount' for normal: {df[~df['is_anomaly']]['Scaled_Amount'].mean():.2f}
    
    Sample Anomalous Transaction Details:
    """
    for i, row in anomalies.head(3).iterrows():
        summary += f"- Sample {i}: Scaled_Amount: {row['Scaled_Amount']:.2f}, Anomaly Score: {row['anomaly_score']:.2f}\n"
    
    return summary

def setup_rag_pipeline(summary_text, api_key):
    """
    Creates a RAG pipeline using Gemini and Chroma. This function is now only called
    explicitly when needed, bypassing Streamlit's unpredictable caching behavior.
    """
    embeddings = get_embeddings(api_key)
    llm = get_llm(api_key)
    
    if isinstance(embeddings, str): # Check for error string returned by get_embeddings
        raise RuntimeError(embeddings)
    if isinstance(llm, str): # Check for error string returned by get_llm
        raise RuntimeError(llm)
    if not embeddings or not llm:
        raise RuntimeError("API Key or Model initialization failed.")

    docs = [summary_text]
    
    # The expensive embedding API call happens here during Chroma.from_texts
    try:
        vectorstore = Chroma.from_texts(docs, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        # Raise a specific error for the UI to catch and display
        raise RuntimeError(f"Error setting up ChromaDB: {e}")

    template = """
    You are 'FinAgent', a financial analysis assistant.
    You have just run an anomaly detection model on a batch of transactions.
    Use the following CONTEXT (your analysis report) to answer the user's question.
    If the context doesn't have the answer, say so.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main Streamlit App UI ---
st.title("🤖 FinAgent v0.3 - Elite Quota Management")
st.write("This version uses explicit Streamlit session state to guarantee the expensive embedding step runs exactly once per data load.")

# --- Check for API Key before proceeding ---
if not google_api_key:
    st.info("Please enter your Google API Key in the sidebar to start.")
    st.session_state.rag_chain = None # Reset state if key is removed
    st.stop()

st.success("Google API Key loaded.")

# 1. Setup & Data Upload
st.header("1. Data Ingestion (Data Agent)")
st.write("Download the 'Credit Card Fraud Detection' dataset from Kaggle.")
st.markdown("🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file:
    # --- Run Agents on Data ---
    with st.spinner("Processing Data..."):
        df_processed, df_original, data_hash = load_data(uploaded_file)
    
    if df_processed is not None:
        st.success("Data loaded and preprocessed.")
        
        # 2. Risk Agent
        st.header("2. Risk Analysis (Risk Agent)")
        with st.spinner("Training Anomaly Model... (This is fast)"):
            risk_model = train_risk_model(df_processed)
        
        with st.spinner("Running Risk Analysis..."):
            # We run risk analysis every time to update the display easily
            df_with_anomalies = run_risk_analysis(risk_model, df_processed.copy())
            
            df_display = df_original.copy()
            df_display['is_anomaly'] = df_with_anomalies['is_anomaly']
            df_display['anomaly_score'] = df_with_anomalies['anomaly_score']
            
            anomalies_df = df_display[df_display['is_anomaly']].sort_values(by='anomaly_score')
            
            st.subheader("Fraud & Anomaly Alerts")
            st.write(f"Found **{len(anomalies_df)}** potential anomalies/fraud events.")
            st.dataframe(anomalies_df.head(10))

            # 3. Insight Agent (RAG) Setup - **Elite State Management**
            st.header("3. Conversational Analysis (Insight Agent)")
            
            # This is the CRITICAL change: ONLY run the expensive setup if the data has changed.
            if st.session_state.rag_chain is None or st.session_state.df_hash != data_hash:
                
                # Reset chat history when new data is loaded
                st.session_state.messages = []
                st.session_state.df_hash = data_hash
                st.session_state.rag_chain = None # Ensure it is reset before trying again

                try:
                    with st.spinner("Generating Insights & Setting up RAG with Gemini (This is the only time the embedding API is called!)..."):
                        summary = get_anomaly_summary(df_with_anomalies)
                        st.session_state.rag_chain = setup_rag_pipeline(summary, google_api_key)
                    st.success("RAG pipeline successfully set up. Ready for chat.")
                except RuntimeError as e:
                    st.error(f"Failed to set up RAG pipeline (Quota Management Error): {e}")
                    # Keep st.session_state.rag_chain as None to disable chat
            
            # Chat UI logic relies on session state
            if st.session_state.rag_chain:
                st.subheader("Chat with your Analysis")
                st.write("The Gemini LLM has read the anomaly report. Ask it questions about the findings.")

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("E.g., How many anomalies were found?"):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.spinner("FinAgent (Gemini) is thinking..."):
                        # This LLM call is an API call, but it is typically not the quota bottleneck.
                        response = st.session_state.rag_chain.invoke(prompt)
                    
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                 st.warning("RAG Chat is unavailable. Check error messages above and ensure your API key is valid.")

else:
    # Reset state if no file is uploaded
    st.session_state.rag_chain = None
    st.session_state.df_hash = None
    st.info("Awaiting CSV file upload to begin analysis.")