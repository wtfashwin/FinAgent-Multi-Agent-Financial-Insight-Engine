import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib 

try:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None
    FastEmbedEmbeddings = None
    Chroma = None
    ChatPromptTemplate = None
    RunnablePassthrough = None
    StrOutputParser = None
    st.error("Missing RAG dependencies. Install with: `pip install langchain-groq fastembed python-dotenv`")

st.set_page_config(
    page_title="FinAgent",
    page_icon="🤖",
    layout="wide"
)

GROQ_LLM_MODEL = "moonshotai/kimi-k2-instruct-0905" 
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'df_hash' not in st.session_state:
    st.session_state.df_hash = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("🔑 Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key", 
    type="password"
)

@st.cache_resource
def get_llm(api_key):
    if not api_key:
        return None
    if not LANGCHAIN_AVAILABLE or ChatGroq is None:
        return f"Error: Langchain Groq components not available."
    try:
        return ChatGroq(
            model=GROQ_LLM_MODEL, 
            groq_api_key=api_key
        )
    except Exception as e:
        return f"Error initializing LLM: {e}"

@st.cache_resource
def get_embeddings(): 
    if not LANGCHAIN_AVAILABLE or FastEmbedEmbeddings is None:
        return f"Error: FastEmbed components not available."
    try:
        return FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    except Exception as e:
        return f"Error initializing Embeddings: {e}"

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None, None, None
    try:
        df = pd.read_csv(uploaded_file)
        amount_column = None
        possible_amount_columns = ['Amount', 'amount', 'transaction_amount', 'txn_amount', 'value']
        for col in possible_amount_columns:
            if col in df.columns:
                amount_column = col
                break       

        if amount_column is None:
            available_columns = list(df.columns)
            raise ValueError(f"No amount column found. Available columns: {available_columns}")
        df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        missing_count = df[amount_column].isnull().sum()
        if missing_count > 0:
            median_val = df[amount_column].median()
            df[amount_column].fillna(median_val, inplace=True)
            st.warning(f"Imputed {missing_count} missing/invalid values in '{amount_column}' with the median ({median_val:.2f}).")

        scaler = StandardScaler()
        amount_values = df[amount_column].values.reshape(-1, 1)
        df['Scaled_Amount'] = scaler.fit_transform(amount_values).flatten()        

        time_column = 'Time'
        possible_time_columns = ['Time', 'time', 'timestamp', 'date']
        for col in possible_time_columns:
            if col in df.columns:
                time_column = col
                break        

        columns_to_drop = []
        if time_column in df.columns:
            columns_to_drop.append(time_column)
        if amount_column in df.columns:
            columns_to_drop.append(amount_column)            
        df_processed = df.drop(columns_to_drop, axis=1, errors='ignore').select_dtypes(include=np.number).fillna(0)
        data_bytes = df_processed.to_csv(index=False).encode('utf-8')
        data_hash = hashlib.md5(data_bytes).hexdigest()
        return df_processed, df, data_hash

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your CSV is clean and contains a clear transaction amount column.")
        return None, None, None

@st.cache_resource
def train_risk_model(df):
    if 'Class' in df.columns:
        features = df.drop(['Class'], axis=1, errors='ignore')
    else:
        features = df.copy()
        st.warning("No 'Class' column found in data. Using unsupervised anomaly detection only (Isolation Forest).")    
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(features)
    return model

def run_risk_analysis(model, df):
    df_copy = df.copy()
    features = df_copy.drop(['Class'], axis=1, errors='ignore')
    df_copy['anomaly_score'] = model.decision_function(features)
    df_copy['is_anomaly'] = model.predict(features)
    df_copy['is_anomaly'] = df_copy['is_anomaly'] == -1
    return df_copy

def get_anomaly_summary(df):
    anomalies = df[df['is_anomaly']].sort_values(by='anomaly_score', ascending=True)
    if anomalies.empty:
        return "No anomalies detected in this batch based on the Isolation Forest model (contamination=0.01)."        
    amount_col = [col for col in df.columns if col not in ['Scaled_Amount', 'anomaly_score', 'is_anomaly', 'Class'] and df[col].dtype != object][0] if len(df.columns) > 5 else 'Amount (Check Original DF)'

    summary = f"""
    Anomaly Detection Report (Isolation Forest, Contamination=1%):
    - Total transactions processed: {len(df)}
    - Total anomalies detected: {len(anomalies)}
    - The lowest anomaly score (most anomalous) was: {anomalies['anomaly_score'].min():.4f}
    - The Isolation Forest model used scaled features.
    Key observations from the detected anomalies:
    - Average Scaled_Amount for anomalies: {anomalies['Scaled_Amount'].mean():.2f}
    - Average Scaled_Amount for normal transactions: {df[~df['is_anomaly']]['Scaled_Amount'].mean():.2f}
    Sample Anomalous Transaction Details (Top 3 most isolated events):
    """
    for i, row in anomalies.head(3).iterrows():
        original_amount_val = df_original.loc[i, amount_col] if 'df_original' in globals() else 'N/A'
        summary += f"- Index {i}: Original '{amount_col}': {original_amount_val}, Scaled_Amount: {row['Scaled_Amount']:.2f}, Anomaly Score (Lower is worse): {row['anomaly_score']:.4f}\n"
    return summary

def setup_rag_pipeline(summary_text, api_key):
    embeddings_result = get_embeddings()
    llm_result = get_llm(api_key)  
    if isinstance(embeddings_result, str):
        raise RuntimeError(embeddings_result)
    if isinstance(llm_result, str):
        raise RuntimeError(llm_result)   
    embeddings = embeddings_result
    llm = llm_result
    docs = [summary_text]  
    try:
        with st.spinner(f"Embedding text locally with {LOCAL_EMBEDDING_MODEL}..."):
            vectorstore = Chroma.from_texts(docs, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        raise RuntimeError(f"Error setting up ChromaDB/FastEmbed: {e}")
    template = """
    You are 'FinAgent', a financial analysis assistant specializing in anomaly detection.
    You have analyzed a batch of transactions using an Isolation Forest model.
    Use the following CONTEXT (the anomaly report) to answer the user's question concisely.
    If the context does not contain the answer, state that clearly ("I cannot answer that based on the provided report.").
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

st.title("FinAgent 🤖")

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to start.")
    st.session_state.rag_chain = None 
    st.stop()
st.success(f"Groq API Key loaded. Conversational model: {GROQ_LLM_MODEL}.")
st.header("1. Data Ingestion (Data Agent)")
st.write("Upload a CSV dataset (e.g., Credit Card Fraud Detection).")
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file:
    with st.spinner("Processing Data and Feature Engineering..."):
        df_processed, df_original, data_hash = load_data(uploaded_file)
    if df_processed is not None:
        st.success("Data loaded and robustly preprocessed.")      
        st.header("2. Risk Analysis (Risk Agent)")
        with st.spinner("Training Isolation Forest Model..."):
            risk_model = train_risk_model(df_processed)       
        with st.spinner("Running Risk Analysis..."):
            df_with_anomalies = run_risk_analysis(risk_model, df_processed)
            df_display = df_original.copy()
            df_display['is_anomaly'] = df_with_anomalies['is_anomaly']
            df_display['anomaly_score'] = df_with_anomalies['anomaly_score']
            anomalies_df = df_display[df_display['is_anomaly']].sort_values(by='anomaly_score', ascending=True) 
            st.subheader("Fraud & Anomaly Alerts")
            st.write(f"Found **{len(anomalies_df)}** potential anomalies/fraud events.")
            st.dataframe(anomalies_df.head(10))
            st.header("3. Conversational Analysis (Insight Agent)")
            if st.session_state.rag_chain is None or st.session_state.df_hash != data_hash:
                st.session_state.messages = []
                st.session_state.df_hash = data_hash
                st.session_state.rag_chain = None 
                try:
                    with st.spinner("Generating Insight Report & Setting up RAG..."):
                        summary = get_anomaly_summary(df_with_anomalies)
                        st.session_state.rag_chain = setup_rag_pipeline(summary, groq_api_key) 
                    st.success("RAG pipeline successfully set up. Ready for chat.")
                except RuntimeError as e:
                    st.error(f"Failed to set up RAG pipeline: {e}")
            if st.session_state.rag_chain:
                st.subheader("Chat with your Analysis")
                st.write(f"The **{GROQ_LLM_MODEL}** LLM is ready to answer questions about the generated report.")
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("E.g., How many anomalies were found? What is the average scaled amount for them?"):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.spinner(f"FinAgent ({GROQ_LLM_MODEL}) is thinking..."):
                        try:
                            response = st.session_state.rag_chain.invoke(prompt)
                        except Exception as e:
                            response = f"**Error:** An API issue occurred during the Groq LLM call. Please check your Groq API key or rate limits. Details: {e}"
                            st.error(response)                        
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("RAG Chat is unavailable. Check error messages above and ensure RAG components are installed.")
    else:
        st.session_state.rag_chain = None
        st.session_state.df_hash = None
else:
    st.session_state.rag_chain = None
    st.session_state.df_hash = None
    st.info("Awaiting CSV file upload to begin analysis.")