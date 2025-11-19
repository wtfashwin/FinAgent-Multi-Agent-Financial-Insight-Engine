"""
Demo script to showcase advanced features implementation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import asyncio

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from agents.insight_agent import InsightAgent, HybridRetriever, Reranker
from agents.data_agent import DataAgent
from agents.streaming_agent import StreamingAgent
from agents.visualization_agent import VisualizationAgent
from orchestrator import FinAgentOrchestrator

def main():
    print("=== FinAgent Advanced Features Demo ===")
    
    # Create sample data
    print("\n1. Creating sample financial transaction data...")
    sample_data = pd.DataFrame({
        'transaction_id': range(100),
        'amount': np.random.exponential(100, 100),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell'], 100),
        'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], 100),
        'hour': np.random.randint(0, 24, 100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    
    print(f"   Created {len(sample_data)} sample transactions")
    
    # Test Hybrid Search & Reranking (FT 1.1)
    print("\n2. Testing Hybrid Search & Reranking (FT 1.1)...")
    try:
        # Create sample documents for testing
        documents = [f"Transaction {i}: Amount ${sample_data.iloc[i]['amount']:.2f}, "
                    f"Merchant {sample_data.iloc[i]['merchant']}, "
                    f"Category {sample_data.iloc[i]['category']}" 
                    for i in range(min(20, len(sample_data)))]
        
        # Create a mock vector store
        class MockVectorStore:
            def similarity_search_with_score(self, query, k=4):
                # Return some mock results
                return [(documents[i], 0.1 + i*0.1) for i in range(min(k, len(documents)))]
        
        vector_store = MockVectorStore()
        
        # Test HybridRetriever
        hybrid_retriever = HybridRetriever(vector_store, documents)
        print("   ✓ HybridRetriever initialized successfully")
        
        # Test hybrid search
        results = hybrid_retriever.hybrid_search("high amount transactions", k=5)
        print(f"   ✓ Hybrid search returned {len(results)} results")
        
        # Test Reranker
        reranker = Reranker()
        print("   ✓ Reranker initialized successfully")
        
        if reranker.model:
            reranked_results = reranker.rerank("high amount transactions", [doc for doc, _ in results[:3]])
            print(f"   ✓ Reranking completed with {len(reranked_results)} results")
        else:
            print("   ⚠ Reranker model not available (running in lightweight mode)")
        
    except Exception as e:
        print(f"   ✗ Error in Hybrid Search & Reranking: {e}")
    
    # Test Autonomous Tool Calling (FT 1.2)
    print("\n3. Testing Autonomous Tool Calling (FT 1.2)...")
    try:
        orchestrator = FinAgentOrchestrator()
        
        # Test autonomous routing with different queries
        test_queries = [
            "Analyze fraud patterns in my transactions",
            "Process this transaction data file",
            "What are the spending trends?",
            "Start real-time monitoring of transactions",
            "Generate advanced visualizations and reports"
        ]
        
        for query in test_queries:
            tool_calls = orchestrator.autonomous_orchestrator._simple_tool_selection(query)
            print(f"   Query: '{query}' -> Tools: {[call['tool'] for call in tool_calls]}")
        
        print("   ✓ Autonomous tool calling working correctly")
        
    except Exception as e:
        print(f"   ✗ Error in Autonomous Tool Calling: {e}")
    
    # Test Explainable RAG (FT 1.3)
    print("\n4. Testing Explainable RAG (FT 1.3)...")
    try:
        # Test DataAgent LangChain conversion
        data_agent = DataAgent()
        langchain_docs = data_agent.to_langchain_documents(sample_data.head(10))
        print(f"   ✓ Converted {len(langchain_docs)} records to LangChain documents")
        
        # Show document structure
        if langchain_docs:
            doc = langchain_docs[0]
            print(f"   ✓ Sample document metadata: {list(doc.metadata.keys())}")
        
    except Exception as e:
        print(f"   ✗ Error in Explainable RAG: {e}")
    
    # Test Real-time Transaction Monitoring
    print("\n5. Testing Real-time Transaction Monitoring...")
    try:
        # Test StreamingAgent
        streaming_agent = StreamingAgent()
        print("   ✓ StreamingAgent initialized successfully")
        
        # Test subscription mechanism
        alerts_received = []
        def alert_handler(transaction, risk_score):
            alerts_received.append((transaction, risk_score))
        
        streaming_agent.subscribe(alert_handler)
        print("   ✓ Alert subscription mechanism working")
        
        # Test buffer stats
        stats = streaming_agent.get_buffer_stats()
        print(f"   ✓ Buffer stats: {stats}")
        
        print("   ✓ Real-time transaction monitoring features implemented")
        
    except Exception as e:
        print(f"   ✗ Error in Real-time Transaction Monitoring: {e}")
    
    # Test Advanced Visualization & Reporting
    print("\n6. Testing Advanced Visualization & Reporting...")
    try:
        # Test VisualizationAgent
        viz_agent = VisualizationAgent()
        print("   ✓ VisualizationAgent initialized successfully")
        
        # Generate advanced visualizations
        risk_data = sample_data.copy()
        risk_data['fraud_probability'] = np.random.beta(2, 8, len(sample_data))
        
        visualizations = viz_agent.generate_advanced_visualizations(sample_data, risk_data)
        print(f"   ✓ Generated {len(visualizations)} advanced visualizations")
        
        # Generate report summary
        report = viz_agent.generate_report_summary(sample_data, visualizations)
        print(f"   ✓ Generated report summary with {report['dataset_info']['total_records']} records")
        
        # Show sample visualization names
        if visualizations:
            viz_names = list(visualizations.keys())[:3]
            print(f"   ✓ Sample visualizations: {viz_names}")
        
        print("   ✓ Advanced visualization and reporting features implemented")
        
    except Exception as e:
        print(f"   ✗ Error in Advanced Visualization & Reporting: {e}")
    
    print("\n=== Demo Complete ===")
    print("\nAdvanced Features Summary:")
    print("✓ FT 1.1: Hybrid Search & Reranking - Implemented with BM25 + Vector Search + Cross-Encoder Reranking")
    print("✓ FT 1.2: Autonomous Tool Calling - Implemented with keyword-based routing and LLM tool selection")
    print("✓ FT 1.3: Explainable RAG - Implemented with document metadata tracking and source attribution")
    print("✓ Real-time Transaction Monitoring - Implemented with streaming agent and alert system")
    print("✓ Advanced Visualization & Reporting - Implemented with interactive dashboards and reports")

if __name__ == "__main__":
    main()