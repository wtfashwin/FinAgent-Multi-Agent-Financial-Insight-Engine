"""
Test cases for advanced features implementation
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.insight_agent import InsightAgent, HybridRetriever, Reranker
from agents.data_agent import DataAgent
from orchestrator import FinAgentOrchestrator


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced features implementation"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'transaction_id': range(50),
            'amount': np.random.exponential(100, 50),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 50),
            'category': np.random.choice(['Online', 'Retail', 'Food'], 50),
            'is_fraud': np.random.choice([0, 1], 50, p=[0.9, 0.1])
        })
        
        self.data_agent = DataAgent()
        self.insight_agent = InsightAgent()
    
    def test_hybrid_retriever_initialization(self):
        """Test HybridRetriever initialization"""
        # Create sample documents
        documents = [f"Document {i}: This is a test document about financial transactions" for i in range(10)]
        
        # Create a mock vector store (we won't actually use it for this test)
        class MockVectorStore:
            def similarity_search_with_score(self, query, k=4):
                return [(doc, 0.5) for doc in documents[:k]]
        
        vector_store = MockVectorStore()
        
        # Test HybridRetriever initialization
        hybrid_retriever = HybridRetriever(vector_store, documents)
        self.assertIsNotNone(hybrid_retriever)
        self.assertEqual(hybrid_retriever.documents, documents)
    
    def test_reranker_initialization(self):
        """Test Reranker initialization"""
        # Test Reranker initialization
        reranker = Reranker()
        self.assertIsNotNone(reranker)
    
    def test_autonomous_tool_selection(self):
        """Test autonomous tool selection functionality"""
        # Initialize orchestrator
        orchestrator = FinAgentOrchestrator()
        
        # Test simple tool selection
        tool_calls = orchestrator.autonomous_orchestrator._simple_tool_selection("I want to analyze fraud in my transactions")
        
        # Should select risk assessment tool
        self.assertGreater(len(tool_calls), 0)
        tool_names = [call['tool'] for call in tool_calls]
        self.assertIn('assess_risk_tool', tool_names)
    
    def test_data_agent_langchain_conversion(self):
        """Test DataAgent to_langchain_documents conversion"""
        # Test conversion to LangChain documents
        documents = self.data_agent.to_langchain_documents(self.sample_data)
        
        # Should create documents
        self.assertGreater(len(documents), 0)
        self.assertTrue(hasattr(documents[0], 'page_content'))
        self.assertTrue(hasattr(documents[0], 'metadata'))


if __name__ == '__main__':
    unittest.main()