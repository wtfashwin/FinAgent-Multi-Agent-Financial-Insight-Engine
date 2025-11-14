"""Integration tests for FinAgent Orchestrator"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from orchestrator import FinAgentOrchestrator, AgentState


class TestFinAgentOrchestrator:
    """Test suite for FinAgentOrchestrator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'transaction_id': range(100),
            'customer_id': np.random.randint(1, 20, 100),
            'amount': np.random.exponential(50, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        })
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = FinAgentOrchestrator()
        
        assert orchestrator.data_agent is not None
        assert orchestrator.insight_agent is not None
        assert orchestrator.risk_agent is not None
        assert orchestrator.workflow is not None
    
    def test_build_workflow(self):
        """Test workflow building"""
        orchestrator = FinAgentOrchestrator()
        workflow = orchestrator._build_workflow()
        
        assert workflow is not None
        # Check that all expected nodes are present
        assert "data_processing" in workflow.nodes
        assert "risk_assessment" in workflow.nodes
        assert "ingest_rag_data" in workflow.nodes
        assert "insight_generation" in workflow.nodes
        assert "summarization" in workflow.nodes
    
    def test_data_processing_node(self, sample_data):
        """Test data processing node"""
        orchestrator = FinAgentOrchestrator()
        initial_state = {
            'data': sample_data,
            'messages': []
        }
        
        result_state = orchestrator._data_processing_node(initial_state)
        
        assert 'processed_data' in result_state
        assert 'anomalies' in result_state
        assert isinstance(result_state['processed_data'], pd.DataFrame)
        assert isinstance(result_state['anomalies'], list)
        assert len(result_state['messages']) > 0
    
    @patch('agents.insight_agent.InsightAgent.get_summary')
    @patch('agents.insight_agent.InsightAgent.analyze_trends')
    def test_insight_generation_node(self, mock_analyze_trends, mock_get_summary, sample_data):
        """Test insight generation node"""
        # Mock the insight agent methods
        mock_get_summary.return_value = "Test summary"
        mock_analyze_trends.return_value = {"test": "trends"}
        
        orchestrator = FinAgentOrchestrator()
        state = {
            'processed_data': sample_data,
            'rag_ingested': True,
            'messages': []
        }
        
        result_state = orchestrator._insight_generation_node(state)
        
        assert 'insights' in result_state
        assert isinstance(result_state['insights'], dict)
        assert len(result_state['messages']) > 0
    
    def test_risk_assessment_node(self, sample_data):
        """Test risk assessment node"""
        orchestrator = FinAgentOrchestrator()
        state = {
            'processed_data': sample_data,
            'messages': []
        }
        
        result_state = orchestrator._risk_assessment_node(state)
        
        assert 'fraud_predictions' in result_state
        assert 'risk_assessment' in result_state
        assert isinstance(result_state['fraud_predictions'], pd.DataFrame)
        assert isinstance(result_state['risk_assessment'], dict)
        assert len(result_state['messages']) > 0
    
    def test_summarization_node(self, sample_data):
        """Test summarization node"""
        orchestrator = FinAgentOrchestrator()
        state = {
            'insights': {
                'summary': 'Test insights summary'
            },
            'risk_assessment': {
                'summary': {
                    'high_risk_transactions': 5,
                    'medium_risk_transactions': 10,
                    'low_risk_transactions': 85,
                    'total_amount_at_risk': 5000.0
                }
            },
            'anomalies': [{'test': 'anomaly1'}, {'test': 'anomaly2'}],
            'messages': []
        }
        
        result_state = orchestrator._summarization_node(state)
        
        assert 'summary' in result_state
        assert isinstance(result_state['summary'], str)
        assert len(result_state['summary']) > 0
        assert len(result_state['messages']) > 0
    
    def test_compile_workflow(self):
        """Test workflow compilation"""
        orchestrator = FinAgentOrchestrator()
        app = orchestrator.compile()
        
        assert app is not None
        assert orchestrator.app is not None
    
    @patch('agents.data_agent.DataAgent.load_data')
    @patch('agents.data_agent.DataAgent.clean_data')
    @patch('agents.data_agent.DataAgent.extract_features')
    @patch('agents.data_agent.DataAgent.detect_anomalies')
    @patch('agents.data_agent.DataAgent.get_statistics')
    @patch('agents.risk_agent.RiskAgent.train_fraud_model')
    @patch('agents.risk_agent.RiskAgent.predict_fraud')
    @patch('agents.risk_agent.RiskAgent.get_risk_summary')
    @patch('agents.insight_agent.InsightAgent.get_summary')
    @patch('agents.insight_agent.InsightAgent.analyze_trends')
    def test_run_workflow(self, mock_analyze_trends, mock_get_summary, 
                         mock_get_risk_summary, mock_predict_fraud, 
                         mock_train_fraud_model, mock_get_statistics, 
                         mock_detect_anomalies, mock_extract_features, 
                         mock_clean_data, mock_load_data, sample_data):
        """Test complete workflow execution"""
        # Set up mocks
        mock_load_data.return_value = sample_data
        mock_clean_data.return_value = sample_data
        mock_extract_features.return_value = sample_data
        mock_detect_anomalies.return_value = []
        mock_get_statistics.return_value = {}
        mock_train_fraud_model.return_value = {}
        mock_predict_fraud.return_value = sample_data
        mock_get_risk_summary.return_value = {}
        mock_get_summary.return_value = "Test summary"
        mock_analyze_trends.return_value = {}
        
        orchestrator = FinAgentOrchestrator()
        orchestrator.compile()
        
        # Run the workflow
        result = orchestrator.run(data=sample_data)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'messages' in result
        assert len(result['messages']) > 0