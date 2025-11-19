"""Comprehensive test suite for FinAgent Multi-Agent System"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import project modules
from api.main import app
from orchestrator import FinAgentOrchestrator
from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent
from agents.insight_agent import InsightAgent

class TestFinAgentComprehensive:
    """Comprehensive test suite for FinAgent components"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 100),
            'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        })
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        df = pd.DataFrame({
            'transaction_id': range(10),
            'amount': [100, 250, 50, 1000, 75, 300, 150, 800, 200, 500],
            'merchant': ['Amazon', 'Walmart', 'Starbucks', 'Apple', 'Target', 
                        'BestBuy', 'McDonalds', 'Samsung', 'Nike', 'Costco'],
            'category': ['Online', 'Retail', 'Food', 'Electronics', 'Retail', 
                        'Electronics', 'Food', 'Electronics', 'Retail', 'Retail']
        })
        return df.to_csv(index=False)
    
    # API Tests
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "orchestrator_ready" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_upload_file(self, client, sample_csv_data):
        """Test file upload endpoint"""
        files = {"file": ("test_data.csv", sample_csv_data, "text/csv")}
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "statistics" in data
        assert data["status"] == "success"
    
    def test_error_handling(self, client):
        """Test error handling for endpoints that require data"""
        # Test analyze endpoint without data
        response = client.post("/api/analyze")
        assert response.status_code == 400
        
        # Test statistics endpoint without data
        response = client.get("/api/statistics")
        assert response.status_code == 400
    
    # Orchestrator Tests
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = FinAgentOrchestrator()
        
        assert orchestrator.data_agent is not None
        assert orchestrator.insight_agent is not None
        assert orchestrator.risk_agent is not None
        assert orchestrator.workflow is not None
    
    def test_compile_workflow(self):
        """Test workflow compilation"""
        orchestrator = FinAgentOrchestrator()
        app = orchestrator.compile()
        
        assert app is not None
        assert orchestrator.app is not None
    
    # Agent Tests
    def test_data_agent_functionality(self, sample_data):
        """Test data agent functionality"""
        data_agent = DataAgent()
        data_agent.df = sample_data
        processed_data = data_agent.clean_data()
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
    
    def test_risk_agent_initialization(self):
        """Test risk agent initialization"""
        risk_agent = RiskAgent()
        assert risk_agent is not None
        assert risk_agent.fraud_model is None  # Not trained yet
    
    def test_insight_agent_initialization(self):
        """Test insight agent initialization"""
        insight_agent = InsightAgent()
        assert insight_agent is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])