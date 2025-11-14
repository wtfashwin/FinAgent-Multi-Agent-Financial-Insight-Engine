"""Unit tests for Risk Agent"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from agents.risk_agent import RiskAgent


class TestRiskAgent:
    """Test suite for RiskAgent class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'transaction_id': range(1000),
            'amount': np.random.exponential(100, 1000),
            'merchant_id': np.random.randint(1, 100, 1000),
            'hour': np.random.randint(0, 24, 1000),
            'day_of_week': np.random.randint(0, 7, 1000),
            'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })
    
    def test_train_fraud_model(self, sample_data):
        """Test supervised fraud model training"""
        agent = RiskAgent()
        metrics = agent.train_fraud_model(sample_data, target_col='is_fraud')
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert agent.fraud_model is not None
    
    def test_predict_fraud(self, sample_data):
        """Test fraud prediction functionality"""
        agent = RiskAgent()
        agent.train_fraud_model(sample_data, target_col='is_fraud')
        
        # Predict on the same data
        predictions_df = agent.predict_fraud(sample_data)
        
        assert isinstance(predictions_df, pd.DataFrame)
        assert 'fraud_probability' in predictions_df.columns
        assert 'fraud_prediction' in predictions_df.columns
        assert 'risk_level' in predictions_df.columns
        
        # Check that probabilities are in valid range
        assert (predictions_df['fraud_probability'] >= 0).all()
        assert (predictions_df['fraud_probability'] <= 1).all()
    
    def test_train_anomaly_model(self, sample_data):
        """Test unsupervised anomaly model training"""
        agent = RiskAgent()
        metrics = agent.train_anomaly_model(sample_data)
        
        assert isinstance(metrics, dict)
        assert 'anomalies_detected' in metrics
        assert 'anomaly_rate' in metrics
        assert 0 <= metrics['anomaly_rate'] <= 1
        assert agent.anomaly_model is not None
    
    def test_detect_anomalies(self, sample_data):
        """Test anomaly detection functionality"""
        agent = RiskAgent()
        agent.train_anomaly_model(sample_data)
        
        # Detect anomalies
        anomalies_df = agent.detect_anomalies(sample_data)
        
        assert isinstance(anomalies_df, pd.DataFrame)
        assert 'is_anomaly' in anomalies_df.columns
        assert 'anomaly_score' in anomalies_df.columns
    
    def test_get_risk_summary(self, sample_data):
        """Test risk summary generation"""
        agent = RiskAgent()
        agent.train_fraud_model(sample_data, target_col='is_fraud')
        predictions_df = agent.predict_fraud(sample_data)
        
        summary = agent.get_risk_summary(predictions_df)
        
        assert isinstance(summary, dict)
        assert 'high_risk_transactions' in summary
        assert 'medium_risk_transactions' in summary
        assert 'low_risk_transactions' in summary
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading functionality"""
        agent = RiskAgent()
        agent.train_fraud_model(sample_data, target_col='is_fraud')
        
        # Save models
        model_dir = tmp_path / "models"
        agent.save_models(str(model_dir))
        
        # Check that files were created
        assert (model_dir / "fraud_model.pkl").exists()
        assert (model_dir / "scaler.pkl").exists()
        
        # Load models
        new_agent = RiskAgent()
        new_agent.load_models(str(model_dir))
        
        assert new_agent.fraud_model is not None
        assert new_agent.scaler is not None
    
    def test_error_handling_no_model(self):
        """Test error handling when models are not trained"""
        agent = RiskAgent()
        
        with pytest.raises(ValueError, match="Fraud model not trained"):
            agent.predict_fraud(pd.DataFrame())
        
        with pytest.raises(ValueError, match="Anomaly model not trained"):
            agent.detect_anomalies(pd.DataFrame())