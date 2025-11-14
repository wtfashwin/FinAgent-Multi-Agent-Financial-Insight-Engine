"""Unit tests for Data Agent"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from agents.data_agent import DataAgent


class TestDataAgent:
    """Test suite for DataAgent class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data for testing"""
        return pd.DataFrame({
            'transaction_id': range(100),
            'customer_id': np.random.randint(1, 20, 100),
            'amount': np.random.exponential(50, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            temp_file_name = tmp.name
            sample_data.to_csv(temp_file_name, index=False)
            
        yield temp_file_name        
        os.unlink(temp_file_name)    
        
    def test_load_data(self, temp_csv_file, sample_data):
        """Test data loading functionality"""
        agent = DataAgent()
        df = agent.load_data(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
        assert list(df.columns) == list(sample_data.columns)
    
    def test_clean_data(self, sample_data):
        """Test data cleaning functionality"""
        # Add some missing values and duplicates
        sample_data.loc[0, 'amount'] = np.nan
        sample_data = pd.concat([sample_data, sample_data.iloc[:5]])
        
        agent = DataAgent()
        agent.df = sample_data
        cleaned_df = agent.clean_data()
        
        assert cleaned_df.isnull().sum().sum() == 0
        assert len(cleaned_df) == 100
    
    def test_detect_anomalies(self, sample_data):
        """Test anomaly detection functionality"""
        agent = DataAgent()
        agent.df = sample_data
        agent.clean_data()
        anomalies = agent.detect_anomalies()
        
        assert isinstance(anomalies, list)
        assert agent.processed_df is not None
        if agent.processed_df is not None:
            assert 'is_anomaly' in agent.processed_df.columns
    
    def test_extract_features(self, sample_data):
        """Test feature extraction functionality"""
        agent = DataAgent()
        agent.df = sample_data
        agent.clean_data()
        features_df = agent.extract_features()
        
        assert isinstance(features_df, pd.DataFrame)
        # Should have more columns after feature extraction
        assert len(features_df.columns) >= len(sample_data.columns)
    
    def test_get_statistics(self, sample_data):
        """Test statistics generation functionality"""
        agent = DataAgent()
        agent.df = sample_data
        agent.clean_data()
        stats = agent.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_transactions' in stats
        assert 'total_anomalies' in stats
        assert stats['total_transactions'] == len(sample_data)
    
    def test_error_handling_no_data(self):
        """Test error handling when no data is loaded"""
        agent = DataAgent()
        
        with pytest.raises(ValueError, match="No data loaded"):
            agent.clean_data()
        
        with pytest.raises(ValueError, match="No processed data"):
            agent.detect_anomalies()