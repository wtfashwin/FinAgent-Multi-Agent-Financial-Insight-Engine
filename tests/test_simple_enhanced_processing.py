"""
Simple test for enhanced data processing features
"""
import pandas as pd
import numpy as np
import tempfile
import os

from agents.data_agent import DataAgent


def test_enhanced_data_processing():
    """Test the enhanced data processing features"""
    # Create sample data
    sample_data = pd.DataFrame({
        'transaction_id': range(50),
        'customer_id': np.random.randint(1, 10, 50),
        'amount': np.random.exponential(50, 50),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target'], 50),
        'is_fraud': np.random.choice([0, 1], 50, p=[0.9, 0.1])
    })
    
    # Add some missing values
    sample_data.loc[5:7, 'amount'] = np.nan
    sample_data.loc[10:12, 'merchant'] = np.nan
    
    # Add duplicates
    sample_data = pd.concat([sample_data, sample_data.iloc[0:2]], ignore_index=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV loading
        csv_path = os.path.join(tmpdir, 'test_data.csv')
        sample_data.to_csv(csv_path, index=False)
        
        # Initialize agent
        agent = DataAgent()
        
        # Test loading data
        df = agent.load_data(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Test data quality assessment
        quality_report = agent.assess_data_quality()
        print(f"Data quality report metrics: {list(quality_report.metrics.keys())}")
        
        # Test cleaning suggestions
        suggestions = agent.suggest_cleaning_operations()
        print(f"Generated {len(suggestions)} cleaning suggestions")
        
        # Test data cleaning
        cleaned_df = agent.clean_data()
        print(f"Cleaned data shape: {cleaned_df.shape}")
        
        # Test anomaly detection
        anomalies = agent.detect_anomalies()
        print(f"Detected {len(anomalies)} anomalies")
        
        print("All enhanced data processing tests passed!")


if __name__ == "__main__":
    test_enhanced_data_processing()