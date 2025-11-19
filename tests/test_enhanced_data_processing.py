"""
Test cases for enhanced data processing features
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from agents.data_agent import DataAgent, DataQualityReport


class TestEnhancedDataProcessing(unittest.TestCase):
    """Test enhanced data processing capabilities"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'transaction_id': range(100),
            'customer_id': np.random.randint(1, 20, 100),
            'amount': np.random.exponential(50, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        # Add some missing values
        self.sample_data.loc[5:10, 'amount'] = np.nan
        self.sample_data.loc[15:20, 'merchant'] = np.nan
        
        # Add duplicates
        self.sample_data = pd.concat([self.sample_data, self.sample_data.iloc[0:2]], ignore_index=True)
        
        self.agent = DataAgent()
    
    def test_data_loading_multiple_formats(self):
        """Test loading data in various formats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV
            csv_path = os.path.join(tmpdir, 'test_data.csv')
            self.sample_data.to_csv(csv_path, index=False)
            df_csv = self.agent.load_data(csv_path)
            self.assertEqual(len(df_csv), len(self.sample_data))
            
            # Test JSON
            json_path = os.path.join(tmpdir, 'test_data.json')
            self.sample_data.to_json(json_path, orient='records')
            df_json = self.agent.load_data(json_path)
            self.assertEqual(len(df_json), len(self.sample_data))
            
            # Test Excel (if available)
            try:
                xlsx_path = os.path.join(tmpdir, 'test_data.xlsx')
                self.sample_data.to_excel(xlsx_path, index=False)
                df_xlsx = self.agent.load_data(xlsx_path)
                self.assertEqual(len(df_xlsx), len(self.sample_data))
            except ImportError:
                # Skip Excel test if not available
                pass
    
    def test_data_quality_assessment(self):
        """Test data quality assessment functionality"""
        self.agent.df = self.sample_data
        quality_report = self.agent.assess_data_quality()
        
        # Check that report contains expected metrics
        self.assertIn('total_rows', quality_report.metrics)
        self.assertIn('missing_values', quality_report.metrics)
        self.assertIn('duplicate_rows', quality_report.metrics)
        
        # Check that suggestions are generated
        self.assertIsInstance(quality_report.suggestions, list)
    
    def test_cleaning_suggestions(self):
        """Test automatic cleaning suggestions"""
        self.agent.df = self.sample_data
        self.agent.assess_data_quality()
        suggestions = self.agent.suggest_cleaning_operations()
        
        # Should have suggestions for missing values and duplicates
        self.assertGreater(len(suggestions), 0)
    
    def test_visualization_generation(self):
        """Test visualization generation"""
        self.agent.df = self.sample_data
        self.agent.clean_data()
        
        try:
            visualizations = self.agent.generate_visualizations()
            # Should generate at least one visualization
            self.assertGreater(len(visualizations), 0)
        except ImportError:
            # Skip if matplotlib/seaborn not available
            pass
    
    def test_time_series_analysis(self):
        """Test time-series pattern detection"""
        self.agent.df = self.sample_data
        self.agent.clean_data()
        
        patterns = self.agent.detect_time_series_patterns()
        # Should return a dictionary with patterns
        self.assertIsInstance(patterns, dict)
    
    def test_data_saving_multiple_formats(self):
        """Test saving data in various formats"""
        self.agent.df = self.sample_data
        self.agent.clean_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV
            csv_path = os.path.join(tmpdir, 'output_data.csv')
            self.agent.save_processed_data(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            # Test JSON
            json_path = os.path.join(tmpdir, 'output_data.json')
            self.agent.save_processed_data(json_path)
            self.assertTrue(os.path.exists(json_path))
    
    def test_data_quality_report_class(self):
        """Test DataQualityReport class functionality"""
        report = DataQualityReport()
        report.add_metric('test_metric', 42)
        report.add_suggestion('Test suggestion')
        
        report_data = report.get_report()
        self.assertIn('test_metric', report_data['metrics'])
        self.assertIn('Test suggestion', report_data['suggestions'])


if __name__ == '__main__':
    unittest.main()