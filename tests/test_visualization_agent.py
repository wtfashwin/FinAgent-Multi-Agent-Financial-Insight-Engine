"""
Test cases for visualization agent functionality
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import base64

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from agents.visualization_agent import VisualizationAgent


class TestVisualizationAgent(unittest.TestCase):
    """Test visualization agent capabilities"""
    
    def setUp(self):
        """Set up test data"""
        self.agent = VisualizationAgent()
        
        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell'], 100),
            'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], 100),
            'hour': np.random.randint(0, 24, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        # Create risk data
        self.risk_data = self.sample_data.copy()
        self.risk_data['fraud_probability'] = np.random.beta(2, 8, 100)
    
    def test_visualization_agent_initialization(self):
        """Test VisualizationAgent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.visualizations, {})
        self.assertEqual(self.agent.report_data, {})
    
    def test_advanced_visualizations_generation(self):
        """Test advanced visualizations generation"""
        try:
            visualizations = self.agent.generate_advanced_visualizations(self.sample_data, self.risk_data)
            # Should generate at least some visualizations
            self.assertIsInstance(visualizations, dict)
            # Note: Actual generation may depend on available libraries
        except ImportError as e:
            # Skip test if required libraries are not available
            self.skipTest(f"Required visualization libraries not available: {e}")
    
    def test_distribution_plots(self):
        """Test distribution plot generation"""
        try:
            visualizations = self.agent._generate_distribution_plots(self.sample_data)
            # Should return a dictionary (may be empty if matplotlib not available)
            self.assertIsInstance(visualizations, dict)
        except ImportError as e:
            self.skipTest(f"Matplotlib not available: {e}")
    
    def test_time_series_plots(self):
        """Test time series plot generation"""
        try:
            visualizations = self.agent._generate_time_series_plots(self.sample_data)
            # Should return a dictionary (may be empty if matplotlib not available)
            self.assertIsInstance(visualizations, dict)
        except ImportError as e:
            self.skipTest(f"Matplotlib not available: {e}")
    
    def test_categorical_analysis(self):
        """Test categorical analysis plot generation"""
        try:
            visualizations = self.agent._generate_categorical_analysis(self.sample_data)
            # Should return a dictionary (may be empty if matplotlib not available)
            self.assertIsInstance(visualizations, dict)
        except ImportError as e:
            self.skipTest(f"Matplotlib not available: {e}")
    
    def test_correlation_analysis(self):
        """Test correlation analysis plot generation"""
        try:
            visualizations = self.agent._generate_correlation_analysis(self.sample_data)
            # Should return a dictionary (may be empty if matplotlib not available)
            self.assertIsInstance(visualizations, dict)
        except ImportError as e:
            self.skipTest(f"Matplotlib not available: {e}")
    
    def test_risk_visualizations(self):
        """Test risk-based visualization generation"""
        try:
            visualizations = self.agent._generate_risk_visualizations(self.sample_data, self.risk_data)
            # Should return a dictionary (may be empty if matplotlib not available)
            self.assertIsInstance(visualizations, dict)
        except ImportError as e:
            self.skipTest(f"Matplotlib not available: {e}")
    
    def test_report_summary_generation(self):
        """Test report summary generation"""
        visualizations = {}  # Mock visualizations
        report = self.agent.generate_report_summary(self.sample_data, visualizations)
        
        # Should return a dictionary with report data
        self.assertIsInstance(report, dict)
        self.assertIn('timestamp', report)
        self.assertIn('dataset_info', report)
        self.assertIn('visualizations_generated', report)
        self.assertIn('data_quality', report)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        quality = self.agent._assess_data_quality(self.sample_data)
        
        # Should return a dictionary with quality metrics
        self.assertIsInstance(quality, dict)
        self.assertIn('missing_values', quality)
        self.assertIn('duplicate_rows', quality)
        self.assertIn('memory_usage_mb', quality)


if __name__ == '__main__':
    unittest.main()