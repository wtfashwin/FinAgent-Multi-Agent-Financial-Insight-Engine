"""
Tests for the ML Enhancement Agent
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from ml_enhancement_agent import MLEnhancementAgent


class TestMLEnhancementAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and agent"""
        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell'], 100),
            'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], 100),
            'hour': np.random.randint(0, 24, 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        })
        
        self.agent = MLEnhancementAgent()
    
    def test_initialize_patterns(self):
        """Test that NL query patterns are initialized correctly"""
        patterns = self.agent.nl_query_patterns
        self.assertIsInstance(patterns, dict)
        self.assertIn('high_risk', patterns)
        self.assertIn('amount_range', patterns)
        self.assertIn('merchant', patterns)
    
    def test_find_columns(self):
        """Test column finding methods"""
        amount_col = self.agent._find_amount_column(self.sample_data)
        self.assertEqual(amount_col, 'amount')
        
        merchant_col = self.agent._find_merchant_column(self.sample_data)
        self.assertEqual(merchant_col, 'merchant')
        
        category_col = self.agent._find_category_column(self.sample_data)
        self.assertEqual(category_col, 'category')
    
    def test_parse_nl_query(self):
        """Test natural language query parsing"""
        # Test high risk query
        query = "Show me high risk transactions"
        filters = self.agent._parse_nl_query(query)
        self.assertEqual(filters['risk_level'], 'high')
        
        # Test amount range query
        query = "Find transactions with amount between 100 and 500"
        filters = self.agent._parse_nl_query(query)
        self.assertEqual(filters['amount_range'], (100.0, 500.0))
        
        # Test merchant query
        query = "Show me Amazon transactions"
        filters = self.agent._parse_nl_query(query)
        self.assertEqual(filters['merchant'], 'Amazon')
        
        # Test top N query
        query = "Top 5 highest amount transactions"
        filters = self.agent._parse_nl_query(query)
        self.assertEqual(filters['limit'], 5)
        self.assertEqual(filters['sort_by'], 'desc')
    
    def test_apply_filters(self):
        """Test applying filters to dataframe"""
        # Test amount range filter
        filters = {'amount_range': (100, 500), 'merchant': None, 'category': None, 'risk_level': None, 'sort_by': None, 'limit': None, 'time_period': None}
        filtered_df = self.agent._apply_filters(self.sample_data, filters)
        amount_col = self.agent._find_amount_column(self.sample_data)
        self.assertTrue((filtered_df[amount_col] >= 100).all())
        self.assertTrue((filtered_df[amount_col] <= 500).all())
        
        # Test merchant filter
        filters = {'amount_range': None, 'merchant': 'Amazon', 'category': None, 'risk_level': None, 'sort_by': None, 'limit': None, 'time_period': None}
        filtered_df = self.agent._apply_filters(self.sample_data, filters)
        self.assertTrue(filtered_df['merchant'].str.contains('Amazon').all())
    
    def test_natural_language_query(self):
        """Test natural language querying"""
        # Test simple query
        result = self.agent.natural_language_query(self.sample_data, "Show me high risk transactions")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(len(result), len(self.sample_data))
        
        # Test amount range query
        result = self.agent.natural_language_query(self.sample_data, "Find transactions with amount between 50 and 200")
        self.assertIsInstance(result, pd.DataFrame)
        if len(result) > 0:
            amount_col = self.agent._find_amount_column(result)
            self.assertTrue((result[amount_col] >= 50).all())
            self.assertTrue((result[amount_col] <= 200).all())
    
    def test_train_enhanced_model(self):
        """Test training enhanced ML models"""
        # Add some numeric features for training
        train_data = self.sample_data.copy()
        train_data['amount_log'] = np.log1p(train_data['amount'])
        train_data['hour_sin'] = np.sin(2 * np.pi * train_data['hour'] / 24)
        train_data['hour_cos'] = np.cos(2 * np.pi * train_data['hour'] / 24)
        
        results = self.agent.train_enhanced_model(train_data, 'is_fraud')
        
        # Check that results contain expected keys
        self.assertIn('model_performance', results)
        self.assertIn('feature_importance', results)
        self.assertIsInstance(results['model_performance'], dict)
        self.assertIsInstance(results['feature_importance'], dict)
        
        # Check that multiple models were trained
        performance = results['model_performance']
        self.assertIn('random_forest', performance)
        self.assertIn('xgboost', performance)
        self.assertIn('gradient_boosting', performance)
        self.assertIn('ensemble', performance)
        
        # Check AUC scores
        for model_name, metrics in performance.items():
            self.assertIn('auc_score', metrics)
            self.assertIsInstance(metrics['auc_score'], float)
    
    def test_explain_prediction(self):
        """Test prediction explanation"""
        # First train a model to have feature importance
        train_data = self.sample_data.copy()
        train_data['amount_log'] = np.log1p(train_data['amount'])
        
        self.agent.train_enhanced_model(train_data, 'is_fraud')
        
        # Test explanation for a sample transaction
        sample_transaction = train_data.iloc[0]
        explanation = self.agent.explain_prediction(sample_transaction)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('transaction_id', explanation)
        # Explanation should have either risk factors or feature contributions
        self.assertTrue(
            'risk_factors' in explanation or 'feature_contributions' in explanation
        )
    
    def test_get_model_insights(self):
        """Test getting model insights"""
        # First train a model
        train_data = self.sample_data.copy()
        train_data['amount_log'] = np.log1p(train_data['amount'])
        
        self.agent.train_enhanced_model(train_data, 'is_fraud')
        
        # Get insights
        insights = self.agent.get_model_insights()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('feature_importance', insights)
        self.assertIn('available_models', insights)
        self.assertIsInstance(insights['feature_importance'], dict)
        self.assertIsInstance(insights['available_models'], list)
        self.assertGreater(len(insights['available_models']), 0)


if __name__ == '__main__':
    unittest.main()