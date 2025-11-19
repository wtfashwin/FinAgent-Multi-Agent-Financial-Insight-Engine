"""
Test cases for enhanced risk assessment features
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from agents.risk_agent import RiskAgent


class TestEnhancedRiskAssessment(unittest.TestCase):
    """Test enhanced risk assessment capabilities"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'merchant_id': np.random.randint(1, 50, 100),
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        })
        
        self.agent = RiskAgent()
    
    def test_ensemble_model_training(self):
        """Test training fraud detection model with ensemble methods"""
        # Train the model
        metrics = self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        
        # Check that metrics are returned
        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('classification_report', metrics)
        self.assertGreaterEqual(metrics['roc_auc'], 0.5)  # Should be better than random
    
    def test_anomaly_detection(self):
        """Test unsupervised anomaly detection"""
        # Train the anomaly model
        metrics = self.agent.train_anomaly_model(self.sample_data)
        
        # Check that metrics are returned
        self.assertIn('anomalies_detected', metrics)
        self.assertIn('anomaly_rate', metrics)
        self.assertGreaterEqual(metrics['anomalies_detected'], 0)
    
    def test_fraud_prediction(self):
        """Test fraud probability prediction"""
        # First train the model
        self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        
        # Make predictions
        predictions_df = self.agent.predict_fraud(self.sample_data)
        
        # Check that predictions are added to dataframe
        self.assertIn('fraud_probability', predictions_df.columns)
        self.assertIn('fraud_prediction', predictions_df.columns)
        self.assertIn('risk_level', predictions_df.columns)
        
        # Check that probabilities are in valid range
        self.assertTrue((predictions_df['fraud_probability'] >= 0).all())
        self.assertTrue((predictions_df['fraud_probability'] <= 1).all())
    
    def test_anomaly_detection_prediction(self):
        """Test anomaly detection prediction"""
        # First train the model
        self.agent.train_anomaly_model(self.sample_data)
        
        # Detect anomalies
        anomalies_df = self.agent.detect_anomalies(self.sample_data)
        
        # Check that anomaly columns are added
        self.assertIn('is_anomaly', anomalies_df.columns)
        self.assertIn('anomaly_score', anomalies_df.columns)
    
    def test_risk_explanation(self):
        """Test explainable AI features for risk factors"""
        # First train the model
        self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        
        # Get risk explanation
        explanation = self.agent.get_risk_explanation(self.sample_data)
        
        # Check that explanation contains expected elements
        self.assertIsInstance(explanation, dict)
        self.assertIn('top_risk_factors', explanation)
    
    def test_dynamic_model_update(self):
        """Test dynamic model updating with new data"""
        # First train the model
        initial_metrics = self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        
        # Create new data for updating
        new_data = self.sample_data.copy()
        new_data['is_fraud'] = np.random.choice([0, 1], len(new_data), p=[0.85, 0.15])
        
        # Update the model
        updated_metrics = self.agent.update_model_with_new_data(new_data, target_col='is_fraud')
        
        # Check that updated metrics are returned
        self.assertIn('accuracy', updated_metrics)
        self.assertIn('roc_auc', updated_metrics)
    
    def test_model_persistence(self):
        """Test saving and loading models"""
        # Train a model
        self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save models
            self.agent.save_models(tmpdir)
            
            # Check that model files are created
            model_files = os.listdir(tmpdir)
            self.assertIn('fraud_model.pkl', model_files)
            self.assertIn('scaler.pkl', model_files)
            self.assertIn('feature_columns.txt', model_files)
            
            # Create a new agent and load models
            new_agent = RiskAgent()
            new_agent.load_models(tmpdir)
            
            # Check that models are loaded
            self.assertIsNotNone(new_agent.fraud_model)
            self.assertIsNotNone(new_agent.scaler)
            self.assertGreater(len(new_agent.feature_columns), 0)
    
    def test_risk_summary(self):
        """Test risk assessment summary generation"""
        # Train model and make predictions
        self.agent.train_fraud_model(self.sample_data, target_col='is_fraud')
        predictions_df = self.agent.predict_fraud(self.sample_data)
        
        # Generate summary
        summary = self.agent.get_risk_summary(predictions_df)
        
        # Check that summary contains expected elements
        self.assertIsInstance(summary, dict)
        self.assertIn('high_risk_transactions', summary)
        self.assertIn('medium_risk_transactions', summary)
        self.assertIn('low_risk_transactions', summary)


if __name__ == '__main__':
    unittest.main()