"""
Test cases for streaming agent functionality
"""
import unittest
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from agents.streaming_agent import StreamingAgent


class TestStreamingAgent(unittest.TestCase):
    """Test streaming agent capabilities"""
    
    def setUp(self):
        """Set up test data"""
        self.agent = StreamingAgent()
    
    def test_streaming_agent_initialization(self):
        """Test StreamingAgent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.buffer_size, 1000)
        self.assertEqual(self.agent.alert_threshold, 0.7)
        self.assertFalse(self.agent.is_monitoring)
    
    def test_subscription_mechanism(self):
        """Test subscription and unsubscription"""
        call_count = 0
        
        def test_callback(transaction, risk_score):
            nonlocal call_count
            call_count += 1
        
        # Test subscription
        self.agent.subscribe(test_callback)
        self.assertEqual(len(self.agent.subscribers), 1)
        
        # Test unsubscription
        self.agent.unsubscribe(test_callback)
        self.assertEqual(len(self.agent.subscribers), 0)
    
    def test_heuristic_risk_scoring(self):
        """Test heuristic risk scoring"""
        # Test normal transaction
        normal_transaction = {
            'amount': 50.0,
            'merchant': 'Amazon',
            'category': 'Online',
            'hour': 14
        }
        risk_score = self.agent._heuristic_risk_score(normal_transaction)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        
        # Test high-risk transaction
        high_risk_transaction = {
            'amount': 15000.0,
            'merchant': 'Casino',
            'category': 'Gambling',
            'hour': 3
        }
        risk_score = self.agent._heuristic_risk_score(high_risk_transaction)
        self.assertGreater(risk_score, 0.5)  # Should be high risk
    
    def test_buffer_statistics(self):
        """Test buffer statistics"""
        stats = self.agent.get_buffer_stats()
        self.assertIn('total_transactions', stats)
        self.assertIn('anomalies_detected', stats)
        self.assertIn('anomaly_rate', stats)
        self.assertIn('is_monitoring', stats)
    
    def test_recent_anomalies(self):
        """Test recent anomalies retrieval"""
        anomalies = self.agent.get_recent_anomalies()
        self.assertIsInstance(anomalies, list)
        self.assertLessEqual(len(anomalies), 10)


if __name__ == '__main__':
    unittest.main()