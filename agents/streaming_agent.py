"""
Streaming Agent: Handles real-time transaction monitoring and processing
"""
import pandas as pd
import numpy as np
import logging
import asyncio
import json
from typing import Dict, List, Optional, Callable
from collections import deque
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingAgent:
    """Agent for real-time transaction monitoring and processing"""
    
    def __init__(self, buffer_size: int = 1000, alert_threshold: float = 0.7):
        self.buffer_size = buffer_size
        self.alert_threshold = alert_threshold
        self.transaction_buffer = deque(maxlen=buffer_size)
        self.anomaly_buffer = deque(maxlen=buffer_size)
        self.subscribers = []
        self.is_monitoring = False
        self.risk_agent = None
        self.data_agent = None
        
    def set_risk_agent(self, risk_agent):
        """Set the risk agent for fraud detection"""
        self.risk_agent = risk_agent
        
    def set_data_agent(self, data_agent):
        """Set the data agent for data processing"""
        self.data_agent = data_agent
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time transaction alerts"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from real-time transaction alerts"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, transaction: Dict, risk_score: float):
        """Notify all subscribers of a high-risk transaction"""
        for subscriber in self.subscribers:
            try:
                subscriber(transaction, risk_score)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    async def process_transaction_stream(self, transaction_generator):
        """
        Process a stream of transactions in real-time
        transaction_generator: An async generator that yields transaction dictionaries
        """
        logger.info("Starting real-time transaction processing...")
        self.is_monitoring = True
        
        async for transaction in transaction_generator:
            if not self.is_monitoring:
                break
                
            # Add to buffer
            self.transaction_buffer.append(transaction)
            
            # Process transaction for risk
            risk_score = await self._assess_transaction_risk(transaction)
            
            # Check for anomalies
            if risk_score > self.alert_threshold:
                self.anomaly_buffer.append({
                    'transaction': transaction,
                    'risk_score': risk_score,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Notify subscribers
                self._notify_subscribers(transaction, risk_score)
                
                logger.warning(f"High-risk transaction detected: {risk_score:.3f}")
            
            # Process in batches for efficiency
            if len(self.transaction_buffer) % 100 == 0:
                logger.info(f"Processed {len(self.transaction_buffer)} transactions")
        
        logger.info("Stopped real-time transaction processing")
    
    async def _assess_transaction_risk(self, transaction: Dict) -> float:
        """
        Assess risk for a single transaction
        Returns risk score between 0.0 and 1.0
        """
        try:
            # If we have a trained risk model, use it
            if self.risk_agent and self.risk_agent.fraud_model:
                # Convert transaction to DataFrame
                df = pd.DataFrame([transaction])
                
                # Prepare features (same as in risk agent)
                if self.data_agent:
                    df_processed = self.data_agent.extract_features()
                else:
                    df_processed = df.copy()
                
                # Predict fraud probability
                try:
                    fraud_prob = self.risk_agent.fraud_model.predict_proba(df_processed)[0][1]
                    return float(fraud_prob)
                except Exception as e:
                    logger.warning(f"Error using fraud model: {e}")
                    # Fallback to simple heuristic scoring
                    return self._heuristic_risk_score(transaction)
            else:
                # Fallback to heuristic scoring
                return self._heuristic_risk_score(transaction)
                
        except Exception as e:
            logger.error(f"Error assessing transaction risk: {e}")
            return 0.0
    
    def _heuristic_risk_score(self, transaction: Dict) -> float:
        """
        Simple heuristic-based risk scoring
        """
        risk_score = 0.0
        
        # Amount-based risk
        amount = transaction.get('amount', 0)
        if amount > 10000:
            risk_score += 0.4
        elif amount > 5000:
            risk_score += 0.2
        elif amount > 1000:
            risk_score += 0.1
            
        # Time-based risk (unusual hours)
        hour = transaction.get('hour', 12)
        if hour < 6 or hour > 22:
            risk_score += 0.2
            
        # Merchant-based risk (if we have a list of high-risk merchants)
        high_risk_merchants = ['casino', 'gambling', 'adult']
        merchant = str(transaction.get('merchant', '')).lower()
        if any(risky in merchant for risky in high_risk_merchants):
            risk_score += 0.3
            
        # Category-based risk
        high_risk_categories = ['cash', 'gambling', 'adult']
        category = str(transaction.get('category', '')).lower()
        if any(risky in category for risky in high_risk_categories):
            risk_score += 0.2
            
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, risk_score))
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the transaction buffer"""
        return {
            'total_transactions': len(self.transaction_buffer),
            'anomalies_detected': len(self.anomaly_buffer),
            'anomaly_rate': len(self.anomaly_buffer) / max(1, len(self.transaction_buffer)),
            'is_monitoring': self.is_monitoring
        }
    
    def get_recent_anomalies(self, limit: int = 10) -> List[Dict]:
        """Get recent anomalous transactions"""
        return list(self.anomaly_buffer)[-limit:]
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        logger.info("Stopped real-time monitoring")


# Example usage and utility functions
async def mock_transaction_stream():
    """Mock transaction stream for testing"""
    merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'Casino']
    categories = ['Online', 'Retail', 'Food', 'Gas', 'Gambling']
    
    for i in range(1000):
        transaction = {
            'transaction_id': i,
            'amount': np.random.exponential(50),
            'merchant': np.random.choice(merchants),
            'category': np.random.choice(categories),
            'hour': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7)
        }
        
        # Occasionally create high-risk transactions
        if np.random.random() < 0.05:
            transaction['amount'] = np.random.exponential(5000)
            transaction['merchant'] = 'Casino'
            transaction['category'] = 'Gambling'
            transaction['hour'] = np.random.choice([2, 3, 4, 5])
        
        yield transaction
        await asyncio.sleep(0.01)  # Simulate real-time arrival


def alert_handler(transaction: Dict, risk_score: float):
    """Example alert handler"""
    print(f"🚨 ALERT: High-risk transaction (score: {risk_score:.3f})")
    print(f"   Transaction: {transaction}")


# Example usage
if __name__ == "__main__":
    async def main():
        agent = StreamingAgent()
        agent.subscribe(alert_handler)
        
        # Process mock stream
        await agent.process_transaction_stream(mock_transaction_stream())
        
        # Print stats
        stats = agent.get_buffer_stats()
        print(f"Final stats: {stats}")
    
    # Run example
    asyncio.run(main())