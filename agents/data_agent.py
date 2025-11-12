"""
Data Agent: Handles data ingestion, cleansing, and anomaly tagging
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAgent:
    """Agent responsible for data processing and anomaly detection"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.anomalies = []
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load transaction data from CSV"""
        path = file_path or self.data_path
        
        try:
            logger.info(f"Loading data from {path}")
            self.df = pd.read_csv(path)
            logger.info(f"Loaded {len(self.df)} transactions")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess transaction data"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning...")
        df = self.df.copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        self.processed_df = df
        logger.info("Data cleaning completed")
        return df
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect anomalies in transaction data using statistical methods"""
        if self.processed_df is None:
            raise ValueError("No processed data. Call clean_data() first.")
        
        logger.info("Detecting anomalies...")
        df = self.processed_df.copy()
        anomalies = []
        
        # Find amount column (common names)
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
        
        if amount_col:
            # Z-score based anomaly detection
            mean = df[amount_col].mean()
            std = df[amount_col].std()
            z_scores = np.abs((df[amount_col] - mean) / std)
            
            # Flag transactions with z-score > 3
            anomaly_mask = z_scores > 3
            anomaly_indices = df[anomaly_mask].index.tolist()
            
            for idx in anomaly_indices:
                anomalies.append({
                    'index': int(idx),
                    'type': 'statistical_outlier',
                    'reason': f'Amount {df.loc[idx, amount_col]:.2f} is {z_scores[idx]:.2f} standard deviations from mean',
                    'severity': 'high' if z_scores[idx] > 4 else 'medium'
                })
            
            logger.info(f"Detected {len(anomaly_indices)} statistical outliers")
        
        # Detect unusual transaction frequencies (if customer_id exists)
        customer_cols = [col for col in df.columns if 'customer' in col.lower() or 'user' in col.lower()]
        if customer_cols:
            customer_col = customer_cols[0]
            transaction_counts = df[customer_col].value_counts()
            
            # Flag customers with unusually high transaction counts
            mean_transactions = transaction_counts.mean()
            std_transactions = transaction_counts.std()
            
            unusual_customers = transaction_counts[transaction_counts > mean_transactions + 3 * std_transactions]
            
            for customer, count in unusual_customers.items():
                customer_transactions = df[df[customer_col] == customer].index.tolist()
                anomalies.append({
                    'indices': customer_transactions,
                    'type': 'high_frequency',
                    'reason': f'Customer {customer} has {count} transactions (mean: {mean_transactions:.1f})',
                    'severity': 'medium'
                })
            
            logger.info(f"Detected {len(unusual_customers)} high-frequency customers")
        
        # Add anomaly flags to dataframe
        df['is_anomaly'] = False
        for anomaly in anomalies:
            if 'index' in anomaly:
                df.loc[anomaly['index'], 'is_anomaly'] = True
            elif 'indices' in anomaly:
                df.loc[anomaly['indices'], 'is_anomaly'] = True
        
        self.processed_df = df
        self.anomalies = anomalies
        
        logger.info(f"Total anomalies detected: {len(anomalies)}")
        return anomalies
    
    def extract_features(self) -> pd.DataFrame:
        """Extract features for ML models"""
        if self.processed_df is None:
            raise ValueError("No processed data available")
        
        logger.info("Extracting features...")
        df = self.processed_df.copy()
        
        # Add temporal features if date column exists
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        if date_cols:
            date_col = date_cols[0]
            df['hour'] = df[date_col].dt.hour
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df[date_col].dt.month
        
        # One-hot encode categorical variables (limited to prevent memory issues)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if df[col].nunique() < 50:  # Only encode if less than 50 unique values
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        logger.info(f"Extracted features. Final shape: {df.shape}")
        self.processed_df = df
        return df
    
    def get_statistics(self) -> Dict:
        """Get summary statistics of the dataset"""
        if self.processed_df is None:
            return {}
        
        df = self.processed_df
        
        stats = {
            'total_transactions': len(df),
            'total_anomalies': len(self.anomalies),
            'anomaly_rate': len(self.anomalies) / len(df) if len(df) > 0 else 0,
            'columns': list(df.columns),
            'numeric_summary': df.describe().to_dict(),
        }
        
        # Add fraud statistics if fraud column exists
        fraud_cols = [col for col in df.columns if 'fraud' in col.lower()]
        if fraud_cols:
            fraud_col = fraud_cols[0]
            stats['fraud_count'] = int(df[fraud_col].sum())
            stats['fraud_rate'] = float(df[fraud_col].mean())
        
        return stats
    
    def save_processed_data(self, output_path: str):
        """Save processed data to CSV"""
        if self.processed_df is None:
            raise ValueError("No processed data to save")
        
        self.processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


# Example usage
if __name__ == "__main__":
    agent = DataAgent()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'transaction_id': range(100),
        'customer_id': np.random.randint(1, 20, 100),
        'amount': np.random.exponential(50, 100),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks'], 100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    
    sample_data.to_csv('/tmp/sample_transactions.csv', index=False)
    
    # Test the agent
    agent.load_data('/tmp/sample_transactions.csv')
    agent.clean_data()
    agent.detect_anomalies()
    stats = agent.get_statistics()
    
    print("\n=== Data Agent Statistics ===")
    print(f"Total Transactions: {stats['total_transactions']}")
    print(f"Total Anomalies: {stats['total_anomalies']}")
    print(f"Anomaly Rate: {stats['anomaly_rate']:.2%}")