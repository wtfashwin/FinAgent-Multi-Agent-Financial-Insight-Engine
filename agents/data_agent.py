"""
Data Agent: Handles data ingestion, cleansing, and anomaly tagging
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from langchain_core.documents import Document
import logging
from io import BytesIO
import base64

# Optional imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityReport:
    """Class to store and manage data quality metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.suggestions = []
        
    def add_metric(self, name: str, value):
        """Add a quality metric"""
        self.metrics[name] = value
        
    def add_suggestion(self, suggestion: str):
        """Add a data cleaning suggestion"""
        self.suggestions.append(suggestion)
        
    def get_report(self) -> Dict:
        """Get the complete quality report"""
        return {
            'metrics': self.metrics,
            'suggestions': self.suggestions
        }


class DataAgent:
    """Agent responsible for data processing and anomaly detection"""
    
    SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx', '.xls', '.parquet']
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.anomalies = []
        self.quality_report = DataQualityReport()
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load transaction data from various formats"""
        path = file_path or self.data_path
        
        if not path:
            raise ValueError("No file path provided")
            
        file_path_obj = Path(path)
        file_extension = file_path_obj.suffix.lower()
        
        try:
            logger.info(f"Loading data from {path}")
            
            if file_extension == '.csv':
                self.df = pd.read_csv(path)
            elif file_extension == '.json':
                self.df = pd.read_json(path)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(path)
            elif file_extension == '.parquet':
                self.df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {self.SUPPORTED_FORMATS}")
                
            logger.info(f"Loaded {len(self.df)} transactions")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_schema(self, required_columns: List[str] = None) -> bool:
        """Validate data schema and check for required columns"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if required_columns is None:
            # Default required columns for financial transactions
            required_columns = ['amount']
            
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
            
        # Check data types
        for col in self.df.columns:
            if col == 'amount' and not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column '{col}' should be numeric but is {self.df[col].dtype}")
                
        logger.info("Schema validation completed")
        return True
    
    def assess_data_quality(self) -> DataQualityReport:
        """Assess overall data quality and generate report"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        report = DataQualityReport()
        
        # Basic metrics
        report.add_metric('total_rows', len(self.df))
        report.add_metric('total_columns', len(self.df.columns))
        report.add_metric('missing_values', self.df.isnull().sum().sum())
        report.add_metric('duplicate_rows', self.df.duplicated().sum())
        
        # Missing value percentage per column
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        report.add_metric('missing_percentage_by_column', missing_pct.to_dict())
        
        # Data type information
        dtypes_info = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        report.add_metric('column_data_types', dtypes_info)
        
        # Generate suggestions based on quality metrics
        if report.metrics['duplicate_rows'] > 0:
            report.add_suggestion(f"Remove {report.metrics['duplicate_rows']} duplicate rows")
            
        for col, pct in missing_pct.items():
            if pct > 50:
                report.add_suggestion(f"Column '{col}' has {pct:.1f}% missing values - consider dropping or imputing")
            elif pct > 10:
                report.add_suggestion(f"Column '{col}' has {pct:.1f}% missing values - consider imputation")
                
        # Check for potential date columns
        potential_date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if potential_date_cols:
            report.add_suggestion(f"Potential date columns detected: {potential_date_cols} - ensure proper datetime conversion")
            
        self.quality_report = report
        return report
    
    def suggest_cleaning_operations(self) -> List[str]:
        """Generate automatic cleaning suggestions based on data quality"""
        if not hasattr(self, 'quality_report') or not self.quality_report.metrics:
            self.assess_data_quality()
            
        return self.quality_report.suggestions
    
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
    
    def generate_visualizations(self) -> Dict[str, str]:
        """Generate basic data visualizations for exploratory analysis"""
        if self.processed_df is None:
            raise ValueError("No processed data available")
            
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualizations.")
            return {}
            
        df = self.processed_df
        visualizations = {}
        
        # Find amount column
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
                
        if amount_col:
            # Distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(df[amount_col], bins=50, alpha=0.7)
            plt.xlabel(amount_col)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {amount_col}')
            plt.grid(True, alpha=0.3)
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations['amount_distribution'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            # Box plot for outlier detection
            plt.figure(figsize=(8, 6))
            plt.boxplot(df[amount_col])
            plt.ylabel(amount_col)
            plt.title(f'Box Plot of {amount_col}')
            plt.grid(True, alpha=0.3)
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations['amount_boxplot'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        # Correlation heatmap for numeric columns (only if seaborn is available)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1 and SEABORN_AVAILABLE:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        elif len(numeric_df.columns) > 1:
            # Fallback to matplotlib only correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.title('Correlation Matrix')
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def detect_time_series_patterns(self) -> Dict:
        """Detect temporal patterns in time-series data"""
        if self.processed_df is None:
            raise ValueError("No processed data available")
            
        df = self.processed_df
        patterns = {}
        
        # Find date columns
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        if not date_cols:
            logger.info("No datetime columns found for time-series analysis")
            return patterns
            
        date_col = date_cols[0]
        df_sorted = df.sort_values(by=date_col)
        
        # Find amount column
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
                
        if amount_col:
            # Resample to daily frequency and calculate statistics
            daily_stats = df_sorted.set_index(date_col).resample('D')[amount_col].agg(['sum', 'count', 'mean'])
            
            # Detect trends
            patterns['daily_transaction_volume'] = {
                'total_days': len(daily_stats),
                'avg_daily_transactions': daily_stats['count'].mean(),
                'total_amount': daily_stats['sum'].sum(),
                'avg_transaction_amount': daily_stats['mean'].mean()
            }
            
            # Simple trend detection (increasing/decreasing)
            if len(daily_stats) > 2:
                volume_trend = np.polyfit(range(len(daily_stats)), daily_stats['sum'], 1)[0]
                patterns['volume_trend'] = 'increasing' if volume_trend > 0 else 'decreasing'
                
                count_trend = np.polyfit(range(len(daily_stats)), daily_stats['count'], 1)[0]
                patterns['count_trend'] = 'increasing' if count_trend > 0 else 'decreasing'
        
        return patterns
    
    def to_langchain_documents(self, df:pd.DataFrame,content_cols: List[str] = None) -> List[Document]:
        """
        Convert processed DataFrame into LangChain Documents
        Args:
            df(pd.DataFrame): DataFrame processed by Risk Agent
            content_cols(List[str]): Columns to include in the main document content. 

        Returns:
            List[Document]: List of LangChain Document objects
        """
        if content_cols is None:
            content_cols = [col for col in df.columns if col not in ['V1', 'V2', 'V3', 'is_anomaly', 'fraud_probability']]
        documents = []
        for idx, row in df.iterrows():
            content = f"Transaction ID: {idx}\n"
            content += "".join([f"{col}: {row[col]}\n" for col in content_cols if col in row])

            metadata = {
                "transaction_index": int(idx),
                "is_anomaly": bool(row.get('is_anomaly', False)),
                "risk_score": float(row.get('fraud_probability', 0.0)),
                "source_file": "transactions.csv"
            } 
            documents.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Converted {len(documents)} records to LangChain Documents")
        return documents

    def run(self, file_name:str) -> pd.DataFrame:
        """ Executes Data Agent's workflow end-to-end """
        df = self.ingest_data(file_name)
        df = self.clean_data(df)
        return df
    
    def save_processed_data(self, output_path: str):
        """Save processed data to various formats"""
        if self.processed_df is None:
            raise ValueError("No processed data to save")
        
        file_path_obj = Path(output_path)
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.csv':
            self.processed_df.to_csv(output_path, index=False)
        elif file_extension == '.json':
            self.processed_df.to_json(output_path, orient='records')
        elif file_extension in ['.xlsx', '.xls']:
            self.processed_df.to_excel(output_path, index=False)
        elif file_extension == '.parquet':
            self.processed_df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
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