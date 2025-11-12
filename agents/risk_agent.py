"""
Risk Agent: Computes fraud probability using ML models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import logging
from typing import Dict, Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAgent:
    """Agent for fraud detection and risk assessment"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.fraud_model = None
        self.anomaly_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML models"""
        logger.info("Preparing features for ML model...")
        
        df_processed = df.copy()
        
        # Identify target column if not specified
        if target_col is None:
            fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
            target_col = fraud_cols[0] if fraud_cols else None
        
        # Separate features and target
        y = df_processed[target_col] if target_col and target_col in df_processed.columns else None
        
        if target_col and target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
        else:
            X = df_processed
        
        # Handle datetime columns
        for col in X.select_dtypes(include=['datetime64']).columns:
            X[f'{col}_hour'] = X[col].dt.hour
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X = X.drop(columns=[col])
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen labels
                X[col] = X[col].astype(str).map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.feature_columns = list(X.columns)
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
        return X_scaled, y
    
    def train_fraud_model(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """Train fraud detection model"""
        logger.info("Training fraud detection model...")
        
        X, y = self.prepare_features(df, target_col)
        
        if y is None:
            raise ValueError("No target column found. Cannot train supervised model.")
        
        # Check class balance
        fraud_rate = y.mean()
        logger.info(f"Fraud rate in training data: {fraud_rate:.2%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        if fraud_rate < 0.1:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Training samples: {len(X_train)}")
        
        # Train Random Forest
        logger.info("Training Random Forest classifier...")
        self.fraud_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.fraud_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.fraud_model.predict(X_test)
        y_pred_proba = self.fraud_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(self.fraud_model.score(X_test, y_test)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(zip(
                self.feature_columns,
                self.fraud_model.feature_importances_.tolist()
            ))
        }
        
        logger.info(f"Model trained - ROC AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_anomaly_model(self, df: pd.DataFrame) -> Dict:
        """Train unsupervised anomaly detection model"""
        logger.info("Training anomaly detection model...")
        
        X, _ = self.prepare_features(df, target_col=None)
        
        # Train Isolation Forest
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Assume 10% anomalies
            random_state=42,
            n_jobs=-1
        )
        
        self.anomaly_model.fit(X)
        
        # Predict anomalies
        predictions = self.anomaly_model.predict(X)
        anomaly_scores = self.anomaly_model.score_samples(X)
        
        # -1 for anomalies, 1 for normal
        n_anomalies = (predictions == -1).sum()
        
        metrics = {
            'total_samples': len(X),
            'anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(X)),
            'mean_anomaly_score': float(anomaly_scores.mean()),
        }
        
        logger.info(f"Anomaly detection trained - Found {n_anomalies} anomalies ({metrics['anomaly_rate']:.2%})")
        
        return metrics
    
    def predict_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud probability for transactions"""
        if self.fraud_model is None:
            raise ValueError("Fraud model not trained. Call train_fraud_model() first.")
        
        logger.info("Predicting fraud probabilities...")
        
        X, _ = self.prepare_features(df, target_col=None)
        
        # Ensure columns match training
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns]
        
        # Predict
        fraud_probabilities = self.fraud_model.predict_proba(X)[:, 1]
        fraud_predictions = self.fraud_model.predict(X)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['fraud_probability'] = fraud_probabilities
        result_df['fraud_prediction'] = fraud_predictions
        result_df['risk_level'] = pd.cut(
            fraud_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        logger.info(f"Predictions complete - {fraud_predictions.sum()} flagged as fraud")
        
        return result_df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in transactions"""
        if self.anomaly_model is None:
            raise ValueError("Anomaly model not trained. Call train_anomaly_model() first.")
        
        logger.info("Detecting anomalies...")
        
        X, _ = self.prepare_features(df, target_col=None)
        
        # Ensure columns match
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_columns]
        
        # Predict
        predictions = self.anomaly_model.predict(X)
        anomaly_scores = self.anomaly_model.score_samples(X)
        
        # Add to dataframe
        result_df = df.copy()
        result_df['is_anomaly'] = (predictions == -1).astype(int)
        result_df['anomaly_score'] = anomaly_scores
        
        logger.info(f"Anomaly detection complete - {result_df['is_anomaly'].sum()} anomalies found")
        
        return result_df
    
    def get_risk_summary(self, df: pd.DataFrame) -> Dict:
        """Generate risk assessment summary"""
        summary = {}
        
        if 'fraud_probability' in df.columns:
            summary['high_risk_transactions'] = int((df['fraud_probability'] > 0.7).sum())
            summary['medium_risk_transactions'] = int(
                ((df['fraud_probability'] > 0.3) & (df['fraud_probability'] <= 0.7)).sum()
            )
            summary['low_risk_transactions'] = int((df['fraud_probability'] <= 0.3).sum())
            summary['average_fraud_probability'] = float(df['fraud_probability'].mean())
            
            # Amount at risk
            if 'amount' in df.columns:
                high_risk_mask = df['fraud_probability'] > 0.7
                summary['total_amount_at_risk'] = float(df[high_risk_mask]['amount'].sum())
        
        if 'is_anomaly' in df.columns:
            summary['total_anomalies'] = int(df['is_anomaly'].sum())
            summary['anomaly_percentage'] = float(df['is_anomaly'].mean() * 100)
        
        return summary
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.fraud_model:
            joblib.dump(self.fraud_model, output_path / 'fraud_model.pkl')
            logger.info(f"Saved fraud model to {output_path / 'fraud_model.pkl'}")
        
        if self.anomaly_model:
            joblib.dump(self.anomaly_model, output_path / 'anomaly_model.pkl')
            logger.info(f"Saved anomaly model to {output_path / 'anomaly_model.pkl'}")
        
        if self.scaler:
            joblib.dump(self.scaler, output_path / 'scaler.pkl')
        
        if self.label_encoders:
            joblib.dump(self.label_encoders, output_path / 'label_encoders.pkl')
        
        # Save feature columns
        with open(output_path / 'feature_columns.txt', 'w') as f:
            f.write('\n'.join(self.feature_columns))
    
    def load_models(self, model_dir: str):
        """Load trained models"""
        model_path = Path(model_dir)
        
        if (model_path / 'fraud_model.pkl').exists():
            self.fraud_model = joblib.load(model_path / 'fraud_model.pkl')
            logger.info("Loaded fraud model")
        
        if (model_path / 'anomaly_model.pkl').exists():
            self.anomaly_model = joblib.load(model_path / 'anomaly_model.pkl')
            logger.info("Loaded anomaly model")
        
        if (model_path / 'scaler.pkl').exists():
            self.scaler = joblib.load(model_path / 'scaler.pkl')
        
        if (model_path / 'label_encoders.pkl').exists():
            self.label_encoders = joblib.load(model_path / 'label_encoders.pkl')
        
        if (model_path / 'feature_columns.txt').exists():
            with open(model_path / 'feature_columns.txt', 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'transaction_id': range(n_samples),
        'amount': np.random.exponential(100, n_samples),
        'merchant_id': np.random.randint(1, 100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    agent = RiskAgent()
    
    print("\n=== Training Fraud Detection Model ===")
    metrics = agent.train_fraud_model(sample_df, target_col='is_fraud')
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n=== Training Anomaly Detection Model ===")
    anomaly_metrics = agent.train_anomaly_model(sample_df)
    print(f"Anomalies detected: {anomaly_metrics['anomalies_detected']}")