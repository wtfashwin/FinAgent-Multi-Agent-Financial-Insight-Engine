"""
ML Enhancement Agent: Provides natural language querying and enhanced ML capabilities
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLEnhancementAgent:
    """Agent for enhanced ML capabilities and natural language querying"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.nl_query_patterns = self._initialize_nl_patterns()
    
    def _initialize_nl_patterns(self) -> Dict:
        """Initialize natural language query patterns"""
        return {
            'high_risk': [
                r'high risk', r'high fraud', r'suspicious', r'risky transactions',
                r'transactions with high probability of fraud'
            ],
            'low_risk': [
                r'low risk', r'safe transactions', r'normal transactions',
                r'transactions with low probability of fraud'
            ],
            'amount_range': [
                r'amount between (\d+) and (\d+)', r'transactions from (\d+) to (\d+)',
                r'amount > (\d+)', r'amount < (\d+)', r'amount >= (\d+)', r'amount <= (\d+)'
            ],
            'merchant': [
                r'merchant (.+)', r'transactions at (.+)', r'(.+) transactions'
            ],
            'category': [
                r'category (.+)', r'(.+) category transactions'
            ],
            'time_period': [
                r'last (\d+) days', r'past (\d+) days', r'recent (\d+) days'
            ],
            'top_n': [
                r'top (\d+)', r'bottom (\d+)', r'highest (\d+)', r'lowest (\d+)'
            ]
        }
    
    def train_enhanced_model(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict:
        """
        Train enhanced ML models with ensemble methods
        """
        try:
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle categorical variables
            X_processed = pd.get_dummies(X, drop_first=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {}
            predictions = {}
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
            models['random_forest'] = rf_model
            predictions['random_forest'] = rf_pred
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)  # Handle imbalance
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
            models['xgboost'] = xgb_model
            predictions['xgboost'] = xgb_pred
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
            models['gradient_boosting'] = gb_model
            predictions['gradient_boosting'] = gb_pred
            
            # Ensemble prediction (average of all models)
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            predictions['ensemble'] = ensemble_pred
            
            # Evaluate models
            results = {}
            for model_name, preds in predictions.items():
                auc_score = roc_auc_score(y_test, preds)
                results[model_name] = {
                    'auc_score': auc_score,
                    'predictions': preds.tolist()
                }
            
            # Store models and scaler
            self.models = models
            self.scalers['feature_scaler'] = scaler
            
            # Get feature importance (from Random Forest as it's most interpretable)
            feature_names = X_processed.columns.tolist()
            importances = rf_model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            self.feature_importance = dict(sorted_importance[:20])  # Top 20 features
            
            logger.info("Enhanced ML models trained successfully")
            return {
                'model_performance': results,
                'feature_importance': self.feature_importance,
                'trained_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training enhanced models: {e}")
            raise
    
    def natural_language_query(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Process natural language queries on transaction data
        """
        try:
            # Parse the query to extract filters and operations
            filters = self._parse_nl_query(query)
            
            # Apply filters to the dataframe
            filtered_df = self._apply_filters(df, filters)
            
            # Apply operations (sorting, limiting, etc.)
            result_df = self._apply_operations(filtered_df, filters)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            # Return original dataframe if parsing fails
            return df
    
    def _parse_nl_query(self, query: str) -> Dict:
        """
        Parse natural language query into structured filters
        """
        query_lower = query.lower()
        filters = {
            'amount_range': None,
            'merchant': None,
            'category': None,
            'risk_level': None,
            'sort_by': None,
            'limit': None,
            'time_period': None
        }
        
        # Check for risk level
        if any(pattern in query_lower for pattern in self.nl_query_patterns['high_risk']):
            filters['risk_level'] = 'high'
        elif any(pattern in query_lower for pattern in self.nl_query_patterns['low_risk']):
            filters['risk_level'] = 'low'
        
        # Check for amount range
        for pattern in self.nl_query_patterns['amount_range']:
            match = re.search(pattern, query_lower)
            if match:
                if 'between' in pattern or 'from' in pattern:
                    filters['amount_range'] = (float(match.group(1)), float(match.group(2)))
                elif '>' in pattern:
                    filters['amount_range'] = (float(match.group(1)), None)
                elif '<' in pattern:
                    filters['amount_range'] = (None, float(match.group(1)))
                break
        
        # Check for merchant
        for pattern in self.nl_query_patterns['merchant']:
            match = re.search(pattern, query_lower)
            if match:
                filters['merchant'] = match.group(1)
                break
        
        # Check for category
        for pattern in self.nl_query_patterns['category']:
            match = re.search(pattern, query_lower)
            if match:
                filters['category'] = match.group(1)
                break
        
        # Check for top N
        for pattern in self.nl_query_patterns['top_n']:
            match = re.search(pattern, query_lower)
            if match:
                filters['limit'] = int(match.group(1))
                if 'bottom' in pattern or 'lowest' in pattern:
                    filters['sort_by'] = 'asc'
                else:
                    filters['sort_by'] = 'desc'
                break
        
        return filters
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Apply parsed filters to dataframe
        """
        filtered_df = df.copy()
        
        # Apply amount range filter
        if filters['amount_range']:
            min_amount, max_amount = filters['amount_range']
            amount_col = self._find_amount_column(df)
            if amount_col:
                if min_amount is not None:
                    filtered_df = filtered_df[filtered_df[amount_col] >= min_amount]
                if max_amount is not None:
                    filtered_df = filtered_df[filtered_df[amount_col] <= max_amount]
        
        # Apply merchant filter
        if filters['merchant']:
            merchant_col = self._find_merchant_column(df)
            if merchant_col:
                filtered_df = filtered_df[
                    filtered_df[merchant_col].str.lower().str.contains(
                        filters['merchant'].lower(), na=False
                    )
                ]
        
        # Apply category filter
        if filters['category']:
            category_col = self._find_category_column(df)
            if category_col:
                filtered_df = filtered_df[
                    filtered_df[category_col].str.lower().str.contains(
                        filters['category'].lower(), na=False
                    )
                ]
        
        return filtered_df
    
    def _apply_operations(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Apply operations like sorting and limiting
        """
        result_df = df.copy()
        
        # Sort if requested
        if filters['sort_by'] and len(result_df) > 0:
            amount_col = self._find_amount_column(result_df)
            if amount_col:
                ascending = filters['sort_by'] == 'asc'
                result_df = result_df.sort_values(by=amount_col, ascending=ascending)
        
        # Limit results if requested
        if filters['limit'] and len(result_df) > filters['limit']:
            result_df = result_df.head(filters['limit'])
        
        return result_df
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the amount column in dataframe
        """
        amount_columns = ['amount', 'transaction_amount', 'amt', 'value']
        for col in amount_columns:
            if col in df.columns:
                return col
        return None
    
    def _find_merchant_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the merchant column in dataframe
        """
        merchant_columns = ['merchant', 'vendor', 'store']
        for col in merchant_columns:
            if col in df.columns:
                return col
        return None
    
    def _find_category_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the category column in dataframe
        """
        category_columns = ['category', 'type', 'transaction_type']
        for col in category_columns:
            if col in df.columns:
                return col
        return None
    
    def explain_prediction(self, transaction: pd.Series) -> Dict:
        """
        Explain why a transaction was flagged as high-risk
        """
        try:
            # Get top contributing features
            top_features = list(self.feature_importance.items())[:5]
            
            # Analyze the transaction against these features
            explanation = {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'risk_factors': [],
                'feature_contributions': []
            }
            
            for feature, importance in top_features:
                if feature in transaction:
                    value = transaction[feature]
                    explanation['feature_contributions'].append({
                        'feature': feature,
                        'value': value,
                        'importance': float(importance)
                    })
                    
                    # Add contextual explanation
                    if 'amount' in feature.lower() and value > 1000:
                        explanation['risk_factors'].append(
                            f"High transaction amount (${value:,.2f})"
                        )
                    elif 'hour' in feature.lower() and (value < 6 or value > 22):
                        explanation['risk_factors'].append(
                            f"Transaction at unusual hour ({value}:00)"
                        )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def get_model_insights(self) -> Dict:
        """
        Get insights from trained models
        """
        return {
            'feature_importance': self.feature_importance,
            'model_timestamp': datetime.now().isoformat(),
            'available_models': list(self.models.keys()) if self.models else []
        }


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'transaction_id': range(1000),
        'amount': np.random.exponential(100, 1000),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell'], 1000),
        'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], 1000),
        'hour': np.random.randint(0, 24, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Test the ML enhancement agent
    agent = MLEnhancementAgent()
    
    # Train models
    print("Training enhanced ML models...")
    results = agent.train_enhanced_model(sample_data, 'is_fraud')
    print(f"Model training completed. AUC scores: {results['model_performance']}")
    
    # Test natural language querying
    print("\nTesting natural language queries...")
    queries = [
        "Show me high risk transactions",
        "Find transactions with amount between 100 and 500",
        "Show me Amazon transactions",
        "Top 10 highest amount transactions"
    ]
    
    for query in queries:
        result = agent.natural_language_query(sample_data, query)
        print(f"Query: '{query}' -> Found {len(result)} transactions")
    
    # Test explanation
    print("\nTesting prediction explanation...")
    sample_transaction = sample_data.iloc[0]
    explanation = agent.explain_prediction(sample_transaction)
    print(f"Explanation for transaction: {explanation}")