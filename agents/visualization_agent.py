"""
Visualization Agent: Advanced data visualization and reporting capabilities
"""
import pandas as pd
import numpy as np
import logging
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Try to import optional visualization libraries
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    plotly = None

try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationAgent:
    """Agent for advanced data visualization and reporting"""
    
    def __init__(self):
        self.visualizations = {}
        self.report_data = {}
        
    def generate_advanced_visualizations(self, df: pd.DataFrame, risk_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Generate advanced visualizations for financial transaction data
        Args:
            df: Processed transaction data
            risk_data: Risk assessment data with fraud probabilities
        Returns:
            Dict of base64 encoded visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualizations.")
            return {}
            
        visualizations = {}
        
        try:
            # Generate all visualizations
            visualizations.update(self._generate_distribution_plots(df))
            visualizations.update(self._generate_time_series_plots(df))
            visualizations.update(self._generate_categorical_analysis(df))
            visualizations.update(self._generate_correlation_analysis(df))
            
            # Generate risk-based visualizations if risk data is provided
            if risk_data is not None:
                visualizations.update(self._generate_risk_visualizations(df, risk_data))
            
            # Generate interactive visualizations if Plotly is available
            if PLOTLY_AVAILABLE:
                visualizations.update(self._generate_interactive_visualizations(df, risk_data))
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            
        self.visualizations = visualizations
        return visualizations
    
    def _generate_distribution_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate distribution plots for numerical features"""
        visualizations = {}
        
        # Find amount column
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
                
        if amount_col:
            # Enhanced distribution plot with multiple views
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Amount Distribution Analysis', fontsize=16)
            
            # Histogram
            axes[0, 0].hist(df[amount_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel(amount_col)
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Histogram')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            box_plot = axes[0, 1].boxplot(df[amount_col], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightcoral')
            axes[0, 1].set_ylabel(amount_col)
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Log scale histogram (for skewed data)
            axes[1, 0].hist(np.log1p(df[amount_col]), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel(f'Log({amount_col} + 1)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Log-Scale Histogram')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot approximation
            sorted_amounts = np.sort(df[amount_col])
            theoretical_quantiles = np.linspace(0, 1, len(sorted_amounts))
            axes[1, 1].scatter(theoretical_quantiles, sorted_amounts, alpha=0.6, color='purple')
            axes[1, 1].set_xlabel('Theoretical Quantiles')
            axes[1, 1].set_ylabel('Sample Quantiles')
            axes[1, 1].set_title('Q-Q Plot Approximation')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['amount_distribution_analysis'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def _generate_time_series_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate time-series analysis plots"""
        visualizations = {}
        
        # Find date columns
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        if not date_cols:
            # Try to create a synthetic time index if none exists
            date_col = 'synthetic_date'
            df[date_col] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        else:
            date_col = date_cols[0]
            
        # Find amount column
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
                
        if amount_col:
            # Sort by date
            df_sorted = df.sort_values(by=date_col)
            
            # Create time series plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Time Series Analysis', fontsize=16)
            
            # Daily transaction volume
            daily_volume = df_sorted.set_index(date_col).resample('D')[amount_col].sum()
            axes[0, 0].plot(daily_volume.index, daily_volume.values, marker='o', markersize=3, linewidth=1)
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Total Amount')
            axes[0, 0].set_title('Daily Transaction Volume')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Daily transaction count
            daily_count = df_sorted.set_index(date_col).resample('D')[amount_col].count()
            axes[0, 1].plot(daily_count.index, daily_count.values, marker='s', markersize=3, linewidth=1, color='orange')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Transaction Count')
            axes[0, 1].set_title('Daily Transaction Count')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Moving average (7-day)
            daily_volume_ma = daily_volume.rolling(window=7).mean()
            axes[1, 0].plot(daily_volume.index, daily_volume.values, alpha=0.3, label='Daily', color='lightblue')
            axes[1, 0].plot(daily_volume_ma.index, daily_volume_ma.values, linewidth=2, label='7-day MA', color='darkblue')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Total Amount')
            axes[1, 0].set_title('Transaction Volume with Moving Average')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Hourly pattern (if hour column exists)
            if 'hour' in df.columns:
                hourly_pattern = df.groupby('hour')[amount_col].mean()
                axes[1, 1].bar(hourly_pattern.index, hourly_pattern.values, color='purple', alpha=0.7)
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Average Amount')
                axes[1, 1].set_title('Average Transaction Amount by Hour')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Hour data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hourly Pattern')
            
            plt.tight_layout()
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['time_series_analysis'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def _generate_categorical_analysis(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate categorical data analysis plots"""
        visualizations = {}
        
        # Find categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            # Take first few categorical columns for analysis
            cols_to_analyze = categorical_cols[:3]
            
            fig, axes = plt.subplots(1, len(cols_to_analyze), figsize=(5*len(cols_to_analyze), 6))
            if len(cols_to_analyze) == 1:
                axes = [axes]
            
            fig.suptitle('Categorical Data Analysis', fontsize=16)
            
            # Find amount column
            amount_col = None
            for col in ['amount', 'transaction_amount', 'amt', 'value']:
                if col in df.columns:
                    amount_col = col
                    break
            
            for i, col in enumerate(cols_to_analyze):
                if amount_col:
                    # Group by category and calculate statistics
                    category_stats = df.groupby(col)[amount_col].agg(['count', 'mean', 'sum']).head(10)
                    
                    # Bar plot of transaction counts
                    axes[i].bar(range(len(category_stats)), category_stats['count'], 
                               color=plt.cm.Set3(np.linspace(0, 1, len(category_stats))))
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Transaction Count')
                    axes[i].set_title(f'{col} - Transaction Count')
                    axes[i].set_xticks(range(len(category_stats)))
                    axes[i].set_xticklabels(category_stats.index, rotation=45, ha='right')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['categorical_analysis'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def _generate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate correlation analysis plots"""
        visualizations = {}
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            # Compute correlation matrix
            correlation_matrix = numeric_df.corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            
            if SEABORN_AVAILABLE:
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', cbar_kws={"shrink": .8})
            else:
                plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                plt.colorbar()
                
                # Add correlation values as text
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                ha='center', va='center', color='black')
            
            plt.title('Feature Correlation Matrix')
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            plt.tight_layout()
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def _generate_risk_visualizations(self, df: pd.DataFrame, risk_data: pd.DataFrame) -> Dict[str, str]:
        """Generate risk-based visualizations"""
        visualizations = {}
        
        # Check if risk data has fraud probability column
        if 'fraud_probability' in risk_data.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Risk Analysis Dashboard', fontsize=16)
            
            # Risk score distribution
            axes[0, 0].hist(risk_data['fraud_probability'], bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[0, 0].set_xlabel('Fraud Probability')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Risk Score Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk score vs Amount (scatter plot)
            amount_col = None
            for col in ['amount', 'transaction_amount', 'amt', 'value']:
                if col in df.columns:
                    amount_col = col
                    break
                    
            if amount_col:
                # Scatter plot with color coding for risk
                scatter = axes[0, 1].scatter(df[amount_col], risk_data['fraud_probability'], 
                                           alpha=0.6, c=risk_data['fraud_probability'], cmap='Reds')
                axes[0, 1].set_xlabel(amount_col)
                axes[0, 1].set_ylabel('Fraud Probability')
                axes[0, 1].set_title('Risk Score vs Transaction Amount')
                axes[0, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 1])
            
            # Risk categories
            risk_data['risk_category'] = pd.cut(risk_data['fraud_probability'], 
                                               bins=[0, 0.3, 0.7, 1.0], 
                                               labels=['Low', 'Medium', 'High'])
            risk_counts = risk_data['risk_category'].value_counts()
            
            axes[1, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                          colors=['lightgreen', 'orange', 'red'])
            axes[1, 0].set_title('Risk Category Distribution')
            
            # Time-based risk analysis (if date column exists)
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            if date_cols and amount_col:
                date_col = date_cols[0]
                df_with_risk = df.copy()
                df_with_risk['fraud_probability'] = risk_data['fraud_probability']
                df_with_risk[date_col] = pd.to_datetime(df_with_risk[date_col])
                
                # Average risk by day
                daily_risk = df_with_risk.set_index(date_col).resample('D')['fraud_probability'].mean()
                axes[1, 1].plot(daily_risk.index, daily_risk.values, marker='o', linewidth=2, color='purple')
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Average Fraud Probability')
                axes[1, 1].set_title('Average Risk Score Over Time')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Time-based risk analysis\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Time-based Risk Analysis')
            
            plt.tight_layout()
            
            # Save to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations['risk_analysis_dashboard'] = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        return visualizations
    
    def _generate_interactive_visualizations(self, df: pd.DataFrame, risk_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Generate interactive visualizations using Plotly"""
        visualizations = {}
        
        if not PLOTLY_AVAILABLE:
            return visualizations
            
        # Find amount column
        amount_col = None
        for col in ['amount', 'transaction_amount', 'amt', 'value']:
            if col in df.columns:
                amount_col = col
                break
                
        if amount_col:
            # Interactive distribution plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Distribution', 'Box Plot', 'Log Distribution', 'Cumulative Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[amount_col], name='Distribution', nbinsx=50),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=df[amount_col], name='Box Plot'),
                row=1, col=2
            )
            
            # Log distribution
            fig.add_trace(
                go.Histogram(x=np.log1p(df[amount_col]), name='Log Distribution', nbinsx=50),
                row=2, col=1
            )
            
            # Cumulative distribution
            sorted_vals = np.sort(df[amount_col])
            cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            fig.add_trace(
                go.Scatter(x=sorted_vals, y=cumulative, mode='lines', name='Cumulative'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Interactive Amount Distribution Analysis")
            
            # Convert to HTML
            visualizations['interactive_distribution'] = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return visualizations
    
    def generate_report_summary(self, df: pd.DataFrame, visualizations: Dict[str, str]) -> Dict:
        """
        Generate a summary report of the analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': len(df),
                'columns': list(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            },
            'visualizations_generated': list(visualizations.keys()),
            'data_quality': self._assess_data_quality(df)
        }
        
        self.report_data = report
        return report
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess basic data quality metrics"""
        return {
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
    
    def export_report(self, output_path: str, format: str = 'html'):
        """
        Export the analysis report
        Args:
            output_path: Path to save the report
            format: Export format ('html', 'pdf', 'json')
        """
        # Implementation would depend on the specific export format
        # This is a placeholder for the export functionality
        pass


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
        'date': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Create risk data
    risk_data = sample_data.copy()
    risk_data['fraud_probability'] = np.random.beta(2, 8, 1000)  # Most transactions have low risk
    
    # Test the visualization agent
    agent = VisualizationAgent()
    visualizations = agent.generate_advanced_visualizations(sample_data, risk_data)
    
    print(f"Generated {len(visualizations)} visualizations")
    for name in visualizations.keys():
        print(f"  - {name}")
    
    report = agent.generate_report_summary(sample_data, visualizations)
    print(f"\nReport generated: {report['dataset_info']['total_records']} records")