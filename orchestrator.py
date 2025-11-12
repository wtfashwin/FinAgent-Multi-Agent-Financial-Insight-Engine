"""
Multi-Agent Orchestrator using LangGraph
Coordinates Data Agent, Insight Agent, and Risk Agent
"""
import logging
from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator
import pandas as pd
from pathlib import Path

from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from agents.risk_agent import RiskAgent
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state that gets passed between agents
class AgentState(TypedDict):
    """State shared across all agents"""
    raw_data_path: str
    data: pd.DataFrame
    processed_data: pd.DataFrame
    anomalies: List[Dict]
    insights: Dict
    risk_assessment: Dict
    fraud_predictions: pd.DataFrame
    summary: str
    error: str
    messages: Annotated[List[str], operator.add]


class FinAgentOrchestrator:
    """Orchestrates multiple agents using LangGraph"""
    
    def __init__(self, config=None):
        self.config = config or Config
        
        # Initialize agents
        self.data_agent = DataAgent()
        self.insight_agent = InsightAgent(config=self.config)
        self.risk_agent = RiskAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = None
        
    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow using LangGraph"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("data_processing", self._data_processing_node)
        workflow.add_node("insight_generation", self._insight_generation_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("summarization", self._summarization_node)
        
        # Define the workflow edges
        workflow.set_entry_point("data_processing")
        
        workflow.add_edge("data_processing", "insight_generation")
        workflow.add_edge("data_processing", "risk_assessment")
        
        workflow.add_edge("insight_generation", "summarization")
        workflow.add_edge("risk_assessment", "summarization")
        
        workflow.add_edge("summarization", END)
        
        return workflow
    
    def _data_processing_node(self, state: AgentState) -> AgentState:
        """Node for data processing using Data Agent"""
        logger.info("🔄 Running Data Processing Agent...")
        
        try:
            # Load data
            if state.get('raw_data_path'):
                self.data_agent.load_data(state['raw_data_path'])
            elif state.get('data') is not None:
                self.data_agent.df = state['data']
            else:
                raise ValueError("No data provided")
            
            # Clean data
            processed_df = self.data_agent.clean_data()
            
            # Extract features
            processed_df = self.data_agent.extract_features()
            
            # Detect anomalies
            anomalies = self.data_agent.detect_anomalies()
            
            # Get statistics
            stats = self.data_agent.get_statistics()
            
            state['processed_data'] = processed_df
            state['anomalies'] = anomalies
            state['messages'] = [
                f"✓ Data Processing Complete: {len(processed_df)} transactions processed",
                f"✓ Detected {len(anomalies)} anomalies"
            ]
            
            logger.info("✓ Data Processing Agent completed")
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            state['error'] = str(e)
            state['messages'] = [f"✗ Data Processing Error: {e}"]
        
        return state
    
    def _insight_generation_node(self, state: AgentState) -> AgentState:
        """Node for generating insights using Insight Agent"""
        logger.info("🔄 Running Insight Generation Agent...")
        
        try:
            processed_df = state.get('processed_data')
            
            if processed_df is None:
                raise ValueError("No processed data available")
            
            # Generate summary statistics
            summary = self.insight_agent.get_summary(processed_df)
            
            # Analyze trends
            trends = self.insight_agent.analyze_trends(processed_df)
            
            # Try to generate AI insights if LLM is available
            ai_insights = {}
            try:
                # Ingest data into vector store (sample for efficiency)
                sample_size = min(1000, len(processed_df))
                sample_df = processed_df.sample(n=sample_size, random_state=42)
                
                self.insight_agent.ingest_data(sample_df)
                
                # Generate insights on key questions
                questions = [
                    "What are the main spending patterns in this dataset?",
                    "Are there any unusual transaction behaviors?",
                    "What merchant categories have the highest transaction volumes?"
                ]
                
                for question in questions:
                    result = self.insight_agent.generate_insights(question)
                    ai_insights[question] = result['answer']
                
            except Exception as e:
                logger.warning(f"Could not generate AI insights: {e}")
                ai_insights = {"note": "AI insights unavailable (check API keys)"}
            
            state['insights'] = {
                'summary': summary,
                'trends': trends,
                'ai_insights': ai_insights
            }
            
            state['messages'] = state.get('messages', []) + [
                "✓ Insight Generation Complete",
                f"✓ Generated trend analysis and AI insights"
            ]
            
            logger.info("✓ Insight Generation Agent completed")
            
        except Exception as e:
            logger.error(f"Error in insight generation: {e}")
            state['error'] = state.get('error', '') + f"\nInsight Error: {e}"
            state['messages'] = state.get('messages', []) + [f"✗ Insight Generation Error: {e}"]
        
        return state
    
    def _risk_assessment_node(self, state: AgentState) -> AgentState:
        """Node for risk assessment using Risk Agent"""
        logger.info("🔄 Running Risk Assessment Agent...")
        
        try:
            processed_df = state.get('processed_data')
            
            if processed_df is None:
                raise ValueError("No processed data available")
            
            # Check if we have fraud labels for training
            fraud_cols = [col for col in processed_df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
            
            if fraud_cols:
                # Train supervised fraud detection model
                logger.info("Training fraud detection model...")
                fraud_col = fraud_cols[0]
                metrics = self.risk_agent.train_fraud_model(processed_df, target_col=fraud_col)
                
                # Make predictions
                predictions_df = self.risk_agent.predict_fraud(processed_df)
                
            else:
                # Use unsupervised anomaly detection
                logger.info("Training anomaly detection model (no fraud labels)...")
                metrics = self.risk_agent.train_anomaly_model(processed_df)
                
                # Detect anomalies
                predictions_df = self.risk_agent.detect_anomalies(processed_df)
            
            # Generate risk summary
            risk_summary = self.risk_agent.get_risk_summary(predictions_df)
            
            # Save models
            self.risk_agent.save_models(str(self.config.MODELS_DIR))
            
            state['fraud_predictions'] = predictions_df
            state['risk_assessment'] = {
                'metrics': metrics,
                'summary': risk_summary
            }
            
            state['messages'] = state.get('messages', []) + [
                "✓ Risk Assessment Complete",
                f"✓ Model trained and predictions generated"
            ]
            
            logger.info("✓ Risk Assessment Agent completed")
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            state['error'] = state.get('error', '') + f"\nRisk Assessment Error: {e}"
            state['messages'] = state.get('messages', []) + [f"✗ Risk Assessment Error: {e}"]
        
        return state
    
    def _summarization_node(self, state: AgentState) -> AgentState:
        """Final node to create comprehensive summary"""
        logger.info("🔄 Creating Final Summary...")
        
        try:
            summary_parts = []
            
            # Add insights summary
            if state.get('insights'):
                summary_parts.append(state['insights'].get('summary', ''))
            
            # Add risk summary
            if state.get('risk_assessment'):
                risk_summary = state['risk_assessment'].get('summary', {})
                
                risk_text = "\n🚨 Risk Assessment:\n"
                if 'high_risk_transactions' in risk_summary:
                    risk_text += f"  • High Risk: {risk_summary['high_risk_transactions']} transactions\n"
                    risk_text += f"  • Medium Risk: {risk_summary['medium_risk_transactions']} transactions\n"
                    risk_text += f"  • Low Risk: {risk_summary['low_risk_transactions']} transactions\n"
                
                if 'total_amount_at_risk' in risk_summary:
                    risk_text += f"  • Amount at Risk: ${risk_summary['total_amount_at_risk']:,.2f}\n"
                
                summary_parts.append(risk_text)
            
            # Add anomalies
            if state.get('anomalies'):
                anomaly_text = f"\n⚠️  Anomalies Detected: {len(state['anomalies'])} unusual transactions\n"
                summary_parts.append(anomaly_text)
            
            state['summary'] = '\n'.join(summary_parts)
            state['messages'] = state.get('messages', []) + ["✓ Analysis Complete"]
            
            logger.info("✓ Summarization completed")
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            state['error'] = state.get('error', '') + f"\nSummarization Error: {e}"
        
        return state
    
    def compile(self):
        """Compile the workflow"""
        self.app = self.workflow.compile()
        logger.info("✓ Workflow compiled")
        return self.app
    
    def run(self, data_path: str = None, data: pd.DataFrame = None) -> Dict:
        """Run the complete multi-agent workflow"""
        
        if self.app is None:
            self.compile()
        
        # Initialize state
        initial_state = {
            'raw_data_path': data_path,
            'data': data,
            'messages': []
        }
        
        logger.info("🚀 Starting FinAgent Multi-Agent Workflow...")
        
        # Run the workflow
        final_state = self.app.invoke(initial_state)
        
        logger.info("✅ Workflow completed!")
        
        # Print messages
        for msg in final_state.get('messages', []):
            print(msg)
        
        return final_state


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample transaction data
    np.random.seed(42)
    n_samples = 500
    
    sample_data = pd.DataFrame({
        'transaction_id': range(n_samples),
        'customer_id': np.random.randint(1, 50, n_samples),
        'amount': np.random.exponential(100, n_samples),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell'], n_samples),
        'category': np.random.choice(['Online', 'Retail', 'Food', 'Gas'], n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    # Run orchestrator
    orchestrator = FinAgentOrchestrator()
    result = orchestrator.run(data=sample_data)
    
    # Display results
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(result.get('summary', 'No summary available'))
    
    if result.get('error'):
        print(f"\n⚠️  Errors encountered: {result['error']}")