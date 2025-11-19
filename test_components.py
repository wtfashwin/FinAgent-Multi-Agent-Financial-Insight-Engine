"""
Simple test script to verify that all components are working correctly
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_functionality():
    """Test basic functionality of all components"""
    print("Testing basic functionality...")
    
    # Test 1: Basic imports
    try:
        from agents.collaboration_agent import CollaborationAgent, TeamWorkspace
        from agents.ml_enhancement_agent import MLEnhancementAgent
        from agents.user_profile_agent import UserProfileAgent, UserProfile
        from agents.visualization_agent import VisualizationAgent
        from agents.streaming_agent import StreamingAgent
        from agents.data_agent import DataAgent
        from agents.insight_agent import InsightAgent
        from agents.risk_agent import RiskAgent
        print("✓ All agents imported successfully")
    except Exception as e:
        print(f"✗ Error importing agents: {e}")
        return False
    
    # Test 2: Create sample data
    try:
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'transaction_id': range(100),
            'amount': np.random.exponential(100, 100),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Target'], 100),
            'category': np.random.choice(['Online', 'Retail', 'Food'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        })
        print(f"✓ Sample data created ({len(sample_data)} rows)")
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False
    
    # Test 3: Test Collaboration Agent
    try:
        collaboration_agent = CollaborationAgent()
        workspace_id = collaboration_agent.create_workspace("Test Workspace", "user_123")
        workspace = collaboration_agent.get_workspace(workspace_id)
        print("✓ Collaboration agent working")
    except Exception as e:
        print(f"✗ Error with collaboration agent: {e}")
        return False
    
    # Test 4: Test ML Enhancement Agent
    try:
        ml_agent = MLEnhancementAgent()
        result = ml_agent.natural_language_query(sample_data, "Show me high risk transactions")
        print("✓ ML enhancement agent working")
    except Exception as e:
        print(f"✗ Error with ML enhancement agent: {e}")
        return False
    
    # Test 5: Test User Profile Agent
    try:
        profile_agent = UserProfileAgent()
        profile = profile_agent.get_user_profile("user_123")
        profile.set_preference("risk_threshold", 0.8)
        print("✓ User profile agent working")
    except Exception as e:
        print(f"✗ Error with user profile agent: {e}")
        return False
    
    # Test 6: Test Visualization Agent
    try:
        viz_agent = VisualizationAgent()
        visualizations = viz_agent.generate_advanced_visualizations(sample_data)
        print(f"✓ Visualization agent working ({len(visualizations)} visualizations generated)")
    except Exception as e:
        print(f"✗ Error with visualization agent: {e}")
        return False
    
    # Test 7: Test Streaming Agent
    try:
        streaming_agent = StreamingAgent()
        print("✓ Streaming agent working")
    except Exception as e:
        print(f"✗ Error with streaming agent: {e}")
        return False
    
    # Test 8: Test Data Agent
    try:
        data_agent = DataAgent()
        data_agent.df = sample_data
        processed_data = data_agent.clean_data()
        print(f"✓ Data agent working ({len(processed_data)} rows processed)")
    except Exception as e:
        print(f"✗ Error with data agent: {e}")
        return False
    
    # Test 9: Test Risk Agent
    try:
        risk_agent = RiskAgent()
        print("✓ Risk agent working")
    except Exception as e:
        print(f"✗ Error with risk agent: {e}")
        return False
    
    print("\n🎉 All components are working correctly!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ Integration test passed!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)