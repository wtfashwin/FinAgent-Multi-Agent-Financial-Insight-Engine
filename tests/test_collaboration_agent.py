"""
Tests for the Collaboration Agent
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from collaboration_agent import CollaborationAgent, TeamWorkspace


class TestCollaborationAgent(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.agent = CollaborationAgent(workspaces_dir=self.test_dir)
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_create_workspace(self):
        """Test creating a new workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        self.assertIsNotNone(workspace_id)
        self.assertIsInstance(workspace_id, str)
        self.assertEqual(len(workspace_id), 16)  # MD5 hash truncated to 16 chars
    
    def test_get_workspace(self):
        """Test getting a workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        workspace = self.agent.get_workspace(workspace_id)
        self.assertIsNotNone(workspace)
        self.assertIsInstance(workspace, TeamWorkspace)
        self.assertEqual(workspace.workspace_name, "Test Workspace")
        self.assertEqual(workspace.owner_id, "user_123")
    
    def test_list_workspaces_for_user(self):
        """Test listing workspaces for a user"""
        # Create multiple workspaces
        workspace_id1 = self.agent.create_workspace("Workspace 1", "user_123")
        workspace_id2 = self.agent.create_workspace("Workspace 2", "user_456")
        
        # Ensure workspaces are saved to disk by getting them
        workspace1 = self.agent.get_workspace(workspace_id1)
        workspace2 = self.agent.get_workspace(workspace_id2)
        
        # Add user to second workspace
        workspace2.add_member("user_123")
        
        # List workspaces for user_123
        workspaces = self.agent.list_workspaces_for_user("user_123")
        self.assertEqual(len(workspaces), 2)
        workspace_ids = [w["workspace_id"] for w in workspaces]
        self.assertIn(workspace_id1, workspace_ids)
        self.assertIn(workspace_id2, workspace_ids)
    
    def test_share_analysis_with_workspace(self):
        """Test sharing an analysis with a workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        analysis_data = {
            "total_transactions": 1000,
            "anomalies_detected": 25,
            "risk_score": 0.75
        }
        
        analysis_id = self.agent.share_analysis_with_workspace(workspace_id, analysis_data, "user_123")
        self.assertIsNotNone(analysis_id)
        
        # Verify the analysis was shared
        workspace = self.agent.get_workspace(workspace_id)
        shared_analyses = workspace.workspace_data["shared_analyses"]
        self.assertEqual(len(shared_analyses), 1)
        self.assertEqual(shared_analyses[0]["analysis_id"], analysis_id)
        self.assertEqual(shared_analyses[0]["shared_by"], "user_123")
    
    def test_add_comment_to_analysis(self):
        """Test adding a comment to a shared analysis"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        analysis_data = {"summary": "Test analysis"}
        
        analysis_id = self.agent.share_analysis_with_workspace(workspace_id, analysis_data, "user_123")
        self.assertIsNotNone(analysis_id)
        
        comment_id = self.agent.add_comment_to_analysis(workspace_id, analysis_id, "user_456", "Great analysis!")
        self.assertIsNotNone(comment_id)
        
        # Verify the comment was added
        workspace = self.agent.get_workspace(workspace_id)
        shared_analyses = workspace.workspace_data["shared_analyses"]
        self.assertEqual(len(shared_analyses[0]["comments"]), 1)
        self.assertEqual(shared_analyses[0]["comments"][0]["comment_id"], comment_id)
        self.assertEqual(shared_analyses[0]["comments"][0]["user_id"], "user_456")
        self.assertEqual(shared_analyses[0]["comments"][0]["comment"], "Great analysis!")
    
    def test_share_visualization_with_workspace(self):
        """Test sharing a visualization with a workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        viz_data = {
            "chart_type": "bar",
            "data": [1, 2, 3, 4, 5]
        }
        
        viz_id = self.agent.share_visualization_with_workspace(workspace_id, viz_data, "user_123")
        self.assertIsNotNone(viz_id)
        
        # Verify the visualization was shared
        workspace = self.agent.get_workspace(workspace_id)
        shared_visualizations = workspace.workspace_data["shared_visualizations"]
        self.assertEqual(len(shared_visualizations), 1)
        self.assertEqual(shared_visualizations[0]["viz_id"], viz_id)
        self.assertEqual(shared_visualizations[0]["shared_by"], "user_123")
    
    def test_add_collaborative_rule(self):
        """Test adding a collaborative rule to a workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        rule_data = {
            "rule_name": "High Amount Alert",
            "condition": "amount > 10000",
            "action": "send_notification"
        }
        
        rule_id = self.agent.add_collaborative_rule(workspace_id, rule_data, "user_123")
        self.assertIsNotNone(rule_id)
        
        # Verify the rule was added
        workspace = self.agent.get_workspace(workspace_id)
        collaborative_rules = workspace.workspace_data["collaborative_rules"]
        self.assertEqual(len(collaborative_rules), 1)
        self.assertEqual(collaborative_rules[0]["rule_id"], rule_id)
        self.assertEqual(collaborative_rules[0]["created_by"], "user_123")
        self.assertEqual(collaborative_rules[0]["rule"], rule_data)
    
    def test_add_and_remove_members(self):
        """Test adding and removing members from a workspace"""
        workspace_id = self.agent.create_workspace("Test Workspace", "user_123")
        workspace = self.agent.get_workspace(workspace_id)
        
        # Add a member
        workspace.add_member("user_456")
        self.assertIn("user_456", workspace.workspace_data["members"])
        
        # Remove a member
        workspace.remove_member("user_456")
        self.assertNotIn("user_456", workspace.workspace_data["members"])
    
    def test_workspace_persistence(self):
        """Test that workspace data persists to disk"""
        workspace_id = self.agent.create_workspace("Persistent Workspace", "user_123")
        workspace = self.agent.get_workspace(workspace_id)
        
        # Add some data
        analysis_data = {"test": "data"}
        workspace.share_analysis(analysis_data, "user_123")
        
        # Create a new agent to simulate app restart
        new_agent = CollaborationAgent(workspaces_dir=self.test_dir)
        loaded_workspace = new_agent.get_workspace(workspace_id)
        
        # Verify data was loaded correctly
        self.assertIsNotNone(loaded_workspace)
        self.assertEqual(len(loaded_workspace.workspace_data["shared_analyses"]), 1)


if __name__ == '__main__':
    unittest.main()