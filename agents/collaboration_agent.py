"""
Collaboration Agent: Manages team workspaces, shared analyses, and collaborative workflows
"""
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import hashlib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamWorkspace:
    """Represents a team workspace with shared resources and collaboration features"""
    
    def __init__(self, workspace_id: str, workspace_name: str, owner_id: str, workspaces_dir: str = "team_workspaces"):
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name
        self.owner_id = owner_id
        self.workspaces_dir = Path(workspaces_dir)
        self.workspaces_dir.mkdir(exist_ok=True)
        self.workspace_file = self.workspaces_dir / f"{workspace_id}.json"
        self.workspace_data = self._load_workspace()
    
    def _load_workspace(self) -> Dict:
        """Load workspace from file or create new one"""
        if self.workspace_file.exists():
            try:
                with open(self.workspace_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading workspace {self.workspace_id}: {e}")
                return self._create_default_workspace()
        else:
            return self._create_default_workspace()
    
    def _create_default_workspace(self) -> Dict:
        """Create a default workspace"""
        return {
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "owner_id": self.owner_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "members": [self.owner_id],
            "shared_analyses": [],
            "comments": [],
            "shared_visualizations": [],
            "collaborative_rules": [],
            "activity_log": []
        }
    
    def save_workspace(self):
        """Save workspace to file"""
        try:
            self.workspace_data["last_updated"] = datetime.now().isoformat()
            with open(self.workspace_file, 'w') as f:
                json.dump(self.workspace_data, f, indent=2)
            logger.info(f"Saved workspace {self.workspace_id}")
        except Exception as e:
            logger.error(f"Error saving workspace {self.workspace_id}: {e}")
    
    def add_member(self, user_id: str):
        """Add a member to the workspace"""
        if user_id not in self.workspace_data["members"]:
            self.workspace_data["members"].append(user_id)
            self._log_activity(f"User {user_id} added to workspace", user_id)
            self.save_workspace()
    
    def remove_member(self, user_id: str):
        """Remove a member from the workspace"""
        if user_id in self.workspace_data["members"]:
            self.workspace_data["members"].remove(user_id)
            self._log_activity(f"User {user_id} removed from workspace", user_id)
            self.save_workspace()
    
    def share_analysis(self, analysis_data: Dict, shared_by: str):
        """Share an analysis with the workspace"""
        shared_analysis = {
            "analysis_id": hashlib.md5(str(datetime.now().isoformat()).encode()).hexdigest()[:8],
            "data": analysis_data,
            "shared_by": shared_by,
            "shared_at": datetime.now().isoformat(),
            "comments": []
        }
        self.workspace_data["shared_analyses"].append(shared_analysis)
        self._log_activity(f"Analysis shared by {shared_by}", shared_by)
        self.save_workspace()
        return shared_analysis["analysis_id"]
    
    def add_comment(self, analysis_id: str, user_id: str, comment: str):
        """Add a comment to a shared analysis"""
        for analysis in self.workspace_data["shared_analyses"]:
            if analysis["analysis_id"] == analysis_id:
                comment_entry = {
                    "comment_id": hashlib.md5(str(datetime.now().isoformat()).encode()).hexdigest()[:8],
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
                analysis["comments"].append(comment_entry)
                self._log_activity(f"Comment added to analysis {analysis_id}", user_id)
                self.save_workspace()
                return comment_entry["comment_id"]
        return None
    
    def share_visualization(self, viz_data: Dict, shared_by: str):
        """Share a visualization with the workspace"""
        shared_viz = {
            "viz_id": hashlib.md5(str(datetime.now().isoformat()).encode()).hexdigest()[:8],
            "data": viz_data,
            "shared_by": shared_by,
            "shared_at": datetime.now().isoformat()
        }
        self.workspace_data["shared_visualizations"].append(shared_viz)
        self._log_activity(f"Visualization shared by {shared_by}", shared_by)
        self.save_workspace()
        return shared_viz["viz_id"]
    
    def add_collaborative_rule(self, rule: Dict, created_by: str):
        """Add a collaborative rule to the workspace"""
        rule_entry = {
            "rule_id": hashlib.md5(str(datetime.now().isoformat()).encode()).hexdigest()[:8],
            "rule": rule,
            "created_by": created_by,
            "created_at": datetime.now().isoformat()
        }
        self.workspace_data["collaborative_rules"].append(rule_entry)
        self._log_activity(f"Rule added by {created_by}", created_by)
        self.save_workspace()
        return rule_entry["rule_id"]
    
    def _log_activity(self, activity: str, user_id: str):
        """Log workspace activity"""
        activity_entry = {
            "activity": activity,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        self.workspace_data["activity_log"].append(activity_entry)
        # Keep only the last 100 activities
        if len(self.workspace_data["activity_log"]) > 100:
            self.workspace_data["activity_log"] = self.workspace_data["activity_log"][-100:]


class CollaborationAgent:
    """Agent for managing team collaboration and shared workspaces"""
    
    def __init__(self, workspaces_dir: str = "team_workspaces"):
        self.workspaces_dir = Path(workspaces_dir)
        self.workspaces_dir.mkdir(exist_ok=True)
        self.active_workspaces: Dict[str, TeamWorkspace] = {}
    
    def create_workspace(self, workspace_name: str, owner_id: str) -> str:
        """Create a new team workspace"""
        workspace_id = hashlib.md5(f"{workspace_name}_{owner_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        workspace = TeamWorkspace(workspace_id, workspace_name, owner_id, str(self.workspaces_dir))
        self.active_workspaces[workspace_id] = workspace
        logger.info(f"Created workspace {workspace_name} with ID {workspace_id}")
        return workspace_id
    
    def get_workspace(self, workspace_id: str) -> Optional[TeamWorkspace]:
        """Get a workspace by ID"""
        if workspace_id not in self.active_workspaces:
            # Try to load from file
            workspace_file = self.workspaces_dir / f"{workspace_id}.json"
            if workspace_file.exists():
                # Extract workspace name and owner from file
                try:
                    with open(workspace_file, 'r') as f:
                        data = json.load(f)
                        workspace_name = data.get("workspace_name", "Unknown Workspace")
                        owner_id = data.get("owner_id", "unknown_user")
                        workspace = TeamWorkspace(workspace_id, workspace_name, owner_id, str(self.workspaces_dir))
                        self.active_workspaces[workspace_id] = workspace
                        return workspace
                except Exception as e:
                    logger.error(f"Error loading workspace {workspace_id}: {e}")
                    return None
            else:
                return None
        return self.active_workspaces[workspace_id]
    
    def list_workspaces_for_user(self, user_id: str) -> List[Dict]:
        """List all workspaces a user is a member of"""
        workspaces = []
        for file_path in self.workspaces_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if user_id in data.get("members", []):
                        workspaces.append({
                            "workspace_id": data["workspace_id"],
                            "workspace_name": data["workspace_name"],
                            "owner_id": data["owner_id"],
                            "created_at": data["created_at"],
                            "member_count": len(data.get("members", []))
                        })
            except Exception as e:
                logger.warning(f"Error reading workspace file {file_path}: {e}")
        return workspaces
    
    def share_analysis_with_workspace(self, workspace_id: str, analysis_data: Dict, shared_by: str) -> Optional[str]:
        """Share an analysis with a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.share_analysis(analysis_data, shared_by)
        return None
    
    def add_comment_to_analysis(self, workspace_id: str, analysis_id: str, user_id: str, comment: str) -> Optional[str]:
        """Add a comment to a shared analysis"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.add_comment(analysis_id, user_id, comment)
        return None
    
    def share_visualization_with_workspace(self, workspace_id: str, viz_data: Dict, shared_by: str) -> Optional[str]:
        """Share a visualization with a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.share_visualization(viz_data, shared_by)
        return None
    
    def add_collaborative_rule(self, workspace_id: str, rule: Dict, created_by: str) -> Optional[str]:
        """Add a collaborative rule to a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.add_collaborative_rule(rule, created_by)
        return None
    
    def get_workspace_analyses(self, workspace_id: str) -> List[Dict]:
        """Get all shared analyses in a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.workspace_data.get("shared_analyses", [])
        return []
    
    def get_workspace_visualizations(self, workspace_id: str) -> List[Dict]:
        """Get all shared visualizations in a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.workspace_data.get("shared_visualizations", [])
        return []
    
    def get_workspace_activity_log(self, workspace_id: str) -> List[Dict]:
        """Get activity log for a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.workspace_data.get("activity_log", [])
        return []
    
    def add_member_to_workspace(self, workspace_id: str, user_id: str):
        """Add a member to a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            workspace.add_member(user_id)
    
    def remove_member_from_workspace(self, workspace_id: str, user_id: str):
        """Remove a member from a workspace"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            workspace.remove_member(user_id)


# Example usage
if __name__ == "__main__":
    # Create collaboration agent
    agent = CollaborationAgent()
    
    # Create a workspace
    workspace_id = agent.create_workspace("Fraud Detection Team", "user_123")
    print(f"Created workspace with ID: {workspace_id}")
    
    # Get workspace
    workspace = agent.get_workspace(workspace_id)
    if workspace:
        # Add members
        workspace.add_member("user_456")
        workspace.add_member("user_789")
        
        # Share an analysis
        sample_analysis = {
            "total_transactions": 1000,
            "anomalies_detected": 25,
            "risk_score": 0.75,
            "summary": "High-risk transactions identified in Q1 data"
        }
        analysis_id = agent.share_analysis_with_workspace(workspace_id, sample_analysis, "user_123")
        print(f"Shared analysis with ID: {analysis_id}")
        
        # Add a comment
        if analysis_id:
            comment_id = agent.add_comment_to_analysis(workspace_id, analysis_id, "user_456", "This looks interesting, let's investigate further")
            print(f"Added comment with ID: {comment_id}")
        
        # List workspaces for a user
        workspaces = agent.list_workspaces_for_user("user_123")
        print(f"User is member of {len(workspaces)} workspaces")
        
        # Get workspace activity
        activity = agent.get_workspace_activity_log(workspace_id)
        print(f"Workspace activity log has {len(activity)} entries")