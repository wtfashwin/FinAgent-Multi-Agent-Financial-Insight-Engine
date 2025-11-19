"""
User Profile Agent: Manages user preferences, history, and personalization
"""
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserProfile:
    """Represents a user's profile with preferences and history"""
    
    def __init__(self, user_id: str, profiles_dir: str = "user_profiles"):
        self.user_id = user_id
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.profile_file = self.profiles_dir / f"{user_id}.json"
        self.profile_data = self._load_profile()
    
    def _load_profile(self) -> Dict:
        """Load user profile from file or create new one"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading profile for {self.user_id}: {e}")
                return self._create_default_profile()
        else:
            return self._create_default_profile()
    
    def _create_default_profile(self) -> Dict:
        """Create a default user profile"""
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "preferences": {
                "risk_threshold": 0.7,
                "high_risk_threshold": 0.5,
                "visualization_style": "default",
                "report_format": "summary",
                "notification_preferences": {
                    "email": False,
                    "push": True,
                    "sms": False
                }
            },
            "history": {
                "recent_analyses": [],
                "favorite_queries": [],
                "saved_reports": [],
                "viewed_visualizations": []
            },
            "custom_rules": [],
            "watchlist": []
        }
    
    def save_profile(self):
        """Save user profile to file"""
        try:
            self.profile_data["last_updated"] = datetime.now().isoformat()
            with open(self.profile_file, 'w') as f:
                json.dump(self.profile_data, f, indent=2)
            logger.info(f"Saved profile for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving profile for {self.user_id}: {e}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference value"""
        keys = key.split('.')
        value = self.profile_data["preferences"]
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference value"""
        keys = key.split('.')
        prefs = self.profile_data["preferences"]
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in prefs:
                prefs[k] = {}
            prefs = prefs[k]
        
        # Set the final value
        prefs[keys[-1]] = value
        self.save_profile()
    
    def add_to_history(self, category: str, item: Dict):
        """Add an item to user history"""
        if category in self.profile_data["history"]:
            self.profile_data["history"][category].append(item)
            # Keep only the last 50 items
            if len(self.profile_data["history"][category]) > 50:
                self.profile_data["history"][category] = self.profile_data["history"][category][-50:]
            self.save_profile()
    
    def get_history(self, category: str) -> List[Dict]:
        """Get user history for a category"""
        return self.profile_data["history"].get(category, [])
    
    def add_custom_rule(self, rule: Dict):
        """Add a custom rule to user profile"""
        self.profile_data["custom_rules"].append(rule)
        self.save_profile()
    
    def get_custom_rules(self) -> List[Dict]:
        """Get all custom rules"""
        return self.profile_data["custom_rules"]
    
    def add_to_watchlist(self, item: Dict):
        """Add an item to user's watchlist"""
        self.profile_data["watchlist"].append(item)
        self.save_profile()
    
    def get_watchlist(self) -> List[Dict]:
        """Get user's watchlist"""
        return self.profile_data["watchlist"]


class UserProfileAgent:
    """Agent for managing user profiles and personalization"""
    
    def __init__(self, profiles_dir: str = "user_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.active_profiles: Dict[str, UserProfile] = {}
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile"""
        if user_id not in self.active_profiles:
            self.active_profiles[user_id] = UserProfile(user_id, str(self.profiles_dir))
        return self.active_profiles[user_id]
    
    def personalize_analysis(self, user_id: str, analysis_result: Dict) -> Dict:
        """
        Personalize analysis results based on user preferences
        """
        profile = self.get_user_profile(user_id)
        
        # Apply user's risk thresholds
        risk_threshold = profile.get_preference("risk_threshold", 0.7)
        high_risk_threshold = profile.get_preference("high_risk_threshold", 0.5)
        
        personalized_result = analysis_result.copy()
        
        # Adjust risk assessment based on user preferences
        if "risk_assessment" in personalized_result:
            risk_assessment = personalized_result["risk_assessment"]
            if "summary" in risk_assessment:
                summary = risk_assessment["summary"]
                # Adjust thresholds in the summary
                # Note: This is a simplified example - in practice, you would re-run
                # the risk assessment with the user's thresholds
                pass
        
        return personalized_result
    
    def get_personalized_visualization_preferences(self, user_id: str) -> Dict:
        """Get user's visualization preferences"""
        profile = self.get_user_profile(user_id)
        return {
            "style": profile.get_preference("visualization_style", "default"),
            "report_format": profile.get_preference("report_format", "summary")
        }
    
    def save_analysis_to_history(self, user_id: str, analysis_data: Dict, analysis_type: str):
        """Save analysis to user's history"""
        profile = self.get_user_profile(user_id)
        
        history_item = {
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_transactions": analysis_data.get("total_transactions", 0),
                "anomalies_detected": analysis_data.get("anomalies_detected", 0),
                "risk_score": analysis_data.get("average_risk_score", 0)
            }
        }
        
        profile.add_to_history("recent_analyses", history_item)
    
    def get_user_dashboard_data(self, user_id: str) -> Dict:
        """Get data for user's personalized dashboard"""
        profile = self.get_user_profile(user_id)
        
        return {
            "user_id": user_id,
            "preferences": profile.profile_data["preferences"],
            "recent_analyses": profile.get_history("recent_analyses")[-5:],  # Last 5 analyses
            "favorite_queries": profile.get_history("favorite_queries")[-10:],  # Last 10 queries
            "watchlist": profile.get_watchlist(),
            "custom_rules": profile.get_custom_rules()
        }
    
    def add_favorite_query(self, user_id: str, query: str):
        """Add a query to user's favorites"""
        profile = self.get_user_profile(user_id)
        
        favorite_item = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "usage_count": 1
        }
        
        # Check if query already exists
        favorites = profile.get_history("favorite_queries")
        existing_item = None
        for item in favorites:
            if item.get("query") == query:
                existing_item = item
                break
        
        if existing_item:
            existing_item["usage_count"] += 1
            existing_item["timestamp"] = datetime.now().isoformat()
        else:
            profile.add_to_history("favorite_queries", favorite_item)
    
    def generate_user_hash(self, user_identifier: str) -> str:
        """Generate a hash for user identification"""
        return hashlib.md5(user_identifier.encode()).hexdigest()


# Example usage
if __name__ == "__main__":
    # Create user profile agent
    agent = UserProfileAgent()
    
    # Get a user profile
    user_id = "user_123"
    profile = agent.get_user_profile(user_id)
    
    # Set some preferences
    profile.set_preference("risk_threshold", 0.8)
    profile.set_preference("visualization_style", "detailed")
    
    # Add to history
    profile.add_to_history("recent_analyses", {
        "analysis_type": "fraud_detection",
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_transactions": 1000,
            "anomalies_detected": 25,
            "risk_score": 0.75
        }
    })
    
    # Get dashboard data
    dashboard_data = agent.get_user_dashboard_data(user_id)
    print(f"Dashboard data for {user_id}:")
    print(json.dumps(dashboard_data, indent=2))
    
    # Test personalization
    sample_analysis = {
        "risk_assessment": {
            "summary": {
                "high_risk_transactions": 10,
                "medium_risk_transactions": 20,
                "low_risk_transactions": 70
            }
        }
    }
    
    personalized = agent.personalize_analysis(user_id, sample_analysis)
    print(f"\nPersonalized analysis: {personalized}")