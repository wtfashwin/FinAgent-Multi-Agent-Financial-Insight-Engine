"""
Security Agent: Handles data encryption, access control, and compliance monitoring
"""
import hashlib
import hmac
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityAgent:
    """Agent for handling security, encryption, and compliance"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.access_log = []
        self.compliance_rules = self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self) -> Dict:
        """Initialize compliance rules for financial data"""
        return {
            'gdpr': {
                'data_retention_days': 365,
                'requires_consent': True,
                'encryption_required': True
            },
            'pci_dss': {
                'card_data_encryption': True,
                'access_logging': True,
                'regular_audits': True
            },
            'sox': {
                'data_integrity': True,
                'access_controls': True,
                'audit_trail': True
            }
        }
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data
        """
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def hash_data(self, data: str) -> str:
        """
        Create a secure hash of data (for data integrity checks)
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def generate_hmac(self, data: str, key: str) -> str:
        """
        Generate HMAC for data authentication
        """
        return hmac.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()
    
    def verify_hmac(self, data: str, key: str, signature: str) -> bool:
        """
        Verify HMAC signature
        """
        expected_signature = self.generate_hmac(data, key)
        return hmac.compare_digest(expected_signature, signature)
    
    def mask_sensitive_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Mask sensitive data in a DataFrame
        """
        masked_df = df.copy()
        for col in columns:
            if col in masked_df.columns:
                # Mask all but last 4 characters for string columns
                masked_df[col] = masked_df[col].apply(
                    lambda x: self._mask_value(str(x)) if pd.notnull(x) else x
                )
        return masked_df
    
    def _mask_value(self, value: str) -> str:
        """
        Mask a sensitive value
        """
        if len(value) <= 4:
            return '*' * len(value)
        return '*' * (len(value) - 4) + value[-4:]
    
    def log_access(self, user_id: str, resource: str, action: str, success: bool = True):
        """
        Log access to sensitive resources
        """
        access_record = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'ip_address': '127.0.0.1'  # In a real implementation, this would be the actual IP
        }
        self.access_log.append(access_record)
        logger.info(f"Access logged: {user_id} {action} {resource} - {'SUCCESS' if success else 'FAILED'}")
    
    def check_compliance(self, data: pd.DataFrame, regulations: List[str]) -> Dict:
        """
        Check if data handling complies with specified regulations
        """
        compliance_report = {}
        
        for regulation in regulations:
            if regulation in self.compliance_rules:
                rules = self.compliance_rules[regulation]
                compliance_report[regulation] = self._check_regulation_compliance(data, rules)
            else:
                compliance_report[regulation] = {
                    'status': 'UNKNOWN',
                    'details': f'Regulation {regulation} not supported'
                }
        
        return compliance_report
    
    def _check_regulation_compliance(self, data: pd.DataFrame, rules: Dict) -> Dict:
        """
        Check compliance with specific regulation rules
        """
        violations = []
        compliant = True
        
        # Check encryption requirement
        if rules.get('encryption_required', False):
            # In a real implementation, we would check if sensitive columns are encrypted
            pass  # Placeholder for encryption check
        
        # Check data retention
        if 'data_retention_days' in rules:
            # In a real implementation, we would check data creation dates
            pass  # Placeholder for retention check
        
        # Check access logging
        if rules.get('access_logging', False) and not self.access_log:
            violations.append("Access logging required but no logs found")
            compliant = False
        
        return {
            'status': 'COMPLIANT' if compliant else 'NON-COMPLIANT',
            'violations': violations,
            'checked_at': datetime.now().isoformat()
        }
    
    def generate_audit_report(self) -> Dict:
        """
        Generate a security audit report
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'total_access_logs': len(self.access_log),
            'recent_access_logs': self.access_log[-10:],  # Last 10 access logs
            'encryption_key_hash': self.hash_data(self.encryption_key.decode()),
            'compliance_status': self._get_compliance_status()
        }
    
    def _get_compliance_status(self) -> Dict:
        """
        Get overall compliance status
        """
        # In a real implementation, this would check actual compliance
        return {
            'gdpr': 'COMPLIANT',
            'pci_dss': 'COMPLIANT',
            'sox': 'COMPLIANT'
        }
    
    def rotate_encryption_key(self) -> bytes:
        """
        Rotate the encryption key for enhanced security
        """
        old_key = self.encryption_key
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        logger.info("Encryption key rotated successfully")
        return old_key


# Example usage
if __name__ == "__main__":
    # Create security agent
    agent = SecurityAgent()
    
    # Test encryption/decryption
    sensitive_data = "This is sensitive financial data"
    encrypted = agent.encrypt_data(sensitive_data)
    decrypted = agent.decrypt_data(encrypted)
    print(f"Original: {sensitive_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    
    # Test hashing
    data_hash = agent.hash_data(sensitive_data)
    print(f"Data hash: {data_hash}")
    
    # Test HMAC
    hmac_key = "secret_key"
    signature = agent.generate_hmac(sensitive_data, hmac_key)
    is_valid = agent.verify_hmac(sensitive_data, hmac_key, signature)
    print(f"HMAC signature: {signature}")
    print(f"Signature valid: {is_valid}")
    
    # Test data masking
    sample_data = pd.DataFrame({
        'account_number': ['1234567890123456', '9876543210987654'],
        'ssn': ['123-45-6789', '987-65-4321'],
        'amount': [1000.00, 2000.00]
    })
    
    masked_data = agent.mask_sensitive_data(sample_data, ['account_number', 'ssn'])
    print("\nOriginal data:")
    print(sample_data)
    print("\nMasked data:")
    print(masked_data)
    
    # Test access logging
    agent.log_access("user_123", "transaction_data", "READ")
    agent.log_access("user_456", "customer_data", "WRITE")
    
    # Test compliance checking
    compliance_report = agent.check_compliance(sample_data, ['gdpr', 'pci_dss'])
    print(f"\nCompliance report: {compliance_report}")
    
    # Test audit report
    audit_report = agent.generate_audit_report()
    print(f"\nAudit report generated at: {audit_report['generated_at']}")