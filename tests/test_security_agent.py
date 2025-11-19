"""
Tests for the Security Agent
"""
import unittest
import pandas as pd
import sys
import os

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from security_agent import SecurityAgent


class TestSecurityAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test security agent"""
        self.agent = SecurityAgent()
    
    def test_encryption_decryption(self):
        """Test encryption and decryption functionality"""
        original_data = "This is sensitive financial data"
        
        # Test encryption
        encrypted_data = self.agent.encrypt_data(original_data)
        self.assertIsInstance(encrypted_data, str)
        self.assertNotEqual(encrypted_data, original_data)
        
        # Test decryption
        decrypted_data = self.agent.decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, original_data)
    
    def test_hash_data(self):
        """Test data hashing functionality"""
        data = "test data"
        hash1 = self.agent.hash_data(data)
        hash2 = self.agent.hash_data(data)
        
        # Hash should be consistent
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hashes
        hash3 = self.agent.hash_data("different data")
        self.assertNotEqual(hash1, hash3)
        
        # Hash should be 64 characters (SHA-256)
        self.assertEqual(len(hash1), 64)
    
    def test_hmac_generation_and_verification(self):
        """Test HMAC generation and verification"""
        data = "test data"
        key = "secret_key"
        
        # Generate HMAC
        signature = self.agent.generate_hmac(data, key)
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA-256 hex digest
        
        # Verify valid signature
        is_valid = self.agent.verify_hmac(data, key, signature)
        self.assertTrue(is_valid)
        
        # Verify invalid signature
        is_invalid = self.agent.verify_hmac(data, key, "invalid_signature")
        self.assertFalse(is_invalid)
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking"""
        # Create test data
        test_data = pd.DataFrame({
            'account_number': ['1234567890123456', '9876543210987654'],
            'ssn': ['123-45-6789', '987-65-4321'],
            'amount': [1000.00, 2000.00]
        })
        
        # Mask sensitive columns
        masked_data = self.agent.mask_sensitive_data(test_data, ['account_number', 'ssn'])
        
        # Check that sensitive data is masked
        self.assertEqual(masked_data.iloc[0]['account_number'], '****************3456')
        self.assertEqual(masked_data.iloc[0]['ssn'], '*******6789')
        
        # Check that non-sensitive data is unchanged
        self.assertEqual(masked_data.iloc[0]['amount'], 1000.00)
    
    def test_mask_value(self):
        """Test individual value masking"""
        # Test long value
        masked = self.agent._mask_value("1234567890123456")
        self.assertEqual(masked, "****************3456")
        
        # Test short value
        masked = self.agent._mask_value("123")
        self.assertEqual(masked, "***")
        
        # Test empty value
        masked = self.agent._mask_value("")
        self.assertEqual(masked, "")
    
    def test_access_logging(self):
        """Test access logging functionality"""
        initial_log_count = len(self.agent.access_log)
        
        # Log an access
        self.agent.log_access("user_123", "transaction_data", "READ")
        
        # Check that log was added
        self.assertEqual(len(self.agent.access_log), initial_log_count + 1)
        
        # Check log content
        last_log = self.agent.access_log[-1]
        self.assertEqual(last_log['user_id'], "user_123")
        self.assertEqual(last_log['resource'], "transaction_data")
        self.assertEqual(last_log['action'], "READ")
        self.assertTrue(last_log['success'])
        self.assertIn('timestamp', last_log)
    
    def test_compliance_checking(self):
        """Test compliance checking functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'amount': [1000.00, 2000.00]
        })
        
        # Check compliance
        regulations = ['gdpr', 'pci_dss', 'sox']
        compliance_report = self.agent.check_compliance(test_data, regulations)
        
        # Check report structure
        self.assertIsInstance(compliance_report, dict)
        self.assertEqual(len(compliance_report), len(regulations))
        
        # Check individual regulation reports
        for reg in regulations:
            self.assertIn(reg, compliance_report)
            self.assertIn('status', compliance_report[reg])
            self.assertIn('checked_at', compliance_report[reg])
    
    def test_audit_report_generation(self):
        """Test audit report generation"""
        # Generate some access logs
        self.agent.log_access("user_123", "transaction_data", "READ")
        self.agent.log_access("user_456", "customer_data", "WRITE")
        
        # Generate audit report
        audit_report = self.agent.generate_audit_report()
        
        # Check report structure
        self.assertIsInstance(audit_report, dict)
        self.assertIn('generated_at', audit_report)
        self.assertIn('total_access_logs', audit_report)
        self.assertIn('recent_access_logs', audit_report)
        self.assertIn('encryption_key_hash', audit_report)
        self.assertIn('compliance_status', audit_report)
        
        # Check specific values
        self.assertEqual(audit_report['total_access_logs'], 2)
        self.assertEqual(len(audit_report['recent_access_logs']), 2)
    
    def test_key_rotation(self):
        """Test encryption key rotation"""
        original_key = self.agent.encryption_key
        
        # Rotate key
        old_key = self.agent.rotate_encryption_key()
        
        # Check that key was rotated
        self.assertNotEqual(self.agent.encryption_key, original_key)
        self.assertEqual(old_key, original_key)
        
        # Check that new key works
        test_data = "test data"
        encrypted = self.agent.encrypt_data(test_data)
        decrypted = self.agent.decrypt_data(encrypted)
        self.assertEqual(decrypted, test_data)


if __name__ == '__main__':
    unittest.main()