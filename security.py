"""
Security utilities for Financial P&L Anomaly Detection Agent
Provides input validation, sanitization, and rate limiting
"""

import re
import os
from pathlib import Path
from typing import Optional
import logging
import time
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_extensions: Optional[list] = None) -> Path:
        """
        Validate and sanitize file path to prevent path traversal attacks
        
        Args:
            file_path: Path to validate
            allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.txt'])
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        # Convert to Path object
        path = Path(file_path).resolve()
        
        # Check if path exists
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Prevent directory traversal
        try:
            # Get the working directory
            working_dir = Path.cwd().resolve()
            # Check if the resolved path is within working directory or subdirectories
            path.relative_to(working_dir)
        except ValueError:
            # If relative_to fails, check if it's an absolute path within allowed locations
            # For now, we'll be strict and only allow paths within working directory
            raise ValueError(f"File path is outside allowed directory: {file_path}")
        
        # Check file extension if specified
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Invalid file extension. Allowed: {allowed_extensions}, got: {path.suffix}")
        
        # Check if it's a file (not a directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        logger.debug(f"✅ File path validated: {path}")
        return path
    
    @staticmethod
    def validate_month_format(month: str) -> str:
        """
        Validate month format is YYYY-MM
        
        Args:
            month: Month string to validate
            
        Returns:
            Validated month string
            
        Raises:
            ValueError: If format is invalid
        """
        pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
        if not re.match(pattern, month):
            raise ValueError(f"Invalid month format. Expected YYYY-MM, got: {month}")
        
        # Additional validation: parse as date
        try:
            datetime.strptime(f"{month}-01", "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid month: {month}. Error: {e}")
        
        return month
    
    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """
        Sanitize SQL identifier (table/column name) to prevent SQL injection
        Only allows alphanumeric characters and underscores
        
        Args:
            identifier: SQL identifier to sanitize
            
        Returns:
            Sanitized identifier
            
        Raises:
            ValueError: If identifier contains invalid characters
        """
        if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}. Only alphanumeric and underscore allowed")
        
        return identifier
    
    @staticmethod
    def mask_sensitive_data(text: str, patterns: Optional[list] = None) -> str:
        """
        Mask sensitive data in text for logging
        
        Args:
            text: Text to mask
            patterns: List of regex patterns to mask (default: API keys, credit cards)
            
        Returns:
            Text with sensitive data masked
        """
        if patterns is None:
            patterns = [
                (r'sk-[a-zA-Z0-9]{48}', 'sk-***'),  # OpenAI API keys
                (r'Bearer [a-zA-Z0-9_\-\.]+', 'Bearer ***'),  # Bearer tokens
                (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '****-****-****-****'),  # Credit cards
                (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),  # SSN
            ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, replacement, masked_text)
        
        return masked_text


class RateLimiter:
    """
    Rate limiter for API calls to prevent excessive costs and API abuse
    Uses token bucket algorithm
    """
    
    def __init__(self, max_calls_per_minute: int = 60, max_calls_per_hour: int = 1000):
        """
        Initialize rate limiter
        
        Args:
            max_calls_per_minute: Maximum API calls per minute
            max_calls_per_hour: Maximum API calls per hour
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour
        
        self.minute_calls = deque()
        self.hour_calls = deque()
        
        logger.info(f"Rate limiter initialized: {max_calls_per_minute}/min, {max_calls_per_hour}/hour")
    
    def acquire(self) -> bool:
        """
        Check if a new API call is allowed under rate limits
        
        Returns:
            True if call is allowed, False otherwise
        """
        now = datetime.now()
        
        # Clean up old calls
        self._cleanup_old_calls(now)
        
        # Check minute limit
        if len(self.minute_calls) >= self.max_calls_per_minute:
            wait_time = 60 - (now - self.minute_calls[0]).total_seconds()
            logger.warning(f"⚠️  Rate limit reached: {self.max_calls_per_minute} calls/min. Wait {wait_time:.1f}s")
            return False
        
        # Check hour limit
        if len(self.hour_calls) >= self.max_calls_per_hour:
            wait_time = 3600 - (now - self.hour_calls[0]).total_seconds()
            logger.warning(f"⚠️  Rate limit reached: {self.max_calls_per_hour} calls/hour. Wait {wait_time:.1f}s")
            return False
        
        # Record this call
        self.minute_calls.append(now)
        self.hour_calls.append(now)
        
        return True
    
    def wait_if_needed(self):
        """Block until a call is allowed under rate limits"""
        while not self.acquire():
            time.sleep(1)
    
    def _cleanup_old_calls(self, now: datetime):
        """Remove call timestamps older than tracking window"""
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Clean minute window
        while self.minute_calls and self.minute_calls[0] < minute_ago:
            self.minute_calls.popleft()
        
        # Clean hour window
        while self.hour_calls and self.hour_calls[0] < hour_ago:
            self.hour_calls.popleft()
    
    def get_stats(self) -> dict:
        """Get current rate limit statistics"""
        now = datetime.now()
        self._cleanup_old_calls(now)
        
        return {
            "calls_last_minute": len(self.minute_calls),
            "calls_last_hour": len(self.hour_calls),
            "minute_limit": self.max_calls_per_minute,
            "hour_limit": self.max_calls_per_hour,
            "minute_remaining": self.max_calls_per_minute - len(self.minute_calls),
            "hour_remaining": self.max_calls_per_hour - len(self.hour_calls)
        }


class APIKeyValidator:
    """Validate API keys before use"""
    
    @staticmethod
    def validate_openai_key(api_key: str) -> bool:
        """
        Validate OpenAI API key format
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format, False otherwise
        """
        if not api_key:
            return False
        
        # OpenAI keys start with 'sk-' and are typically 48-51 characters
        pattern = r'^sk-[a-zA-Z0-9]{48,}$'
        return bool(re.match(pattern, api_key))
    
    @staticmethod
    def validate_keys(keys: dict) -> dict:
        """
        Validate multiple API keys
        
        Args:
            keys: Dictionary of key_name: key_value
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        for name, value in keys.items():
            if not value:
                results[name] = {"valid": False, "error": "Missing"}
                continue
            
            if name == "OPENAI_API_KEY":
                results[name] = {
                    "valid": APIKeyValidator.validate_openai_key(value),
                    "error": None if APIKeyValidator.validate_openai_key(value) else "Invalid format"
                }
            else:
                # Generic validation - just check it's not empty
                results[name] = {"valid": bool(value), "error": None}
        
        return results


# Convenience function for checking all security configurations
def validate_security_config() -> bool:
    """
    Validate security configuration on startup
    
    Returns:
        True if all checks pass, False otherwise
    """
    from config import Config
    
    logger.info("🔒 Validating security configuration...")
    
    issues = []
    
    # Check API key
    if not APIKeyValidator.validate_openai_key(Config.OPENAI_API_KEY):
        issues.append("Invalid OpenAI API key format")
    
    # Check if running in production
    if Config.ENVIRONMENT == "production":
        # More strict checks for production
        if not Config.MASK_SENSITIVE_DATA:
            issues.append("MASK_SENSITIVE_DATA should be enabled in production")
        
        if not Config.ENABLE_AUDIT_LOG:
            issues.append("ENABLE_AUDIT_LOG should be enabled in production")
        
        if Config.LOG_LEVEL == "DEBUG":
            issues.append("LOG_LEVEL should not be DEBUG in production")
    
    # Check file permissions on sensitive files
    if os.path.exists(".env"):
        env_stat = os.stat(".env")
        if env_stat.st_mode & 0o077:  # Check if group/other can read
            issues.append(".env file has insecure permissions (should be 600)")
    
    if issues:
        logger.warning("⚠️  Security issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("✅ Security configuration validated")
    return True


if __name__ == "__main__":
    # Test security utilities
    logging.basicConfig(level=logging.INFO)
    
    # Test input validation
    validator = InputValidator()
    
    # Test month validation
    try:
        validator.validate_month_format("2025-03")
        print("✅ Month validation passed")
    except ValueError as e:
        print(f"❌ Month validation failed: {e}")
    
    # Test sensitive data masking
    text = "My API key is sk-1234567890abcdefghijklmnopqrstuvwxyz123456789012"
    masked = validator.mask_sensitive_data(text)
    print(f"Original: {text}")
    print(f"Masked: {masked}")
    
    # Test rate limiter
    limiter = RateLimiter(max_calls_per_minute=5)
    print(f"\nRate limiter stats: {limiter.get_stats()}")
