# Security Guidelines and Best Practices

## Overview

This document outlines the security measures implemented in the Financial P&L Anomaly Detection Agent and provides guidelines for secure deployment and usage.

## Security Features

### 1. API Key Protection

**Implemented:**
- API keys loaded from environment variables (`.env` file)
- `.gitignore` configured to prevent `.env` commits
- API key format validation on startup
- Sensitive data masking in logs

**Best Practices:**
```bash
# Set restrictive permissions on .env file
chmod 600 .env

# Use different keys for development and production
ENVIRONMENT=production
OPENAI_API_KEY=sk-prod-xxx  # Production key
```

**Never:**
- Commit API keys to version control
- Share API keys in logs or error messages
- Use production keys in development

### 2. SQL Injection Prevention

**Implemented:**
- Parameterized queries throughout database.py
- Input validation for all user-provided data
- SQL identifier sanitization

**Example:**
```python
# SECURE: Parameterized query
cur.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))

# INSECURE: String formatting (DO NOT USE)
cur.execute(f"SELECT * FROM accounts WHERE id = '{account_id}'")
```

### 3. Input Validation

**Implemented:**
- File path validation (prevents directory traversal)
- Month format validation (YYYY-MM)
- Numeric threshold validation
- Allowed file extensions checking

**Usage:**
```python
from security import InputValidator

# Validate file path
safe_path = InputValidator.validate_file_path(
    user_input,
    allowed_extensions=['.csv']
)

# Validate month format
safe_month = InputValidator.validate_month_format("2025-03")
```

### 4. Rate Limiting

**Implemented:**
- Token bucket algorithm for API calls
- Configurable limits per minute and hour
- Automatic throttling to prevent excessive costs

**Configuration:**
```python
# In agents.py
rate_limiter = RateLimiter(
    max_calls_per_minute=60,
    max_calls_per_hour=1000
)
```

**Benefits:**
- Prevents accidental API cost overruns
- Protects against abuse
- Ensures fair resource usage

### 5. Sensitive Data Handling

**Implemented:**
- Automatic masking of API keys in logs
- Optional sensitive data masking in production
- Audit logging with sanitized data

**Configuration:**
```bash
# In .env
MASK_SENSITIVE_DATA=true  # Enable in production
ENABLE_AUDIT_LOG=true
```

**Masked Patterns:**
- OpenAI API keys (sk-xxx)
- Bearer tokens
- Credit card numbers
- SSNs

### 6. Secure File Operations

**Implemented:**
- Path traversal prevention
- Working directory restrictions
- File type validation
- Existence checks before operations

**Security Checks:**
1. Path must exist
2. Path must be within allowed directory
3. File extension must be allowed
4. Must be a file (not directory)

## Deployment Security

### Production Checklist

- [ ] Use production API keys (separate from dev)
- [ ] Set `ENVIRONMENT=production` in .env
- [ ] Enable `MASK_SENSITIVE_DATA=true`
- [ ] Enable `ENABLE_AUDIT_LOG=true`
- [ ] Set `LOG_LEVEL=INFO` or `WARNING` (not DEBUG)
- [ ] Set restrictive permissions on .env (chmod 600)
- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable database connection encryption
- [ ] Set up backup strategy
- [ ] Configure rate limits appropriately
- [ ] Review and update .gitignore
- [ ] Implement access control for database
- [ ] Set up monitoring and alerting

### Network Security

**Recommendations:**
1. Use HTTPS for all external API calls (automatic with OpenAI)
2. Implement VPN for database access in production
3. Use firewall rules to restrict access
4. Enable SSL/TLS for PostgreSQL connections

### Database Security

**SQLite (Development):**
```bash
# Set restrictive permissions
chmod 600 financial_agent.db
```

**PostgreSQL (Production):**
```bash
# Use encrypted connections
DATABASE_URL=postgresql://user:pass@localhost:5432/db?sslmode=require

# Use strong passwords
# Enable role-based access control
# Regular backups with encryption
```

## Vulnerability Reporting

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email the maintainer directly
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work on a fix promptly.

## Security Updates

### Version History

**v1.1.0 (Current)**
- ✅ Added .gitignore to prevent sensitive data exposure
- ✅ Fixed SQL injection vulnerabilities
- ✅ Implemented input validation
- ✅ Added rate limiting for API calls
- ✅ Implemented sensitive data masking
- ✅ Added API key validation
- ✅ Created security documentation

**v1.0.0**
- Initial release
- Basic security measures

## Compliance

### Data Privacy

- Financial data is processed locally
- No data shared with third parties (except OpenAI API)
- Audit logs can be encrypted
- Supports data retention policies

### OpenAI API Usage

- Data sent to OpenAI API follows their [Data Usage Policy](https://openai.com/policies/usage-policies)
- API calls are not used for model training (with API-tier accounts)
- Consider using Azure OpenAI for enhanced compliance requirements

## Security Testing

### Automated Checks

Run security validation:
```bash
python -c "from security import validate_security_config; validate_security_config()"
```

### Manual Testing

1. **Test SQL Injection Prevention:**
```python
# Should raise ValueError
from security import InputValidator
InputValidator.validate_month_format("2025-03'; DROP TABLE accounts; --")
```

2. **Test Path Traversal Prevention:**
```python
# Should raise ValueError
InputValidator.validate_file_path("../../etc/passwd")
```

3. **Test Rate Limiting:**
```python
from security import RateLimiter
limiter = RateLimiter(max_calls_per_minute=5)
for i in range(10):
    if not limiter.acquire():
        print(f"Rate limited at call {i}")
```

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OpenAI Security Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Contact

For security concerns or questions:
- Create a private security advisory on GitHub
- Email: [Your security contact email]

---

**Last Updated:** October 2025  
**Version:** 1.1.0
