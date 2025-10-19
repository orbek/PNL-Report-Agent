# Quick Start Guide - Secure Setup

## Prerequisites

- Python 3.10+
- OpenAI API key

## Secure Installation (5 minutes)

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/orbek/PNL-Report-Agent.git
cd PNL-Report-Agent

# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure Environment Securely

```bash
# Copy environment template
cp .env.example .env

# Set secure permissions (Unix/Linux/Mac)
chmod 600 .env

# Edit configuration
nano .env
```

**Required Configuration:**
```bash
# In .env file
OPENAI_API_KEY=sk-your-actual-key-here
DEFAULT_MODEL=gpt-4o  # Cost-effective option
ENVIRONMENT=development
```

**Important:** Never commit your `.env` file! It's already in `.gitignore`.

### 3. Initialize System with Security Check

```bash
# This will automatically validate security configuration
python main.py init
```

Expected output:
```
🔒 Validating security configuration...
✅ Security configuration validated
🚀 Initializing Financial Anomaly Detection Agent
...
```

### 4. Generate Sample Data (Optional)

```bash
python main.py --generate-sample
```

### 5. Run First Analysis

```bash
python main.py analyze data/pl_reports/pl_2025-03.csv
```

## Security Checklist ✅

Before running in production:

- [ ] `.env` file has restricted permissions (`chmod 600 .env`)
- [ ] API key is production key (not dev/test key)
- [ ] `ENVIRONMENT=production` in `.env`
- [ ] `MASK_SENSITIVE_DATA=true` in `.env`
- [ ] `.gitignore` is in place
- [ ] No `.env` file committed to git
- [ ] Security validation passes on startup
- [ ] Rate limiting configured appropriately

## Common Issues & Solutions

### Issue: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Check if .env exists
ls -la .env

# Check permissions
ls -l .env
# Should show: -rw------- (600)

# Verify API key is set
cat .env | grep OPENAI_API_KEY
```

### Issue: "Invalid OpenAI API key format"

**Solution:**
```bash
# OpenAI keys must start with 'sk-'
# Example: sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890

# Verify your key format
python -c "from security import APIKeyValidator; print(APIKeyValidator.validate_openai_key('YOUR_KEY_HERE'))"
```

### Issue: "File path is outside allowed directory"

**Solution:**
```bash
# Use relative paths from project root
python main.py analyze data/pl_reports/pl_2025-03.csv

# Or absolute paths within project
python main.py analyze /full/path/to/PNL-Report-Agent/data/pl_reports/pl_2025-03.csv
```

### Issue: Rate limit warnings

**Solution:**
```python
# Adjust rate limits in your code if needed
# In agents.py, modify initialization:
self.rate_limiter = RateLimiter(
    max_calls_per_minute=30,  # Lower for free tier
    max_calls_per_hour=500
)
```

## Performance Tips

### Use Cost-Effective Models

```bash
# In .env - recommended for most use cases
DEFAULT_MODEL=gpt-4o  # Best balance of cost/quality

# For testing
DEFAULT_MODEL=gpt-4o-mini  # Cheapest option

# For maximum quality (expensive)
DEFAULT_MODEL=gpt-4
```

### Enable Caching

Caching is enabled by default. To verify:

```python
from cache import get_cache

cache = get_cache()
stats = cache.get_stats()
print(f"Cache stats: {stats}")
```

### Monitor Costs

```bash
# Costs are tracked automatically
# Check cost report after analysis
cat reports/cost_report_*.json
```

## Monitoring Security

### Regular Security Checks

```bash
# Run security validation
python -c "from security import validate_security_config; validate_security_config()"
```

### Check File Permissions

```bash
# Ensure sensitive files have correct permissions
ls -l .env
# Should show: -rw------- (600)

ls -l *.db
# Should show: -rw------- (600)
```

### Review Logs

```bash
# Check audit logs for security events
tail -f logs/audit.log
```

## Production Deployment

### Additional Steps for Production

1. **Use PostgreSQL:**
```bash
# In .env
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://user:pass@localhost:5432/financial_db?sslmode=require
```

2. **Enable Strict Security:**
```bash
# In .env
ENVIRONMENT=production
MASK_SENSITIVE_DATA=true
LOG_LEVEL=INFO
ENABLE_AUDIT_LOG=true
```

3. **Set Up Monitoring:**
```bash
# Monitor API costs daily
# Set up alerts for unusual activity
# Regular security audits
```

4. **Backup Strategy:**
```bash
# Regular database backups
# Encrypt backups
# Store securely off-site
```

## Resources

- [SECURITY.md](SECURITY.md) - Complete security guidelines
- [OPTIMIZATION.md](OPTIMIZATION.md) - Performance optimization
- [CHANGES.md](CHANGES.md) - Detailed change log
- [README.md](README.md) - Full documentation

## Support

For issues:
1. Check troubleshooting section above
2. Review [SECURITY.md](SECURITY.md)
3. Check existing GitHub issues
4. Create new issue with details

For security vulnerabilities:
- **DO NOT** open public issue
- Email maintainer directly
- See [SECURITY.md](SECURITY.md) for contact info

## Version

**Current Version:** 1.1.0  
**Last Updated:** October 2025  
**Security Level:** Production-ready ✅

---

**Ready to start? Run:** `python main.py init`
