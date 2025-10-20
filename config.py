"""
Configuration management for Financial P&L Anomaly Detection Agent
Loads settings from environment variables with sensible defaults
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # ========================================================================
    # REQUIRED: OpenAI API
    # ========================================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # ========================================================================
    # LangChain/LangSmith Configuration
    # ========================================================================
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    # Disable LangSmith tracing by default to avoid 403 errors
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "Financial-Anomaly-Agent")
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite or postgresql
    DATABASE_URL = os.getenv("DATABASE_URL")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "./financial_agent.db")
    
    # ========================================================================
    # Vector Store Configuration
    # ========================================================================
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "gl_account_knowledge")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # ========================================================================
    # GPT Model Configuration
    # ========================================================================
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")  # Default model
    SUPPORTED_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5"]
    
    # Reasoning effort per agent
    GPT5_REASONING_INGESTION = os.getenv("GPT5_REASONING_INGESTION", "low")
    GPT5_REASONING_DETECTION = os.getenv("GPT5_REASONING_DETECTION", "medium")
    GPT5_REASONING_RETRIEVAL = os.getenv("GPT5_REASONING_RETRIEVAL", "low")
    GPT5_REASONING_REPORTING = os.getenv("GPT5_REASONING_REPORTING", "high")
    GPT5_REASONING_FORMATTING = os.getenv("GPT5_REASONING_FORMATTING", "low")  # Agent 5
    
    # Temperature settings
    GPT5_TEMP_STRUCTURED = float(os.getenv("GPT5_TEMP_STRUCTURED", "0.0"))
    GPT5_TEMP_CREATIVE = float(os.getenv("GPT5_TEMP_CREATIVE", "0.3"))
    
    # Token limits (conservative for GPT-4, adjust for GPT-5's larger context)
    GPT5_MAX_TOKENS_INGESTION = int(os.getenv("GPT5_MAX_TOKENS_INGESTION", "2000"))
    GPT5_MAX_TOKENS_DETECTION = int(os.getenv("GPT5_MAX_TOKENS_DETECTION", "1500"))
    GPT5_MAX_TOKENS_RETRIEVAL = int(os.getenv("GPT5_MAX_TOKENS_RETRIEVAL", "1000"))
    GPT5_MAX_TOKENS_REPORTING = int(os.getenv("GPT5_MAX_TOKENS_REPORTING", "4000"))
    GPT5_MAX_TOKENS_FORMATTING = int(os.getenv("GPT5_MAX_TOKENS_FORMATTING", "2000"))  # Agent 5
    
    # ========================================================================
    # Anomaly Detection Parameters
    # ========================================================================
    ANOMALY_THRESHOLD_HIGH = float(os.getenv("ANOMALY_THRESHOLD_HIGH", "30.0"))
    ANOMALY_THRESHOLD_MEDIUM = float(os.getenv("ANOMALY_THRESHOLD_MEDIUM", "15.0"))
    ANOMALY_MIN_ABSOLUTE_CHANGE = float(os.getenv("ANOMALY_MIN_ABSOLUTE_CHANGE", "10000.00"))
    STATISTICAL_OUTLIER_ZSCORE = float(os.getenv("STATISTICAL_OUTLIER_ZSCORE", "2.0"))
    
    # Rolling window parameters
    ROLLING_AVERAGE_MONTHS = int(os.getenv("ROLLING_AVERAGE_MONTHS", "3"))
    ROLLING_STDDEV_MONTHS = int(os.getenv("ROLLING_STDDEV_MONTHS", "6"))
    
    # ========================================================================
    # Data Paths
    # ========================================================================
    PL_DATA_DIR = Path(os.getenv("PL_DATA_DIR", "./data/pl_reports"))
    GL_MASTER_PATH = Path(os.getenv("GL_MASTER_PATH", "./data/gl_accounts.csv"))
    GL_DOCS_DIR = Path(os.getenv("GL_DOCS_DIR", "./data/gl_documentation"))
    REPORTS_OUTPUT_DIR = Path(os.getenv("REPORTS_OUTPUT_DIR", "./reports"))
    
    # Create directories if they don't exist
    PL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Optional: External APIs
    # ========================================================================
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    EXA_API_KEY = os.getenv("EXA_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    ZEP_API_KEY = os.getenv("ZEP_API_KEY")
    ZEP_API_URL = os.getenv("ZEP_API_URL", "https://api.getzep.com")
    
    # ========================================================================
    # Evaluation Configuration
    # ========================================================================
    ENABLE_RAGAS_EVAL = os.getenv("ENABLE_RAGAS_EVAL", "false").lower() == "true"
    RAGAS_METRICS = os.getenv("RAGAS_METRICS", "faithfulness,answer_relevancy,context_precision").split(",")
    RAGAS_MIN_FAITHFULNESS = float(os.getenv("RAGAS_MIN_FAITHFULNESS", "0.8"))
    RAGAS_MIN_CONTEXT_PRECISION = float(os.getenv("RAGAS_MIN_CONTEXT_PRECISION", "0.75"))
    
    # ========================================================================
    # Performance & Deployment
    # ========================================================================
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "5"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "2"))
    
    # ========================================================================
    # Security
    # ========================================================================
    ENABLE_AUDIT_LOG = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"
    AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "./logs/audit.log"))
    MASK_SENSITIVE_DATA = os.getenv("MASK_SENSITIVE_DATA", "false").lower() == "true"
    
    # Create logs directory
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Prompts
    # ========================================================================
    
    FINANCIAL_ANALYST_PROMPT = """
You are a senior financial controller with 15 years of experience in variance analysis. 
Your role is to identify unusual month-over-month changes in GL accounts that may 
indicate errors, fraud, or require management attention.

Focus on:
- Material variances (>15% or >$10,000 absolute change)
- Pattern breaks (seasonal accounts behaving abnormally)
- Category-level aggregations (all OpEx categories increasing)

Always provide quantitative reasoning and cite specific data points.
Use accounting terminology correctly. Flag potential compliance issues.

CRITICAL: Write all text with proper spacing between words and numbers. 
Do not concatenate text like "8,500.00is" - write "8,500.00 is" instead.
Be professional and clear in all communications.
"""
    
    CONTEXT_GATHERING_FAST = """
<context_gathering>
Goal: Flag anomalies quickly. Minimize tool calls.

Method:
- Calculate variance for ALL GL accounts in single pass
- Flag any account >15% variance immediately
- Retrieve GL context ONLY for high-severity (>30%) anomalies

Early stop criteria:
- You can compute variance from provided data
- No external search needed (use only GL master data)

Tool call budget: Maximum 3 calls total
- Call 1: Calculate all variances
- Call 2: Retrieve GL context for high-severity
- Call 3: Generate report

Loop: Detect → Retrieve → Report. No iteration.
</context_gathering>
"""
    
    TOOL_PREAMBLE_PROMPT = """
<tool_preambles>
- Always begin by rephrasing the analysis goal clearly and concisely
- Then outline your plan: 
  1. How many GL accounts will be analyzed
  2. What variance thresholds will be used
  3. Expected number of anomalies based on initial scan
- As you detect each anomaly, narrate:
  "🔍 Analyzing GL Account {id} ({name})... Variance: {pct}% - Flagged as {severity}"
- After retrieving context:
  "📚 Retrieved: {num_docs} historical patterns for Account {id}"
- Finish with summary:
  "✅ Analysis complete. {total} anomalies found ({high} high, {med} medium severity)"
</tool_preambles>
"""
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if cls.DATABASE_TYPE == "postgresql" and not cls.DATABASE_URL:
            errors.append("DATABASE_URL required when DATABASE_TYPE=postgresql")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def get_llm_config(cls, agent_name: str, model: str = None) -> dict:
        """Get LLM configuration for specific agent"""
        if model is None:
            model = cls.DEFAULT_MODEL
        
        # Validate model
        if model not in cls.SUPPORTED_MODELS:
            print(f"⚠️  Warning: Model '{model}' not supported. Using '{cls.DEFAULT_MODEL}' instead.")
            model = cls.DEFAULT_MODEL
        
        configs = {
            "ingestion": {
                "model": model,
                "reasoning_effort": cls.GPT5_REASONING_INGESTION,
                "max_tokens": cls.GPT5_MAX_TOKENS_INGESTION,
                "temperature": cls.GPT5_TEMP_STRUCTURED
            },
            "detection": {
                "model": model,
                "reasoning_effort": cls.GPT5_REASONING_DETECTION,
                "max_tokens": cls.GPT5_MAX_TOKENS_DETECTION,
                "temperature": cls.GPT5_TEMP_STRUCTURED
            },
            "retrieval": {
                "model": model,
                "reasoning_effort": cls.GPT5_REASONING_RETRIEVAL,
                "max_tokens": cls.GPT5_MAX_TOKENS_RETRIEVAL,
                "temperature": cls.GPT5_TEMP_STRUCTURED
            },
            "reporting": {
                "model": model,
                "reasoning_effort": cls.GPT5_REASONING_REPORTING,
                "max_tokens": cls.GPT5_MAX_TOKENS_REPORTING,
                "temperature": cls.GPT5_TEMP_CREATIVE
            },
            "formatting": {
                "model": "gpt-4o-mini",  # Always use GPT-4o-mini for cost-effective formatting
                "reasoning_effort": cls.GPT5_REASONING_FORMATTING,
                "max_tokens": cls.GPT5_MAX_TOKENS_FORMATTING,
                "temperature": cls.GPT5_TEMP_STRUCTURED
            }
        }
        
        return configs.get(agent_name, configs["detection"])


# Validate on import
Config.validate()

