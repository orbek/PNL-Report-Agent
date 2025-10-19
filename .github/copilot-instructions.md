# Copilot Instructions for PNL-Report-Agent

## Project Overview

This is a **Financial P&L Anomaly Detection Agent** built with LangGraph, GPT-4/4o/5, and advanced RAG techniques. The system automatically detects unusual transactions in Profit & Loss reports by comparing month-over-month patterns.

### Core Technology Stack
- **Orchestration**: LangGraph for multi-agent workflow
- **LLM**: OpenAI GPT-4, GPT-4o, and GPT-5 with Instructor for structured outputs
- **Vector Store**: ChromaDB for RAG-based context retrieval
- **Database**: SQLite (default) or PostgreSQL
- **Data Processing**: Pandas, NumPy
- **Validation**: Pydantic models

## Architecture

### 4-Agent LangGraph Workflow

1. **Agent 1: Data Ingestion** (`agents.py::ingest_data`)
   - Loads P&L CSV files and GL master data
   - Validates data quality using Pydantic models
   - Stores transactions in database

2. **Agent 2: Anomaly Detection** (`agents.py::detect_anomalies`)
   - Calculates month-over-month variances
   - Applies statistical analysis (Z-scores, rolling averages)
   - Categorizes anomalies by severity (High/Medium/Low)

3. **Agent 3: Context Retrieval** (`agents.py::retrieve_context`)
   - Uses RAG to gather relevant GL account documentation
   - Performs vector similarity search via ChromaDB
   - Provides historical patterns and context

4. **Agent 4: Report Generation** (`agents.py::generate_report`)
   - Creates detailed explanations using GPT-4/4o/5
   - Generates professional markdown reports
   - Tracks API costs per analysis

### Key Files
- `main.py` - CLI entry point with argparse commands
- `workflow.py` - LangGraph state machine and agent orchestration
- `agents.py` - Core agent implementations
- `models.py` - Pydantic models for all data structures
- `database.py` - SQLAlchemy database operations
- `vector_store.py` - ChromaDB vector store management
- `cost_tracker.py` - Real-time API cost monitoring
- `config.py` - Environment-based configuration

## Coding Standards

### Python Style
- Follow PEP 8 conventions
- Use type hints for all function parameters and return values
- Document functions with docstrings following Google style
- Prefer dataclasses and Pydantic models over dictionaries
- Use pathlib for file operations, not os.path

### Error Handling
- Use specific exception types, not bare `except:`
- Log errors with context using the logging module
- Gracefully handle missing environment variables with defaults
- Validate user inputs early with clear error messages

### Data Validation
- All data structures must use Pydantic models (see `models.py`)
- Validate CSV data before database insertion
- Use `.model_validate()` for Pydantic model creation
- Handle validation errors with helpful messages

### LLM Interactions
- Use Instructor library for structured outputs
- Track costs for all API calls via `CostTracker`
- Set appropriate temperature (0.0 for structured, 0.3 for creative)
- Configure reasoning effort for GPT-5 models per agent
- Always include max_tokens to prevent runaway costs

### Database Operations
- Use context managers for database sessions
- Never commit sensitive data (API keys, passwords)
- Use parameterized queries to prevent SQL injection
- Handle duplicate keys with upsert operations

## Testing & Validation

### Manual Testing
1. Initialize system: `python main.py init`
2. Generate sample data: `python main.py --generate-sample`
3. Run analysis: `python main.py analyze data/pl_2025_03.csv`
4. Verify cost tracking and report generation

### Data Validation
- Ensure P&L files have required columns: `gl_account_id`, `period`, `actual_amount`
- GL master data must include: `gl_account_id`, `account_name`, `category`, `subcategory`
- All monetary amounts should be decimal values
- Dates should follow YYYY-MM format

### Expected Behavior
- Analysis should complete in 2-5 minutes
- Cost reports should show detailed breakdowns by agent
- Reports should be saved to `reports/` directory
- No anomalies detected is valid if data has no significant variances

## Configuration

### Environment Variables
Required:
- `OPENAI_API_KEY` - OpenAI API key for LLM and embeddings

Optional:
- `DEFAULT_MODEL` - Model selection (gpt-4, gpt-4o, gpt-5)
- `VARIANCE_THRESHOLD` - Anomaly detection threshold (default: 15.0)
- `DATABASE_TYPE` - sqlite or postgresql
- `ENABLE_COST_TRACKING` - true/false for cost monitoring

### File Structure
```
data/
├── gl_accounts.csv         # GL master data
├── gl_documentation/        # RAG knowledge base
└── pl_reports/             # P&L CSV files

reports/                    # Generated analysis reports
logs/                       # Audit logs
chroma_db/                  # Vector store persistence
```

## Common Tasks

### Adding a New Agent
1. Define agent function in `agents.py` following the signature: `def agent_name(self, state: FinancialAnalysisState) -> FinancialAnalysisState`
2. Add agent to workflow in `workflow.py`
3. Update `FinancialAnalysisState` in `models.py` if new state fields are needed
4. Configure LLM settings in `Config.get_llm_config()`

### Modifying Anomaly Detection Logic
1. Update thresholds in `config.py` or via environment variables
2. Modify detection logic in `agents.py::detect_anomalies()`
3. Ensure changes are reflected in `models.py::AnomalyFlag`
4. Test with sample data to verify accuracy

### Adding New Data Sources
1. Create Pydantic model in `models.py`
2. Add database table in `database.py`
3. Create ingestion logic in `agents.py`
4. Update workflow state if needed

### Improving RAG Context
1. Add documentation files to `data/gl_documentation/`
2. Run `python main.py rebuild-index` to update vector store
3. Verify retrieval quality with test queries
4. Adjust embedding model if needed in `config.py`

## Performance & Cost Optimization

### Model Selection
- **GPT-4**: High accuracy, $0.03-0.06 per 1M tokens
- **GPT-4o**: Balanced, $0.005-0.015 per 1M tokens (recommended for production)
- **GPT-5**: Maximum reasoning, $1.25-10.00 per 1M tokens (use selectively)

### Optimization Tips
- Use GPT-4o for routine analysis (99% cost savings vs GPT-5)
- Set appropriate max_tokens limits to prevent overuse
- Cache vector embeddings to avoid re-embedding
- Batch process multiple reports when possible
- Use reasoning_effort="low" for simple tasks

## Dependencies

### Core Dependencies
- `langchain>=0.3.27` - LLM orchestration
- `langgraph` - Agent workflow state machine
- `openai>=1.106.1` - OpenAI API client
- `instructor` - Structured outputs from LLMs
- `chromadb` - Vector database for RAG
- `pandas` - Data manipulation
- `pydantic` - Data validation

### Optional Dependencies
- `ragas` - RAG evaluation metrics
- `psycopg2-binary` - PostgreSQL support
- `zep-python` - Long-term memory storage

## Security Considerations

- Never commit `.env` files or API keys
- Use environment variables for all secrets
- Enable audit logging in production (`ENABLE_AUDIT_LOG=true`)
- Mask sensitive data if needed (`MASK_SENSITIVE_DATA=true`)
- Validate all user inputs before database operations
- Use parameterized queries to prevent SQL injection

## Troubleshooting

### "No anomalies detected"
- Check if `gl_monthly_balances` table has data
- Verify variance thresholds aren't too strict
- Ensure historical data exists for comparison

### "Context length exceeded"
- Reduce number of GL accounts in analysis
- Use GPT-4o or GPT-5 with larger context windows
- Check for data quality issues (duplicate entries)

### "Cost tracking shows $0.0000"
- Verify OpenAI API key is valid
- Check if CostTracker is initialized in workflow
- Review model selection and pricing configuration

### Vector store issues
- Rebuild index: `python main.py rebuild-index`
- Check ChromaDB persistence directory exists
- Verify embedding model is available

## Best Practices for Contributors

1. **Understand the architecture** - Review the 4-agent workflow before making changes
2. **Use existing patterns** - Follow the structure in `agents.py` and `models.py`
3. **Test with sample data** - Always test changes with provided sample data
4. **Monitor costs** - Review cost reports after testing LLM changes
5. **Validate data** - Use Pydantic models for all data structures
6. **Document changes** - Update docstrings and this file as needed
7. **Handle errors gracefully** - Provide helpful error messages
8. **Follow Python conventions** - Use type hints, logging, and proper error handling

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Instructor Library](https://github.com/jxnl/instructor)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Related Resources

For additional context on AI agent development and LangChain patterns, refer to educational materials from online courses focusing on:
- Multi-agent systems with LangGraph
- RAG (Retrieval-Augmented Generation) implementations
- Structured outputs with LLMs
- Cost optimization for production AI systems

---

**Last Updated**: October 2025  
**Repository**: https://github.com/orbek/PNL-Report-Agent
