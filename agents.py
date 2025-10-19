"""
Core Agent Implementations for Financial P&L Anomaly Detection
4 Specialized Agents: Ingestion → Detection → Retrieval → Reporting
"""

import pandas as pd
from openai import OpenAI
from typing import List, Optional
import logging
import time

try:
    import instructor
except ImportError:
    print("Installing instructor...")
    import subprocess
    subprocess.check_call(["pip", "install", "instructor"])
    import instructor

from models import (
    PLTransaction, GLAccount, AnomalyFlag, 
    GLContext, FinancialReasoning, AnomalyExplanation,
    FinancialAnalysisState
)
from config import Config
from database import Database
from vector_store import VectorStoreManager
from cost_tracker import CostTracker
from security import RateLimiter, InputValidator

logger = logging.getLogger(__name__)


class FinancialAgents:
    """Collection of specialized agents for financial analysis"""
    
    def __init__(self, model: str = None, cost_tracker: Optional[CostTracker] = None):
        self.model = model or Config.DEFAULT_MODEL
        self.cost_tracker = cost_tracker
        self.db = Database()
        self.vector_store = VectorStoreManager(cost_tracker=cost_tracker)
        self.instructor_client = instructor.from_openai(OpenAI(api_key=Config.OPENAI_API_KEY))
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        # Initialize rate limiter (60 calls/min, 1000 calls/hour)
        self.rate_limiter = RateLimiter(max_calls_per_minute=60, max_calls_per_hour=1000)
    
    # ========================================================================
    # AGENT 1: Data Ingestion & Structuring
    # ========================================================================
    
    def ingest_data(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Agent 1: Parse P&L and GL files using Pandas with Pydantic validation
        For large datasets, direct parsing is more efficient than LLM validation
        """
        logger.info("🔄 Agent 1: Starting data ingestion...")
        
        try:
            # Load P&L CSV
            pl_df = pd.read_csv(state["pl_file_path"])
            logger.info(f"📂 Loaded {len(pl_df)} transactions from {state['pl_file_path']}")
            
            # Validate data quality
            self.validate_data_quality(pl_df, "pl")
            
            # Convert to Pydantic models (direct parsing, no LLM needed)
            transactions = []
            for _, row in pl_df.iterrows():
                txn = PLTransaction(
                    transaction_id=str(row['transaction_id']),
                    gl_account_id=str(row['gl_account_id']),
                    transaction_date=str(row['date']),
                    amount=float(row['amount']),
                    description=str(row['description']),
                    department=str(row.get('department', '')) if 'department' in row and pd.notna(row.get('department')) else None,
                    vendor=str(row.get('vendor', '')) if 'vendor' in row and pd.notna(row.get('vendor')) else None
                )
                transactions.append(txn)
            
            # Load GL Master CSV
            gl_df = pd.read_csv(state["gl_master_path"])
            logger.info(f"📂 Loaded {len(gl_df)} GL accounts from {state['gl_master_path']}")
            
            # Validate data quality
            self.validate_data_quality(gl_df, "gl")
            
            # Convert to Pydantic models
            gl_accounts = []
            for _, row in gl_df.iterrows():
                acc = GLAccount(
                    account_id=str(row['account_id']),
                    account_name=str(row['account_name']),
                    category=str(row['category']),
                    subcategory=str(row.get('subcategory', '')) if 'subcategory' in row and pd.notna(row.get('subcategory')) else None,
                    typical_min=float(row.get('typical_min', 0)) if 'typical_min' in row and pd.notna(row.get('typical_min')) else None,
                    typical_max=float(row.get('typical_max', 0)) if 'typical_max' in row and pd.notna(row.get('typical_max')) else None,
                    variance_threshold_pct=float(row.get('variance_threshold_pct', 15.0)) if 'variance_threshold_pct' in row else 15.0,
                    is_seasonal=bool(row.get('is_seasonal', False)) if 'is_seasonal' in row else False,
                    seasonal_pattern=str(row.get('seasonal_pattern', '')) if 'seasonal_pattern' in row and pd.notna(row.get('seasonal_pattern')) else None,
                    description=str(row.get('description', '')) if 'description' in row and pd.notna(row.get('description')) else None
                )
                gl_accounts.append(acc)
            
            # Store in database
            self.db.insert_transactions(transactions, str(state["pl_file_path"]))
            self.db.insert_gl_accounts(gl_accounts)
            
            logger.info(f"✅ Agent 1 complete: {len(transactions)} transactions, {len(gl_accounts)} accounts")
            
            return {
                **state,
                "pl_transactions": transactions,
                "gl_accounts": gl_accounts,
                "current_step": "ingestion_complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent 1 failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Ingestion error: {str(e)}"],
                "current_step": "ingestion_failed"
            }
    
    # ========================================================================
    # AGENT 2: Anomaly Detection
    # ========================================================================
    
    def detect_anomalies(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Agent 2: Calculate month-over-month variance and flag anomalies
        Uses statistical methods + GPT-5 for intelligent threshold application
        """
        logger.info("🔍 Agent 2: Starting anomaly detection...")
        
        try:
            # Check if this is first run (aggregate all months)
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM gl_monthly_balances")
            balance_count = cur.fetchone()[0]
            conn.close()
            
            if balance_count == 0:
                logger.info("📊 First run detected - aggregating all historical months...")
                self.db.update_monthly_balances(target_month=None)  # Aggregate ALL
            else:
                # Update only target month
                self.db.update_monthly_balances(state["target_month"])
            
            # Query pre-calculated anomalies
            anomaly_dicts = self.db.get_anomalies(
                state["target_month"],
                threshold_pct=Config.ANOMALY_THRESHOLD_MEDIUM
            )
            
            # Convert to Pydantic models
            anomalies = [AnomalyFlag(**a) for a in anomaly_dicts]
            
            logger.info(f"⚠️  Agent 2 complete: {len(anomalies)} anomalies detected")
            logger.info(f"   - High: {sum(1 for a in anomalies if a.severity.value == 'high')}")
            logger.info(f"   - Medium: {sum(1 for a in anomalies if a.severity.value == 'medium')}")
            
            return {
                **state,
                "anomalies": anomalies,
                "current_step": "detection_complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent 2 failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Detection error: {str(e)}"],
                "current_step": "detection_failed"
            }
    
    # ========================================================================
    # AGENT 3: GL Context Retrieval (RAG)
    # ========================================================================
    
    def retrieve_context(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Agent 3: Retrieve GL account documentation and historical patterns
        Uses RAG (vector store) for semantic search
        """
        logger.info("📚 Agent 3: Retrieving GL context...")
        
        try:
            anomaly_contexts = {}
            
            for anomaly in state["anomalies"]:
                # Build query
                query = f"""
                GL Account {anomaly.gl_account_id} ({anomaly.account_name}) showed 
                {anomaly.variance_percent:.1f}% {anomaly.direction} in {anomaly.current_month}.
                What are typical causes for this account to fluctuate?
                """
                
                # Retrieve from vector store
                docs = self.vector_store.similarity_search(
                    query=query,
                    account_id=anomaly.gl_account_id,
                    k=3
                )
                
                # Get historical anomalies from database
                past_anomalies = self.db.get_historical_anomalies(anomaly.gl_account_id, limit=3)
                
                # Structure context
                context = GLContext(
                    account_id=anomaly.gl_account_id,
                    account_documentation=docs[0].page_content if docs else "No documentation found",
                    similar_cases=[d.page_content for d in docs[1:3] if len(docs) > 1],
                    source_files=[d.metadata.get('source_file', 'unknown') for d in docs],
                    past_anomaly_count=len(past_anomalies)
                )
                
                anomaly_contexts[anomaly.gl_account_id] = context
            
            logger.info(f"✅ Agent 3 complete: Retrieved context for {len(anomaly_contexts)} accounts")
            
            return {
                **state,
                "anomaly_contexts": anomaly_contexts,
                "current_step": "retrieval_complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent 3 failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"],
                "current_step": "retrieval_failed"
            }
    
    # ========================================================================
    # AGENT 4: Explanation & Reporting
    # ========================================================================
    
    def generate_explanations(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Agent 4: Generate detailed explanations for each anomaly
        Uses selected model with high reasoning effort for deep analysis
        """
        logger.info("💭 Agent 4: Generating explanations...")
        
        try:
            explanations = []
            config = Config.get_llm_config("reporting", self.model)
            
            for anomaly in state["anomalies"]:
                # Get context
                gl_context = state["anomaly_contexts"].get(anomaly.gl_account_id)
                
                if not gl_context:
                    logger.warning(f"⚠️  No context for {anomaly.gl_account_id}, skipping")
                    continue
                
                # Get historical anomalies
                past_anomalies = self.db.get_historical_anomalies(anomaly.gl_account_id)
                
                # Build historical context string
                historical_summary = ""
                if past_anomalies:
                    historical_summary = f"""
Historical Pattern Recognition:
This account has been flagged {len(past_anomalies)} time(s) previously.

Past Investigations:
{chr(10).join([f"- {a['detected_month']}: {a['variance_percent']:.1f}% variance - {a.get('root_cause', 'No explanation')}" for a in past_anomalies])}

Consider: Is this current variance part of a recurring pattern?
"""
                
                # Track API call timing
                start_time = time.time()
                
                # Apply rate limiting to prevent API abuse
                self.rate_limiter.wait_if_needed()
                
                # Generate explanation with selected model
                reasoning = self.instructor_client.chat.completions.create(
                    model=config["model"],
                    # reasoning_effort=config["reasoning_effort"],  # GPT-5 specific
                    response_model=FinancialReasoning,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    messages=[
                        {
                            "role": "system",
                            "content": Config.FINANCIAL_ANALYST_PROMPT + Config.TOOL_PREAMBLE_PROMPT
                        },
                        {
                            "role": "user",
                            "content": f"""
Analyze this GL account variance:

**Anomaly Details**:
- GL Account: {anomaly.account_name} ({anomaly.gl_account_id})
- Current Month ({anomaly.current_month}): ${anomaly.current_balance:,.2f}
- Previous Month ({anomaly.previous_month}): ${anomaly.previous_balance:,.2f}
- Variance: {anomaly.variance_percent:+.2f}% (${anomaly.variance_amount:+,.2f})
- Severity: {anomaly.severity.value.upper()}
- Transaction Count: {anomaly.transaction_count}

**GL Account Documentation**:
{gl_context.account_documentation}

**Statistical Context**:
- 3-Month Rolling Average: {f'${anomaly.rolling_3mo_avg:,.2f}' if anomaly.rolling_3mo_avg is not None else 'N/A'}
- 6-Month Std Deviation: {f'${anomaly.rolling_6mo_std:,.2f}' if anomaly.rolling_6mo_std is not None else 'N/A'}
- Z-Score: {f'{anomaly.z_score:.2f}' if anomaly.z_score is not None else 'N/A'}

{historical_summary}

**Your Task**:
1. Provide step-by-step analysis of why this variance occurred
2. Determine if this is expected (seasonal, one-time) or truly anomalous
3. Cite specific data points and historical patterns
4. Recommend immediate action if needed

**IMPORTANT**: Use proper spacing between all words and numbers. Do not concatenate text. Write clearly and professionally.

Use accounting terminology. Be concise but thorough.
"""
                        }
                    ]
                )
                
                # Calculate call duration
                call_duration = time.time() - start_time
                
                # Track cost if tracker is available
                if self.cost_tracker:
                    # Get token usage from the response
                    input_tokens = 0
                    output_tokens = 0
                    
                    # Try different ways to get usage information
                    if hasattr(reasoning, 'usage'):
                        input_tokens = getattr(reasoning.usage, 'prompt_tokens', 0)
                        output_tokens = getattr(reasoning.usage, 'completion_tokens', 0)
                    elif hasattr(reasoning, '_raw_response') and hasattr(reasoning._raw_response, 'usage'):
                        input_tokens = getattr(reasoning._raw_response.usage, 'prompt_tokens', 0)
                        output_tokens = getattr(reasoning._raw_response.usage, 'completion_tokens', 0)
                    elif hasattr(reasoning, 'response') and hasattr(reasoning.response, 'usage'):
                        input_tokens = getattr(reasoning.response.usage, 'prompt_tokens', 0)
                        output_tokens = getattr(reasoning.response.usage, 'completion_tokens', 0)
                    
                    # If we still don't have usage info, estimate based on content
                    if input_tokens == 0 and output_tokens == 0:
                        # Rough estimation: ~4 chars per token
                        input_tokens = len(str(messages)) // 4
                        output_tokens = len(str(reasoning)) // 4
                        logger.warning(f"⚠️  Could not get exact token usage for {anomaly.gl_account_id}, using estimates")
                    
                    # Create query preview for debugging
                    query_preview = f"GL Account {anomaly.gl_account_id} ({anomaly.account_name}) variance analysis"
                    
                    self.cost_tracker.track_call(
                        model=config["model"],
                        agent="reporting",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        call_duration=call_duration,
                        query_preview=query_preview
                    )
                
                # Clean up any concatenated text in the reasoning
                reasoning.chain_of_thought = self.clean_text_spacing(reasoning.chain_of_thought)
                reasoning.root_cause = self.clean_text_spacing(reasoning.root_cause)
                reasoning.recommendation = self.clean_text_spacing(reasoning.recommendation)
                
                # Create explanation object
                explanation = AnomalyExplanation(
                    anomaly=anomaly,
                    reasoning=reasoning,
                    retrieved_context=gl_context
                )
                
                explanations.append(explanation)
                
                # Store in database
                self.db.store_anomaly_investigation(explanation)
                
                logger.info(f"✅ Explained {anomaly.gl_account_id}: {reasoning.root_cause[:50]}...")
            
            logger.info(f"✅ Agent 4 complete: Generated {len(explanations)} explanations")
            
            return {
                **state,
                "explanations": explanations,
                "current_step": "reporting_complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent 4 failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Reporting error: {str(e)}"],
                "current_step": "reporting_failed"
            }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def clean_text_spacing(self, text: str) -> str:
        """Clean up concatenated text by adding proper spaces"""
        import re
        
        # Fix concatenated phrases in order (most specific first)
        # Common financial phrases
        text = re.sub(r'isalsowellabovethe', ' is also well above the ', text)
        text = re.sub(r'isalsowellbelowthe', ' is also well below the ', text)
        text = re.sub(r'iswellabovethe', ' is well above the ', text)
        text = re.sub(r'iswellbelowthe', ' is well below the ', text)
        text = re.sub(r'isalsoabovethe', ' is also above the ', text)
        text = re.sub(r'isalsobelowthe', ' is also below the ', text)
        text = re.sub(r'isabovethe', ' is above the ', text)
        text = re.sub(r'isbelowthe', ' is below the ', text)
        text = re.sub(r'typicalmonthlyrangeof', ' typical monthly range of ', text)
        text = re.sub(r'monthlyrangeof', ' monthly range of ', text)
        text = re.sub(r'typicalrange', ' typical range', text)
        text = re.sub(r'rangeof', ' range of ', text)
        
        # Common concatenated words
        text = re.sub(r'isalso', ' is also ', text)
        text = re.sub(r'iswell', ' is well ', text)
        text = re.sub(r'wellabove', ' well above ', text)
        text = re.sub(r'wellbelow', ' well below ', text)
        text = re.sub(r'wellwithin', ' well within ', text)
        text = re.sub(r'welloutside', ' well outside ', text)
        
        # Number followed by word without space (e.g., "8,500.00is" -> "8,500.00 is")
        text = re.sub(r'(\d+(?:,\d{3})*(?:\.\d{2})?)([a-zA-Z])', r'\1 \2', text)
        
        # Word followed by number without space
        text = re.sub(r'([a-zA-Z])(\d+(?:,\d{3})*(?:\.\d{2})?)', r'\1 \2', text)
        
        # Fix preposition + $ (e.g., "to$" -> "to $", "from$" -> "from $")
        text = re.sub(r'\b(to|from|of|at|by|for|with)\$', r'\1 $', text)
        
        # Fix common word-to-word concatenations (more conservative approach)
        # Only match when there are NO spaces between complete words
        # Use negative lookbehind/lookahead to avoid splitting within words
        
        # Common short words that get concatenated with following words
        # Articles (the, a, an) when followed by a capital or another word
        text = re.sub(r'\bthe([A-Z][a-z]+)', r'the \1', text)  # camelCase: theAccount
        text = re.sub(r'\ba([A-Z][a-z]+)', r'a \1', text)  # camelCase: aSignificant
        text = re.sub(r'\ban([A-Z][a-z]+)', r'an \1', text)  # camelCase: anAccount
        
        # Specific common concatenations (whitelist approach to avoid false positives)
        concatenations = [
            # article + noun patterns
            (r'\btheaccount\b', 'the account'),
            (r'\bthemonth\b', 'the month'),
            (r'\btheyear\b', 'the year'),
            (r'\btheamount\b', 'the amount'),
            (r'\bthebalance\b', 'the balance'),
            (r'\bthetransaction\b', 'the transaction'),
            (r'\bthevariance\b', 'the variance'),
            (r'\bthethreshold\b', 'the threshold'),
            (r'\btheanalysis\b', 'the analysis'),
            (r'\btheexpense\b', 'the expense'),
            (r'\basignificant\b', 'a significant'),
            (r'\balarge\b', 'a large'),
            (r'\basmall\b', 'a small'),
            (r'\basingle\b', 'a single'),
            (r'\basimilar\b', 'a similar'),
            
            # verb + auxiliary/adjective patterns (longer patterns first!)
            (r'\baccounthasbeenflagged\b', 'account has been flagged'),
            (r'\bhasbeenflagged\b', 'has been flagged'),
            (r'\bhasbeen\b', 'has been'),
            (r'\baccounthas\b', 'account has'),
            (r'\baccountis\b', 'account is'),
            (r'\bbalancehas\b', 'balance has'),
            (r'\bbalanceis\b', 'balance is'),
            (r'\bvariancehas\b', 'variance has'),
            (r'\bvarianceis\b', 'variance is'),
            
            # adverb + verb patterns
            (r'\btypicallyranges\b', 'typically ranges'),
            (r'\btypicallyvaries\b', 'typically varies'),
            (r'\btypicallyshows\b', 'typically shows'),
            (r'\busuallyranges\b', 'usually ranges'),
            (r'\busuallyshows\b', 'usually shows'),
            (r'\bnotablyhas\b', 'notably has'),
            (r'\bnotablyis\b', 'notably is'),
            (r'\bnotablyhigher\b', 'notably higher'),
            (r'\bnotablylower\b', 'notably lower'),
            (r'\bsignificantlyhas\b', 'significantly has'),
            (r'\bsignificantlyis\b', 'significantly is'),
            (r'\bsignificantlyhigher\b', 'significantly higher'),
            (r'\bsignificantlylower\b', 'significantly lower'),
            
            # preposition + article/word patterns
            (r'\binthe\b', 'in the'),
            (r'\bfromthe\b', 'from the'),
            (r'\btothe\b', 'to the'),
            (r'\bofthe\b', 'of the'),
            (r'\batthe\b', 'at the'),
            (r'\bforthe\b', 'for the'),
            (r'\bwiththe\b', 'with the'),
            (r'\bonthe\b', 'on the'),
            (r'\bbythe\b', 'by the'),
        ]
        
        for pattern, replacement in concatenations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # camelCase (but be careful not to split abbreviations)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix specific financial terms (keep together)
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # $amount (remove space)
        text = re.sub(r'(\d)\s+%', r'\1%', text)   # number% (remove space)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> bool:
        """Validate CSV data quality before processing"""
        logger.info(f"🔍 Validating {data_type} data quality...")
        
        if data_type == "pl":
            required_cols = ['transaction_id', 'gl_account_id', 'date', 'amount', 'description']
            
            # Check required columns
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for nulls
            if df['gl_account_id'].isna().any():
                raise ValueError("Found null GL account IDs")
            
            if df['amount'].isna().any():
                raise ValueError("Found null amounts")
            
            # Validate dates
            try:
                pd.to_datetime(df['date'])
            except:
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
            
            # Validate amounts are numeric
            if not pd.to_numeric(df['amount'], errors='coerce').notna().all():
                raise ValueError("Non-numeric values in amount column")
        
        elif data_type == "gl":
            required_cols = ['account_id', 'account_name', 'category']
            
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required GL master columns: {missing_cols}")
        
        logger.info(f"✅ {data_type.upper()} data validation passed")
        return True


# Convenience functions for use in LangGraph nodes

def data_ingestion_agent(state: FinancialAnalysisState) -> FinancialAnalysisState:
    """LangGraph node: Data ingestion"""
    agents = FinancialAgents()
    return agents.ingest_data(state)


def anomaly_detection_agent(state: FinancialAnalysisState) -> FinancialAnalysisState:
    """LangGraph node: Anomaly detection"""
    agents = FinancialAgents()
    return agents.detect_anomalies(state)


def gl_retrieval_agent(state: FinancialAnalysisState) -> FinancialAnalysisState:
    """LangGraph node: GL context retrieval"""
    agents = FinancialAgents()
    return agents.retrieve_context(state)


def reporting_agent(state: FinancialAnalysisState) -> FinancialAnalysisState:
    """LangGraph node: Explanation generation"""
    agents = FinancialAgents()
    return agents.generate_explanations(state)

