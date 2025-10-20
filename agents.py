"""
Core Agent Implementations for Financial P&L Anomaly Detection
5 Specialized Agents: Ingestion → Detection → Retrieval → Reporting → Formatting
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

**CRITICAL FORMATTING RULES**:
- Currency amounts MUST stay on single lines (e.g., "\\$15,000.00" NOT "\\$15" on one line and "000.00" on next line)
- Percentages MUST stay on single lines (e.g., "+17,961.41%" NOT "+17" on one line and "961.41%" on next line)
- All supporting evidence items should be complete on single lines
- Use proper comma separators for thousands in currency amounts
- ALWAYS escape dollar signs with SINGLE backslashes (\\$) to prevent Markdown math interpretation
- NEVER split currency amounts across multiple lines - keep entire amounts together
- Example: "Current Month Amount: \\$15,000.00" NOT "Current Month Amount: \\$15" followed by "000.00" on next line
- Each supporting evidence item must be a complete sentence on a single line
- Do NOT break currency amounts into separate list items
- Format ALL currency amounts with proper commas: \\$15,000.00, \\$1,500.00, \\$25,000.00
- Ensure supporting evidence is formatted as complete sentences with proper currency formatting
- Use SINGLE backslash before dollar signs: \\$ NOT \\\\$

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
                
                # Post-process all reasoning fields to fix currency formatting issues
                explanation.reasoning.chain_of_thought = self._fix_currency_formatting(explanation.reasoning.chain_of_thought)
                explanation.reasoning.root_cause = self._fix_currency_formatting(explanation.reasoning.root_cause)
                explanation.reasoning.recommendation = self._fix_currency_formatting(explanation.reasoning.recommendation)
                
                if explanation.reasoning.supporting_evidence:
                    fixed_evidence = []
                    for evidence in explanation.reasoning.supporting_evidence:
                        # Apply currency formatting fix
                        fixed_evidence.append(self._fix_currency_formatting(evidence))
                    explanation.reasoning.supporting_evidence = fixed_evidence
                
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
    # AGENT 5: Report Formatting & Text Cleanup
    # ========================================================================
    
    def format_report(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Agent 5: Fix any spacing, formatting, or text concatenation issues in generated explanations
        Uses GPT-4o-mini for cost-effective post-processing
        """
        logger.info("✨ Agent 5: Starting report formatting...")
        
        try:
            # Use GPT-4o-mini for cost-effective formatting
            format_config = Config.get_llm_config("formatting", "gpt-4o-mini")
            
            formatted_explanations = []
            
            for explanation in state["explanations"]:
                # Track API call timing
                start_time = time.time()
                
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Create formatting prompt
                formatting_prompt = f"""
Fix any text spacing, concatenation, or formatting issues in the following financial analysis text.

RULES:
1. Add proper spaces between concatenated words (e.g., "accounthas" → "account has")
2. Fix number-word concatenations (e.g., "$8,500.00is" → "$8,500.00 is")
3. Ensure proper spacing around punctuation
4. Keep all numbers, percentages, and financial data exactly as they are
5. Maintain the professional tone and technical accuracy
6. Do NOT change the meaning or content, only fix spacing/formatting
7. CRITICAL: Fix currency amounts that are split across lines (e.g., "\\$35" on one line and "000.00" on next line should become "\\$35,000.00")
8. Ensure currency amounts stay on single lines with proper comma separators
9. Fix any line breaks within currency values or percentages
10. ALWAYS escape dollar signs with backslashes (\\$) to prevent Markdown math interpretation
11. Look for patterns like "\\$15" followed by "000.00" on separate lines and combine them into "\\$15,000.00"
12. Fix patterns where currency amounts are split across multiple lines in supporting evidence

TEXT TO FIX:

**Root Cause**: {explanation.reasoning.root_cause}

**Chain of Thought**: {explanation.reasoning.chain_of_thought}

**Recommendation**: {explanation.reasoning.recommendation}

**Supporting Evidence**: {', '.join(explanation.reasoning.supporting_evidence) if explanation.reasoning.supporting_evidence else 'None'}

Return the corrected text in the same format, maintaining the structure but with proper spacing. Pay special attention to currency formatting and ensure all dollar amounts are properly formatted on single lines.
"""
                
                # Call GPT-4o-mini to fix formatting
                response = self.openai_client.chat.completions.create(
                    model=format_config["model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a text formatting specialist. Fix spacing and concatenation issues in financial text while preserving all data and meaning. Return only the corrected text in the same format."
                        },
                        {
                            "role": "user",
                            "content": formatting_prompt
                        }
                    ],
                    temperature=format_config["temperature"],
                    max_tokens=format_config["max_tokens"]
                )
                
                # Parse the formatted response
                formatted_text = response.choices[0].message.content.strip()
                
                # Extract the formatted sections (simple parsing)
                import re
                
                # Extract each section
                root_cause_match = re.search(r'\*\*Root Cause\*\*:\s*(.+?)(?=\n\*\*|$)', formatted_text, re.DOTALL)
                chain_match = re.search(r'\*\*Chain of Thought\*\*:\s*(.+?)(?=\n\*\*|$)', formatted_text, re.DOTALL)
                recommendation_match = re.search(r'\*\*Recommendation\*\*:\s*(.+?)(?=\n\*\*|$)', formatted_text, re.DOTALL)
                evidence_match = re.search(r'\*\*Supporting Evidence\*\*:\s*(.+?)(?=\n\*\*|$)', formatted_text, re.DOTALL)
                
                # Additional currency formatting fix for extracted evidence
                def fix_currency_formatting(text):
                    """Fix currency amounts that might be split across lines"""
                    # Pattern to match currency amounts with potential line breaks
                    currency_patterns = [
                        r'\$\s*(\d{1,3}(?:\s+\d{3})*(?:\.\d{2})?)',  # $ 35000.00 or $ 35 000.00
                        r'\$(\d{1,3})\s+(\d{3})\s+(\d{2})',  # $35 000 00 (split amounts)
                        r'\$(\d{1,3})\s+(\d{3})\.(\d{2})',  # $35 000.00 (space before decimal)
                        r'\$(\d{1,3})\n\s*(\d{3})\.(\d{2})',  # $35\n000.00 (line break before decimal)
                        r'\$(\d{1,3})\n\s*(\d{3})\n\s*(\d{2})',  # $35\n000\n00 (multiple line breaks)
                        r'\$(\d{1,3})\n\s*(\d{3})',  # $35\n000 (line break in middle)
                        r'(\d{1,3})\n\s*(\d{3})\.(\d{2})',  # 35\n000.00 (missing $ at start)
                        r'(\d{1,3})\n\s*(\d{3})\n\s*(\d{2})',  # 35\n000\n00 (missing $ at start, multiple breaks)
                        r'\\\\\\$(\d{1,3})\n\s*(\d{3})\.(\d{2})',  # \\$15\n000.00 (escaped $ with line break)
                        r'\\\\\\$(\d{1,3})\n\s*(\d{3})\n\s*(\d{2})',  # \\$15\n000\n00 (escaped $ with multiple breaks)
                        r'\\\\\\$(\d{1,3})\n\s*(\d{3})',  # \\$15\n000 (escaped $ with line break)
                    ]
                    
                    for pattern in currency_patterns:
                        def fix_match(match):
                            groups = match.groups()
                            if len(groups) == 1:
                                # Single group - clean up spaces
                                amount = groups[0].replace(' ', '').replace('\n', '')
                                return f"\\${int(amount):,}" if '.' not in amount else f"\\${float(amount):,.2f}"
                            elif len(groups) == 2:
                                # Two groups - reconstruct amount
                                amount_str = ''.join(groups).replace(' ', '').replace('\n', '')
                                return f"\\${int(amount_str):,}" if '.' not in amount_str else f"\\${float(amount_str):,.2f}"
                            else:
                                # Three groups - reconstruct amount with decimal
                                amount_str = ''.join(groups).replace(' ', '').replace('\n', '')
                                return f"\\${float(amount_str):,.2f}"
                        
                        text = re.sub(pattern, fix_match, text, flags=re.MULTILINE)
                    
                    # Additional fix for percentages with line breaks
                    percent_patterns = [
                        r'(\+?\d{1,3})\n\s*(\d{3})\.(\d{2})%',  # +17\n961.41%
                        r'(\+?\d{1,3})\n\s*(\d{3})\n\s*(\d{2})%',  # +17\n961\n41%
                    ]
                    
                    for pattern in percent_patterns:
                        def fix_percent_match(match):
                            groups = match.groups()
                            percent_str = ''.join(groups).replace(' ', '').replace('\n', '')
                            return f"{percent_str}%"
                        
                        text = re.sub(pattern, fix_percent_match, text, flags=re.MULTILINE)
                    
                    return text
                
                # Update the reasoning with formatted text
                if root_cause_match:
                    explanation.reasoning.root_cause = fix_currency_formatting(root_cause_match.group(1).strip())
                if chain_match:
                    explanation.reasoning.chain_of_thought = fix_currency_formatting(chain_match.group(1).strip())
                if recommendation_match:
                    explanation.reasoning.recommendation = fix_currency_formatting(recommendation_match.group(1).strip())
                if evidence_match:
                    evidence_text = evidence_match.group(1).strip()
                    if evidence_text and evidence_text.lower() != 'none':
                        # Fix split currency amounts before parsing
                        # Pattern: "\\$15\n- 000.00" -> "\\$15,000.00"
                        evidence_text = re.sub(r'\\\\\\$(\d+)\n\s*-\s*(\d+\.\d+)', r'\\\\$\\1,\\2', evidence_text)
                        evidence_text = re.sub(r'\\\\\\$(\d+)\n\s*(\d+\.\d+)', r'\\\\$\\1,\\2', evidence_text)
                        
                        # Parse comma-separated evidence and apply currency formatting
                        explanation.reasoning.supporting_evidence = [fix_currency_formatting(e.strip()) for e in evidence_text.split(',')]
                
                formatted_explanations.append(explanation)
                
                # Calculate call duration
                call_duration = time.time() - start_time
                
                # Track cost
                if self.cost_tracker:
                    usage = response.usage
                    query_preview = f"Format text for GL Account {explanation.anomaly.gl_account_id}"
                    
                    self.cost_tracker.track_call(
                        model=format_config["model"],
                        agent="formatting",
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        call_duration=call_duration,
                        query_preview=query_preview
                    )
                
                logger.info(f"✅ Formatted explanation for {explanation.anomaly.gl_account_id}")
            
            logger.info(f"✅ Agent 5 complete: Formatted {len(formatted_explanations)} explanations")
            
            return {
                **state,
                "explanations": formatted_explanations,
                "current_step": "formatting_complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent 5 failed: {e}")
            # If formatting fails, return original explanations
            logger.warning("⚠️  Formatting failed, using original explanations")
            return {
                **state,
                "current_step": "formatting_failed"
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
        
        # camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix specific financial terms (keep together)
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # $amount (remove space)
        text = re.sub(r'(\d)\s+%', r'\1%', text)   # number% (remove space)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _fix_currency_formatting(self, text: str) -> str:
        """Fix currency formatting issues in text"""
        import re
        
        # First, normalize any double-escaped dollar signs to single-escaped
        text = re.sub(r'\\\\\\$', r'\\$', text)
        
        # Comprehensive replacement: replace ALL unescaped dollar signs with escaped ones
        # This handles cases like: $15,000.00, $+14,916.95, $23,500.00, etc.
        text = re.sub(r'(?<!\\)\$', r'\\$', text)
        
        # Fix currency amounts that might be split across lines or missing commas
        currency_patterns = [
            r'\\$(\d{1,3}(?:\s+\d{3})*(?:\.\d{2})?)',  # \$35000.00 or \$35 000.00
            r'\\$(\d{1,3})\n\s*(\d{3})\.(\d{2})',  # \$35\n000.00 (line break before decimal)
            r'\\$(\d{1,3})\n\s*(\d{3})\n\s*(\d{2})',  # \$35\n000\n00 (multiple line breaks)
            r'\\$(\d{1,3})\n\s*(\d{3})',  # \$35\n000 (line break in middle)
            r'(\d{1,3})\n\s*(\d{3})\.(\d{2})',  # 35\n000.00 (missing $ at start)
            r'(\d{1,3})\n\s*(\d{3})\n\s*(\d{2})',  # 35\n000\n00 (missing $ at start, multiple breaks)
        ]
        
        def fix_currency(match):
            groups = match.groups()
            if len(groups) == 1:
                # Single group - clean up spaces
                amount = groups[0].replace(' ', '').replace('\n', '')
                if len(amount) > 3:
                    # Split into integer and decimal parts
                    if '.' in amount:
                        int_part, dec_part = amount.split('.')
                    else:
                        int_part, dec_part = amount, '00'
                    
                    # Add commas to integer part
                    int_part_with_commas = f"{int(int_part):,}"
                    return f"\\${int_part_with_commas}.{dec_part}"
                return f"\\${amount}"
            elif len(groups) == 2:
                # Two groups - reconstruct amount
                amount_str = ''.join(groups).replace(' ', '').replace('\n', '')
                return f"\\${int(amount_str):,}" if '.' not in amount_str else f"\\${float(amount_str):,.2f}"
            else:
                # Three groups - reconstruct amount with decimal
                amount_str = ''.join(groups).replace(' ', '').replace('\n', '')
                return f"\\${float(amount_str):,.2f}"
        
        for pattern in currency_patterns:
            text = re.sub(pattern, fix_currency, text, flags=re.MULTILINE)
        
        return text
    
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


def formatting_agent(state: FinancialAnalysisState) -> FinancialAnalysisState:
    """LangGraph node: Report formatting and text cleanup"""
    agents = FinancialAgents()
    return agents.format_report(state)

