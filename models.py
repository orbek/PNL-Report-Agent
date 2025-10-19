"""
Pydantic models for Financial P&L Anomaly Detection Agent
Defines schemas for structured data validation and LLM outputs
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class CategoryEnum(str, Enum):
    """GL Account Categories"""
    EXPENSE = "expense"
    REVENUE = "revenue"
    ASSET = "asset"
    LIABILITY = "liability"


class SeverityEnum(str, Enum):
    """Anomaly Severity Levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PLTransaction(BaseModel):
    """P&L Transaction Entry"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    gl_account_id: str = Field(..., description="GL account code")
    transaction_date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    amount: float = Field(..., description="Transaction amount (positive for debit, negative for credit)")
    description: str = Field(..., description="Transaction description")
    department: Optional[str] = Field(None, description="Department or cost center")
    vendor: Optional[str] = Field(None, description="Vendor or payee name")
    
    @field_validator("transaction_date")
    @classmethod
    def validate_date_format(cls, v):
        """Ensure date is valid YYYY-MM-DD format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
    
    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v):
        """Ensure amount is valid number"""
        if v == 0:
            raise ValueError("Transaction amount cannot be zero")
        return round(v, 2)


class GLAccount(BaseModel):
    """General Ledger Account Master"""
    account_id: str = Field(..., description="GL account identifier")
    account_name: str = Field(..., description="Human-readable account name")
    category: CategoryEnum = Field(..., description="Account category")
    subcategory: Optional[str] = Field(None, description="Account subcategory")
    typical_min: Optional[float] = Field(None, description="Typical minimum monthly balance")
    typical_max: Optional[float] = Field(None, description="Typical maximum monthly balance")
    variance_threshold_pct: float = Field(15.0, description="Custom variance threshold percentage")
    is_seasonal: bool = Field(False, description="Does this account have seasonal patterns?")
    seasonal_pattern: Optional[str] = Field(None, description="Description of seasonal pattern")
    description: Optional[str] = Field(None, description="Detailed account purpose and policies")
    
    @field_validator("variance_threshold_pct")
    @classmethod
    def validate_threshold(cls, v):
        """Ensure threshold is reasonable"""
        if v < 0 or v > 100:
            raise ValueError("Variance threshold must be between 0 and 100")
        return v


class MonthlyBalance(BaseModel):
    """Aggregated monthly GL account balance"""
    gl_account_id: str
    month: str = Field(..., description="Month in YYYY-MM format")
    net_balance: float = Field(..., description="Net balance for the month")
    transaction_count: int = Field(..., ge=0)
    prev_month_balance: Optional[float] = None
    variance_amount: Optional[float] = None
    variance_percent: Optional[float] = None
    rolling_3mo_avg: Optional[float] = None
    rolling_6mo_std: Optional[float] = None
    
    @field_validator("month")
    @classmethod
    def validate_month_format(cls, v):
        """Ensure month is YYYY-MM format"""
        try:
            datetime.strptime(f"{v}-01", "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Month must be in YYYY-MM format, got: {v}")


class AnomalyFlag(BaseModel):
    """Detected Anomaly with Severity Classification"""
    gl_account_id: str
    account_name: str
    current_month: str = Field(..., description="YYYY-MM format")
    previous_month: str = Field(..., description="YYYY-MM format")
    current_balance: float
    previous_balance: float
    variance_percent: float = Field(..., description="Percentage change")
    variance_amount: float = Field(..., description="Absolute change")
    severity: SeverityEnum = Field(..., description="Anomaly severity level")
    direction: str = Field(..., description="increase or decrease")
    
    # Statistical context
    rolling_3mo_avg: Optional[float] = None
    rolling_6mo_std: Optional[float] = None
    z_score: Optional[float] = Field(None, description="Statistical Z-score")
    is_statistical_outlier: bool = Field(False, description="Outside 2 std devs?")
    
    # Metadata
    transaction_count: Optional[int] = None
    typical_range: Optional[str] = None
    
    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        """Ensure direction is valid"""
        v_lower = v.lower().strip()
        if v_lower not in {"increase", "decrease"}:
            raise ValueError("Direction must be 'increase' or 'decrease'")
        return v_lower
    
    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        """Ensure severity is valid enum"""
        if isinstance(v, str):
            return SeverityEnum(v.lower())
        return v


class GLContext(BaseModel):
    """Retrieved GL Account Context from RAG"""
    account_id: str
    account_documentation: str = Field(..., description="Primary GL account documentation")
    similar_cases: List[str] = Field(default_factory=list, description="Similar historical patterns")
    source_files: List[str] = Field(default_factory=list, description="Documentation source files")
    past_anomaly_count: int = Field(0, description="Number of past anomalies for this account")


class FinancialReasoning(BaseModel):
    """Chain-of-Thought Financial Analysis"""
    chain_of_thought: str = Field(..., description="Step-by-step reasoning process")
    root_cause: str = Field(..., description="Primary cause of variance")
    is_expected_variance: bool = Field(..., description="Is this expected or truly anomalous?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    supporting_evidence: List[str] = Field(default_factory=list, description="Data points supporting analysis")
    recommendation: str = Field(..., description="Recommended action")
    requires_immediate_attention: bool = Field(..., description="Needs urgent review?")


class AnomalyExplanation(BaseModel):
    """Complete Anomaly Investigation Result"""
    anomaly: AnomalyFlag
    reasoning: FinancialReasoning
    retrieved_context: GLContext
    investigation_timestamp: datetime = Field(default_factory=lambda: datetime.now())
    
    def to_report_section(self) -> str:
        """Format as report section with proper markdown"""
        direction_symbol = "📈" if self.anomaly.direction == "increase" else "📉"
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[self.anomaly.severity.value]
        
        # Format supporting evidence
        evidence_list = "\n".join(f"- {evidence}" for evidence in self.reasoning.supporting_evidence) if self.reasoning.supporting_evidence else "- Analysis based on variance data and GL documentation"
        
        # Format attention flag
        attention_flag = "\n\n⚠️ **REQUIRES IMMEDIATE ATTENTION**\n" if self.reasoning.requires_immediate_attention else ""
        
        return f"""
### {severity_emoji} GL Account {self.anomaly.gl_account_id} - {self.anomaly.account_name}

**Variance**: {direction_symbol} {self.anomaly.variance_percent:+.2f}% (${self.anomaly.variance_amount:+,.2f})  
**Current Month ({self.anomaly.current_month})**: ${self.anomaly.current_balance:,.2f}  
**Previous Month ({self.anomaly.previous_month})**: ${self.anomaly.previous_balance:,.2f}

**Root Cause Analysis**:  
{self.reasoning.root_cause}

**Detailed Reasoning**:  
{self.reasoning.chain_of_thought}

**Expected vs. Anomalous**: {"Expected variance (seasonal/one-time)" if self.reasoning.is_expected_variance else "True anomaly - requires investigation"}

**Supporting Evidence**:  
{evidence_list}

**Recommendation**:  
{self.reasoning.recommendation}

**Confidence**: {self.reasoning.confidence:.1%}{attention_flag}

---

"""


class AnomalyReport(BaseModel):
    """Final Monthly Anomaly Report"""
    analysis_period: str = Field(..., description="Period analyzed (e.g., '2025-01 to 2025-03')")
    target_month: str = Field(..., description="Month analyzed (YYYY-MM)")
    total_accounts_analyzed: int
    anomalies_detected: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    
    explanations: List[AnomalyExplanation]
    
    # Metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now())
    execution_time_seconds: Optional[float] = None
    total_llm_calls: Optional[int] = None
    total_cost_usd: Optional[float] = None
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        high_severity_items = [e for e in self.explanations if e.anomaly.severity == SeverityEnum.HIGH]
        medium_severity_items = [e for e in self.explanations if e.anomaly.severity == SeverityEnum.MEDIUM]
        
        report = f"""
# Financial Anomaly Analysis Report
**Period**: {self.analysis_period}
**Target Month**: {self.target_month}
**Generated**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total GL Accounts Analyzed**: {self.total_accounts_analyzed}
- **Anomalies Detected**: {self.anomalies_detected}
  - 🔴 High Severity: {self.high_severity_count}
  - 🟡 Medium Severity: {self.medium_severity_count}
  - 🟢 Low Severity: {self.low_severity_count}

{"---" if self.execution_time_seconds else ""}
{"**Performance**: " + f"{self.execution_time_seconds:.1f}s | {self.total_llm_calls} LLM calls" + (f" | ${self.total_cost_usd:.2f}" if self.total_cost_usd else "") if self.execution_time_seconds else ""}

---

## Critical Items Requiring Review

{chr(10).join([exp.to_report_section() for exp in high_severity_items])}

{"## Medium Priority Items" if medium_severity_items else ""}

{chr(10).join([exp.to_report_section() for exp in medium_severity_items[:3]])}  

{"*(Showing top 3 medium severity items)*" if len(medium_severity_items) > 3 else ""}

---

## Recommendations

1. **Immediate Action Required**: {self.high_severity_count} high-severity anomalies need investigation
2. **Review Within 48 Hours**: {self.medium_severity_count} medium-severity items
3. **Monitor**: {self.low_severity_count} low-severity variances (informational)

**Next Steps**:
- Finance team to review all high-severity items
- Update GL account thresholds if false positives detected
- Document resolutions in system for future pattern recognition

---

*Report generated by Financial P&L Anomaly Detection Agent powered by GPT-5*
"""
        return report
    
    def to_html(self) -> str:
        """Generate HTML report for email"""
        # Convert markdown to HTML (simplified)
        try:
            import markdown
            return markdown.markdown(self.to_markdown())
        except ImportError:
            # Fallback if markdown not installed
            return f"<pre>{self.to_markdown()}</pre>"


class UserFeedback(BaseModel):
    """User feedback on anomaly investigation"""
    anomaly_id: int = Field(..., description="Database ID of detected anomaly")
    is_true_anomaly: bool = Field(..., description="Was this a real anomaly?")
    correct_explanation: Optional[str] = Field(None, description="User's explanation if incorrect")
    adjust_threshold: bool = Field(False, description="Should threshold be adjusted?")
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None


class AnalysisRun(BaseModel):
    """Metadata for each analysis run"""
    run_id: Optional[int] = None
    analysis_period: str
    target_month: str
    total_accounts_analyzed: int
    anomalies_detected: int
    high_severity_count: int
    medium_severity_count: int
    execution_time_seconds: float
    total_llm_calls: int
    total_cost_usd: Optional[float] = None
    
    # Evaluation metrics (if Ragas enabled)
    avg_faithfulness: Optional[float] = None
    avg_context_precision: Optional[float] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now())


# ============================================================================
# LangGraph State Definition
# ============================================================================

from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import BaseMessage
import operator


class FinancialAnalysisState(TypedDict):
    """Shared state across all agents in LangGraph workflow"""
    
    # Input
    pl_file_path: str
    gl_master_path: str
    analysis_period: str  # "2025-01 to 2025-03"
    target_month: str  # "2025-03"
    
    # Agent 1 outputs
    pl_transactions: List[PLTransaction]
    gl_accounts: List[GLAccount]
    
    # Agent 2 outputs
    monthly_balances: List[MonthlyBalance]
    anomalies: List[AnomalyFlag]
    
    # Agent 3 outputs
    anomaly_contexts: Dict[str, GLContext]  # Keyed by gl_account_id
    
    # Agent 4 outputs
    explanations: List[AnomalyExplanation]
    final_report: Optional[AnomalyReport]
    
    # Workflow metadata
    messages: Annotated[List[BaseMessage], operator.add]  # Conversation history
    current_step: str
    errors: List[str]

