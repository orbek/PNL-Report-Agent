"""
LangGraph Workflow for Financial P&L Anomaly Detection
Orchestrates 4 specialized agents in sequence with state management
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage
import logging
from datetime import datetime
import time

from models import FinancialAnalysisState, AnomalyReport, AnalysisRun
from agents import FinancialAgents
from config import Config
from database import Database
from cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class FinancialAnomalyWorkflow:
    """LangGraph workflow manager"""
    
    def __init__(self, model: str = None, enable_cost_tracking: bool = True):
        self.model = model or Config.DEFAULT_MODEL
        self.cost_tracker = CostTracker(f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if enable_cost_tracking else None
        self.agents = FinancialAgents(model=self.model, cost_tracker=self.cost_tracker)
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow:
        START → Ingest → Detect → Retrieve → Report → END
        """
        logger.info("🔨 Building LangGraph workflow...")
        
        # Create graph
        workflow = StateGraph(FinancialAnalysisState)
        
        # Add nodes (agents)
        workflow.add_node("ingest", self.agents.ingest_data)
        workflow.add_node("detect", self.agents.detect_anomalies)
        workflow.add_node("retrieve", self.agents.retrieve_context)
        workflow.add_node("report", self.agents.generate_explanations)
        
        # Add edges (workflow sequence)
        workflow.add_edge(START, "ingest")
        workflow.add_edge("ingest", "detect")
        workflow.add_edge("detect", "retrieve")
        workflow.add_edge("retrieve", "report")
        workflow.add_edge("report", END)
        
        logger.info("✅ LangGraph workflow built")
        
        return workflow
    
    def run_analysis(
        self, 
        pl_file_path: str, 
        gl_master_path: str = None,
        target_month: str = None
    ) -> AnomalyReport:
        """
        Execute complete anomaly detection workflow
        
        Args:
            pl_file_path: Path to P&L CSV file
            gl_master_path: Path to GL accounts CSV (optional if already in DB)
            target_month: Month to analyze in YYYY-MM format (auto-detected if not provided)
        
        Returns:
            AnomalyReport object with all findings
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("🚀 Starting Financial Anomaly Analysis")
        logger.info(f"🤖 Using Model: {self.model}")
        if self.cost_tracker:
            logger.info("💰 Cost tracking enabled")
        logger.info("=" * 80)
        
        # Auto-detect target month from P&L file if not provided
        if not target_month:
            import pandas as pd
            df = pd.read_csv(pl_file_path)
            target_month = pd.to_datetime(df['date']).max().strftime('%Y-%m')
            logger.info(f"📅 Auto-detected target month: {target_month}")
        
        # Use default GL master path if not provided
        if not gl_master_path:
            gl_master_path = str(Config.GL_MASTER_PATH)
        
        # Determine analysis period (target month only for reporting)
        analysis_period = target_month
        
        # Initial state
        initial_state: FinancialAnalysisState = {
            "pl_file_path": pl_file_path,
            "gl_master_path": gl_master_path,
            "analysis_period": analysis_period,
            "target_month": target_month,
            "pl_transactions": [],
            "gl_accounts": [],
            "monthly_balances": [],
            "anomalies": [],
            "anomaly_contexts": {},
            "explanations": [],
            "final_report": None,
            "messages": [
                HumanMessage(content=f"Analyze P&L for {target_month}")
            ],
            "current_step": "initialized",
            "errors": []
        }
        
        try:
            # Run workflow
            logger.info(f"🔄 Executing workflow for {target_month}...")
            
            final_state = self.app.invoke(initial_state)
            
            # Check for errors
            if final_state.get("errors"):
                logger.error(f"❌ Workflow completed with errors: {final_state['errors']}")
                raise Exception(f"Workflow errors: {'; '.join(final_state['errors'])}")
            
            # Generate final report
            execution_time = time.time() - start_time
            
            report = AnomalyReport(
                analysis_period=analysis_period,
                target_month=target_month,
                total_accounts_analyzed=len(final_state["gl_accounts"]),
                anomalies_detected=len(final_state["anomalies"]),
                high_severity_count=sum(1 for a in final_state["anomalies"] if a.severity.value == "high"),
                medium_severity_count=sum(1 for a in final_state["anomalies"] if a.severity.value == "medium"),
                low_severity_count=sum(1 for a in final_state["anomalies"] if a.severity.value == "low"),
                explanations=final_state["explanations"],
                execution_time_seconds=round(execution_time, 2),
                total_llm_calls=len(final_state["anomalies"]) + 2,  # Approximate
                total_cost_usd=None  # TODO: Track actual cost from LangSmith
            )
            
            logger.info("=" * 80)
            logger.info("✅ Analysis Complete!")
            logger.info(f"   Total Anomalies: {report.anomalies_detected}")
            logger.info(f"   High Severity: {report.high_severity_count}")
            logger.info(f"   Medium Severity: {report.medium_severity_count}")
            logger.info(f"   Execution Time: {execution_time:.1f}s")
            logger.info("=" * 80)
            
            # Save report to database
            db = Database()
            db.store_analysis_run(AnalysisRun(
                analysis_period=report.analysis_period,
                target_month=report.target_month,
                total_accounts_analyzed=report.total_accounts_analyzed,
                anomalies_detected=report.anomalies_detected,
                high_severity_count=report.high_severity_count,
                medium_severity_count=report.medium_severity_count,
                execution_time_seconds=report.execution_time_seconds,
                total_llm_calls=report.total_llm_calls or 0
            ))
            
            # Print cost summary if tracking is enabled
            if self.cost_tracker:
                self.cost_tracker.print_summary()
                
                # Save cost report
                cost_report_path = f"reports/cost_report_{self.cost_tracker.run_id}.json"
                self.cost_tracker.save_report(cost_report_path)
                logger.info(f"💰 Cost report saved to {cost_report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise
    
    def visualize_graph(self, output_path: str = "workflow_graph.png"):
        """Generate visualization of LangGraph workflow"""
        try:
            from IPython.display import Image
            
            # Get graph visualization
            graph_image = self.app.get_graph().draw_mermaid_png()
            
            with open(output_path, 'wb') as f:
                f.write(graph_image)
            
            logger.info(f"📊 Workflow graph saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")


# Convenience function for direct use
def analyze_pl_report(
    pl_csv_path: str,
    target_month: str = None,
    output_report_path: str = None
) -> AnomalyReport:
    """
    Main entry point for P&L analysis
    
    Args:
        pl_csv_path: Path to P&L CSV file
        target_month: Month to analyze (YYYY-MM), auto-detected if None
        output_report_path: Where to save markdown report (optional)
    
    Returns:
        AnomalyReport object
    """
    # Create and run workflow
    workflow = FinancialAnomalyWorkflow()
    report = workflow.run_analysis(pl_csv_path, target_month=target_month)
    
    # Save report if path provided
    if output_report_path:
        markdown = report.to_markdown()
        
        with open(output_report_path, 'w') as f:
            f.write(markdown)
        
        logger.info(f"📄 Report saved to {output_report_path}")
    
    return report


if __name__ == "__main__":
    # Test workflow
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    workflow = FinancialAnomalyWorkflow()
    
    # Visualize graph
    workflow.visualize_graph()
    
    print("✅ Workflow ready. Use: analyze_pl_report('path/to/pl.csv')")

