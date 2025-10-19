#!/usr/bin/env python3
"""
Financial P&L Anomaly Detection Agent - Main CLI
Entry point for all operations: init, analyze, feedback
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from config import Config
from database import init_database
from vector_store import VectorStoreManager, build_initial_gl_knowledge_base
from workflow import FinancialAnomalyWorkflow
from sample_data_generator import create_sample_data_files
import pandas as pd

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.AUDIT_LOG_PATH) if Config.ENABLE_AUDIT_LOG else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


def cmd_init(args):
    """Initialize database and vector store"""
    print("=" * 80)
    print("🚀 Initializing Financial Anomaly Detection Agent")
    print("=" * 80)
    
    # Step 1: Initialize database
    print("\n📊 Step 1: Setting up database...")
    init_database()
    print("✅ Database initialized")
    
    # Step 2: Create vector store
    print("\n📚 Step 2: Setting up vector store...")
    vector_manager = VectorStoreManager()
    print("✅ Vector store ready")
    
    # Step 3: Check for GL master data
    if Config.GL_MASTER_PATH.exists():
        print(f"\n📂 Step 3: Loading GL accounts from {Config.GL_MASTER_PATH}...")
        gl_df = pd.read_csv(Config.GL_MASTER_PATH)
        
        # Build initial knowledge base if GL docs don't exist
        if not any(Config.GL_DOCS_DIR.glob("*.md")):
            print("📝 Generating GL documentation files...")
            build_initial_gl_knowledge_base(gl_df)
            
            # Rebuild vector store with new docs
            print("🔄 Indexing GL documentation...")
            vector_manager.rebuild_index()
        
        print(f"✅ Loaded {len(gl_df)} GL accounts")
    else:
        print(f"\n⚠️  GL master file not found: {Config.GL_MASTER_PATH}")
        print("Run with --generate-sample to create sample data")
    
    print("\n" + "=" * 80)
    print("✅ Initialization Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Place your P&L CSV files in:", Config.PL_DATA_DIR)
    print("2. Run analysis: python main.py analyze <path-to-pl.csv>")
    print("\nOr generate sample data: python main.py --generate-sample")


def cmd_analyze(args):
    """Run anomaly detection analysis"""
    pl_path = Path(args.pl_file)
    
    if not pl_path.exists():
        print(f"❌ P&L file not found: {pl_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"🔍 Analyzing P&L Report: {pl_path.name}")
    print("=" * 80)
    
    # Determine target month
    target_month = args.month
    if not target_month:
        # Auto-detect from filename or data
        df = pd.read_csv(pl_path)
        target_month = pd.to_datetime(df['date']).max().strftime('%Y-%m')
        print(f"📅 Auto-detected target month: {target_month}")
    
    # Run analysis with selected model
    try:
        # Create workflow with model selection
        workflow = FinancialAnomalyWorkflow(
            model=args.model,
            enable_cost_tracking=True
        )
        
        report = workflow.run_analysis(
            pl_file_path=str(pl_path),
            gl_master_path=str(Config.GL_MASTER_PATH),
            target_month=target_month
        )
        
        # Save report
        report_filename = f"anomaly_report_{target_month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = Config.REPORTS_OUTPUT_DIR / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report.to_markdown())
        
        print("\n" + "=" * 80)
        print("📄 REPORT SUMMARY")
        print("=" * 80)
        print(f"Total Anomalies: {report.anomalies_detected}")
        print(f"  - 🔴 High Severity: {report.high_severity_count}")
        print(f"  - 🟡 Medium Severity: {report.medium_severity_count}")
        print(f"  - 🟢 Low Severity: {report.low_severity_count}")
        print(f"\nExecution Time: {report.execution_time_seconds:.1f}s")
        print(f"LLM Calls: {report.total_llm_calls}")
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Print high-severity items
        if report.high_severity_count > 0:
            print("\n" + "=" * 80)
            print("🔴 HIGH SEVERITY ANOMALIES")
            print("=" * 80)
            
            for exp in report.explanations:
                if exp.anomaly.severity.value == "high":
                    print(f"\n{exp.anomaly.account_name} ({exp.anomaly.gl_account_id})")
                    print(f"  Variance: {exp.anomaly.variance_percent:+.2f}%")
                    print(f"  Root Cause: {exp.reasoning.root_cause}")
                    print(f"  Action Required: {exp.reasoning.recommendation[:100]}...")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_generate_sample(args):
    """Generate sample data for testing"""
    print("🔨 Generating sample data...")
    create_sample_data_files(output_dir="./data")
    print("\n✅ Sample data created in ./data/")
    print("\nNext steps:")
    print("1. python main.py init")
    print("2. python main.py analyze data/pl_12_months_historical.csv")


def cmd_rebuild_index(args):
    """Rebuild vector store index"""
    print("🔄 Rebuilding vector store index...")
    vector_manager = VectorStoreManager()
    vector_manager.rebuild_index()
    print("✅ Vector store rebuilt")


def cmd_stats(args):
    """Show database statistics"""
    from database import Database
    
    db = Database()
    conn = db.get_connection()
    
    print("=" * 80)
    print("📊 DATABASE STATISTICS")
    print("=" * 80)
    
    # Count tables
    queries = {
        "P&L Transactions": "SELECT COUNT(*) FROM pl_transactions",
        "GL Accounts": "SELECT COUNT(*) FROM gl_accounts",
        "Monthly Balances": "SELECT COUNT(*) FROM gl_monthly_balances",
        "Detected Anomalies": "SELECT COUNT(*) FROM detected_anomalies",
        "Analysis Runs": "SELECT COUNT(*) FROM analysis_runs"
    }
    
    for label, query in queries.items():
        cur = conn.cursor()
        cur.execute(query)
        count = cur.fetchone()[0]
        print(f"{label:.<40} {count:>6} rows")
    
    # Latest analysis
    cur = conn.cursor()
    cur.execute("""
        SELECT target_month, anomalies_detected, execution_time_seconds
        FROM analysis_runs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    
    result = cur.fetchone()
    if result:
        print("\n" + "-" * 80)
        print("Latest Analysis:")
        print(f"  Month: {result[0]}")
        print(f"  Anomalies: {result[1]}")
        print(f"  Time: {result[2]:.1f}s")
    
    conn.close()
    
    print("=" * 80)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Financial P&L Anomaly Detection Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize system
  python main.py init
  
  # Generate sample data
  python main.py --generate-sample
  
  # Analyze P&L report (default GPT-4)
  python main.py analyze data/pl_2025_03.csv
  
  # Analyze with GPT-5
  python main.py analyze data/pl_2025_03.csv --model gpt-5
  
  # Analyze specific month
  python main.py analyze data/pl_historical.csv --month 2025-03 --model gpt-4o
  
  # View statistics
  python main.py stats
  
  # Rebuild vector store
  python main.py rebuild-index
"""
    )
    
    # Global options
    parser.add_argument('--generate-sample', action='store_true',
                       help='Generate sample data files for testing')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database and vector store')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze P&L report')
    analyze_parser.add_argument('pl_file', help='Path to P&L CSV file')
    analyze_parser.add_argument('--month', '-m', help='Target month in YYYY-MM format (auto-detected if not provided)')
    analyze_parser.add_argument('--model', choices=Config.SUPPORTED_MODELS, default=Config.DEFAULT_MODEL,
                               help=f'AI model to use (default: {Config.DEFAULT_MODEL})')
    analyze_parser.add_argument('--threshold', '-t', type=float, default=15.0,
                               help='Variance threshold percentage (default: 15.0)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Rebuild index command
    rebuild_parser = subparsers.add_parser('rebuild-index', help='Rebuild vector store index')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle generate-sample flag
    if args.generate_sample:
        cmd_generate_sample(args)
        return
    
    # Handle commands
    if args.command == 'init':
        cmd_init(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'rebuild-index':
        cmd_rebuild_index(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

