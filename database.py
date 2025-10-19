"""
Database operations for Financial P&L Anomaly Detection Agent
Supports SQLite (MVP) and PostgreSQL (Production)
"""

import sqlite3
import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging

from models import PLTransaction, GLAccount, AnalysisRun
from config import Config

logger = logging.getLogger(__name__)


class Database:
    """Database abstraction layer"""
    
    def __init__(self):
        self.db_type = Config.DATABASE_TYPE
        
        if self.db_type == "sqlite":
            self.db_path = Config.DATABASE_PATH
            self._init_sqlite()
        elif self.db_type == "postgresql":
            import psycopg2
            self.conn = psycopg2.connect(Config.DATABASE_URL)
            self._init_postgresql()
        else:
            raise ValueError(f"Unsupported DATABASE_TYPE: {self.db_type}")
    
    def _init_sqlite(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pl_transactions (
                transaction_id TEXT PRIMARY KEY,
                gl_account_id TEXT NOT NULL,
                transaction_date DATE NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                department TEXT,
                vendor TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gl_accounts (
                account_id TEXT PRIMARY KEY,
                account_name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                typical_min REAL,
                typical_max REAL,
                variance_threshold_pct REAL DEFAULT 15.0,
                is_seasonal BOOLEAN DEFAULT 0,
                seasonal_pattern TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gl_monthly_balances (
                gl_account_id TEXT,
                month DATE,
                net_balance REAL NOT NULL,
                transaction_count INTEGER,
                prev_month_balance REAL,
                variance_amount REAL,
                variance_percent REAL,
                rolling_3mo_avg REAL,
                rolling_6mo_std REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (gl_account_id, month)
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detected_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gl_account_id TEXT NOT NULL,
                detected_month DATE NOT NULL,
                variance_percent REAL,
                variance_amount REAL,
                severity TEXT,
                root_cause TEXT,
                llm_explanation TEXT,
                llm_confidence REAL,
                is_true_anomaly BOOLEAN,
                investigation_notes TEXT,
                resolved_by TEXT,
                resolved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_period TEXT,
                target_month DATE,
                total_accounts_analyzed INTEGER,
                anomalies_detected INTEGER,
                high_severity_count INTEGER,
                medium_severity_count INTEGER,
                execution_time_seconds REAL,
                total_llm_calls INTEGER,
                total_cost_usd REAL,
                avg_faithfulness REAL,
                avg_context_precision REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ SQLite database initialized: {self.db_path}")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database with schema"""
        # Similar to SQLite but with PostgreSQL-specific syntax
        # For production deployment
        pass
    
    def get_connection(self):
        """Get database connection"""
        if self.db_type == "sqlite":
            return sqlite3.connect(self.db_path)
        else:
            return self.conn
    
    def insert_transactions(self, transactions: List[PLTransaction], source_file: str):
        """Insert P&L transactions (idempotent)"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        for txn in transactions:
            cur.execute("""
                INSERT OR IGNORE INTO pl_transactions 
                (transaction_id, gl_account_id, transaction_date, amount, description, 
                 department, vendor, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                txn.transaction_id,
                txn.gl_account_id,
                txn.transaction_date,
                txn.amount,
                txn.description,
                txn.department,
                txn.vendor,
                source_file
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Inserted {len(transactions)} transactions from {source_file}")
    
    def insert_gl_accounts(self, accounts: List[GLAccount]):
        """Insert or update GL account master data"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        for acc in accounts:
            cur.execute("""
                INSERT OR REPLACE INTO gl_accounts
                (account_id, account_name, category, subcategory, typical_min, typical_max,
                 variance_threshold_pct, is_seasonal, seasonal_pattern, description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                acc.account_id,
                acc.account_name,
                acc.category.value,
                acc.subcategory,
                acc.typical_min,
                acc.typical_max,
                acc.variance_threshold_pct,
                acc.is_seasonal,
                acc.seasonal_pattern,
                acc.description
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Upserted {len(accounts)} GL accounts")
    
    def update_monthly_balances(self, target_month: str = None):
        """
        Aggregate transactions and calculate variances
        If target_month is None, aggregates ALL months (for initial load)
        """
        conn = self.get_connection()
        
        # Aggregate transactions into monthly balances
        if target_month:
            # Single month update
            where_clause = f"WHERE SUBSTR(transaction_date, 1, 7) = '{target_month}'"
        else:
            # Aggregate all months (initial load)
            where_clause = ""
        
        query = f"""
            INSERT OR REPLACE INTO gl_monthly_balances 
            (gl_account_id, month, net_balance, transaction_count)
            SELECT 
                gl_account_id,
                DATE(SUBSTR(transaction_date, 1, 7) || '-01') as month,
                SUM(amount) as net_balance,
                COUNT(*) as transaction_count
            FROM pl_transactions
            {where_clause}
            GROUP BY gl_account_id, month
        """
        
        conn.execute(query)
        conn.commit()
        
        # Calculate variance vs previous month for ALL accounts/months
        # Use a simpler approach: iterate through each month
        cur = conn.cursor()
        
        # Get all distinct months
        cur.execute("SELECT DISTINCT month FROM gl_monthly_balances ORDER BY month")
        all_months = [row[0] for row in cur.fetchall()]
        
        # For each month (except first), calculate variance
        for i, current_month in enumerate(all_months):
            if i == 0:
                continue  # Skip first month (no previous to compare)
            
            prev_month = all_months[i - 1]
            
            # Update variance for this month
            cur.execute("""
                UPDATE gl_monthly_balances AS curr
                SET 
                    prev_month_balance = (
                        SELECT net_balance 
                        FROM gl_monthly_balances AS prev
                        WHERE prev.gl_account_id = curr.gl_account_id
                          AND prev.month = ?
                    ),
                    variance_amount = curr.net_balance - (
                        SELECT net_balance 
                        FROM gl_monthly_balances AS prev
                        WHERE prev.gl_account_id = curr.gl_account_id
                          AND prev.month = ?
                    ),
                    variance_percent = (
                        (curr.net_balance - (
                            SELECT net_balance 
                            FROM gl_monthly_balances AS prev
                            WHERE prev.gl_account_id = curr.gl_account_id
                              AND prev.month = ?
                        )) / ABS((
                            SELECT net_balance 
                            FROM gl_monthly_balances AS prev
                            WHERE prev.gl_account_id = curr.gl_account_id
                              AND prev.month = ?
                        ))
                    ) * 100
                WHERE curr.month = ?
            """, (prev_month, prev_month, prev_month, prev_month, current_month))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Updated monthly balances and variances" + (f" for {target_month}" if target_month else " for all months"))
    
    def get_anomalies(self, target_month: str, threshold_pct: float = None) -> List[Dict]:
        """Query anomalies for target month"""
        if threshold_pct is None:
            threshold_pct = Config.ANOMALY_THRESHOLD_MEDIUM
        
        conn = self.get_connection()
        
        query = f"""
            SELECT 
                b.gl_account_id,
                a.account_name,
                SUBSTR(b.month, 1, 7) as current_month,
                b.net_balance as current_balance,
                b.prev_month_balance as previous_balance,
                b.variance_percent,
                b.variance_amount,
                b.transaction_count,
                b.rolling_3mo_avg,
                b.rolling_6mo_std,
                a.typical_min || '-' || a.typical_max as typical_range,
                CASE
                    WHEN ABS(b.variance_percent) > {Config.ANOMALY_THRESHOLD_HIGH} THEN 'high'
                    WHEN ABS(b.variance_percent) > {Config.ANOMALY_THRESHOLD_MEDIUM} THEN 'medium'
                    ELSE 'low'
                END as severity,
                CASE
                    WHEN b.variance_amount > 0 THEN 'increase'
                    ELSE 'decrease'
                END as direction,
                (
                    SELECT COUNT(*) 
                    FROM detected_anomalies 
                    WHERE gl_account_id = b.gl_account_id
                      AND detected_month < b.month
                ) as past_anomaly_count
            FROM gl_monthly_balances b
            JOIN gl_accounts a ON b.gl_account_id = a.account_id
            WHERE SUBSTR(b.month, 1, 7) = '{target_month}'
              AND b.variance_percent IS NOT NULL
              AND ABS(b.variance_percent) >= {threshold_pct}
            ORDER BY ABS(b.variance_percent) DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Calculate previous month
        target_date = datetime.strptime(f"{target_month}-01", "%Y-%m-%d")
        prev_month = (target_date.replace(day=1) - pd.DateOffset(months=1)).strftime("%Y-%m")
        
        # Add previous_month field
        df['previous_month'] = prev_month
        
        return df.to_dict('records')
    
    def store_anomaly_investigation(self, explanation: 'AnomalyExplanation'):
        """Store anomaly investigation results"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO detected_anomalies
            (gl_account_id, detected_month, variance_percent, variance_amount, severity,
             root_cause, llm_explanation, llm_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            explanation.anomaly.gl_account_id,
            f"{explanation.anomaly.current_month}-01",
            explanation.anomaly.variance_percent,
            explanation.anomaly.variance_amount,
            explanation.anomaly.severity.value,
            explanation.reasoning.root_cause,
            explanation.reasoning.chain_of_thought,
            explanation.reasoning.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def store_analysis_run(self, run: AnalysisRun) -> int:
        """Store analysis run metadata"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO analysis_runs
            (analysis_period, target_month, total_accounts_analyzed, anomalies_detected,
             high_severity_count, medium_severity_count, execution_time_seconds,
             total_llm_calls, total_cost_usd, avg_faithfulness, avg_context_precision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.analysis_period,
            f"{run.target_month}-01",
            run.total_accounts_analyzed,
            run.anomalies_detected,
            run.high_severity_count,
            run.medium_severity_count,
            run.execution_time_seconds,
            run.total_llm_calls,
            run.total_cost_usd,
            run.avg_faithfulness,
            run.avg_context_precision
        ))
        
        run_id = cur.lastrowid
        conn.commit()
        conn.close()
        
        return run_id
    
    def get_historical_anomalies(self, account_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve past anomalies for pattern recognition"""
        conn = self.get_connection()
        
        query = """
            SELECT 
                detected_month,
                variance_percent,
                severity,
                root_cause,
                is_true_anomaly
            FROM detected_anomalies
            WHERE gl_account_id = ?
            ORDER BY detected_month DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(account_id, limit))
        conn.close()
        
        return df.to_dict('records')
    
    def update_user_feedback(self, anomaly_id: int, feedback: 'UserFeedback'):
        """Store user feedback on anomaly"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE detected_anomalies
            SET 
                is_true_anomaly = ?,
                investigation_notes = ?,
                resolved_by = ?,
                resolved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            feedback.is_true_anomaly,
            feedback.correct_explanation or feedback.resolution_notes,
            feedback.resolved_by,
            feedback.anomaly_id
        ))
        
        # Adjust threshold if requested
        if feedback.adjust_threshold and not feedback.is_true_anomaly:
            # Get the variance that caused false positive
            cur.execute("SELECT gl_account_id, variance_percent FROM detected_anomalies WHERE id = ?", 
                       (feedback.anomaly_id,))
            account_id, variance_pct = cur.fetchone()
            
            # Increase threshold by 10%
            new_threshold = abs(variance_pct) * 1.1
            
            cur.execute("""
                UPDATE gl_accounts
                SET variance_threshold_pct = ?
                WHERE account_id = ?
            """, (new_threshold, account_id))
            
            logger.info(f"📊 Adjusted threshold for {account_id} to {new_threshold:.2f}%")
        
        conn.commit()
        conn.close()


def init_database():
    """Initialize database (call once during setup)"""
    db = Database()
    logger.info("✅ Database initialized successfully")
    return db


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    init_database()
    print("✅ Database setup complete")

