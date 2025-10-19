#!/usr/bin/env python3
"""
Generate November 2025 P&L data for testing incremental monthly workflow
Includes new anomalies different from October
"""

import pandas as pd
import random
from datetime import datetime
from pathlib import Path


def generate_november_2025_pl():
    """
    Generate November 2025 P&L transactions with new anomalies
    """
    transactions = []
    txn_id_counter = 10001  # Start after October transactions
    
    # Base patterns for each account (from sample_data_generator.py)
    account_patterns = {
        # Revenue
        '4000': {'base': 65000, 'variance': 5000, 'nov_amount': 72000},  # Normal increase
        '4100': {'base': 22000, 'variance': 8000, 'nov_amount': 55000},  # 🔴 ANOMALY: Large consulting project
        '4200': {'base': 8000, 'variance': 2000, 'nov_amount': 9200},    # Normal
        
        # Operating Expenses
        '5000': {'base': 12500, 'variance': 0, 'nov_amount': 12500},     # Back to normal (deposit was one-time)
        '5100': {'base': 1800, 'variance': 800, 'nov_amount': 6500},     # 🔴 ANOMALY: Bulk supplies purchase
        '5200': {'base': 9500, 'variance': 500, 'nov_amount': 19000},    # 🔴 ANOMALY: New software licenses
        '5300': {'base': 3500, 'variance': 200, 'nov_amount': 3600},     # Back to normal
        
        # Payroll
        '5500': {'base': 85000, 'variance': 2000, 'nov_amount': 87000},  # Normal (bonuses were one-time)
        '5510': {'base': 13000, 'variance': 500, 'nov_amount': 13200},   # Normal
        '5520': {'base': 17000, 'variance': 1000, 'nov_amount': 18500},  # Normal
        '5530': {'base': 15000, 'variance': 8000, 'nov_amount': 28000},  # 🟡 ANOMALY: New contractors hired
        
        # Business Development
        '6200': {'base': 3500, 'variance': 1200, 'nov_amount': 4200},    # Back to normal
        '6300': {'base': 2500, 'variance': 1500, 'nov_amount': 8500},    # 🔴 ANOMALY: Team training offsite
        
        # Facilities
        '7100': {'base': 2200, 'variance': 400, 'nov_amount': 2400},     # Normal
        '7200': {'base': 1800, 'variance': 1200, 'nov_amount': 2100},    # Back to normal
        '7300': {'base': 2200, 'variance': 100, 'nov_amount': 2250},     # Normal
        
        # Marketing
        '8000': {'base': 10000, 'variance': 4000, 'nov_amount': 18000},  # 🟡 ANOMALY: Black Friday campaign
        '8100': {'base': 5000, 'variance': 2500, 'nov_amount': 5800},    # Normal
        '8200': {'base': 5000, 'variance': 5000, 'nov_amount': 0.01},    # 🔴 ANOMALY: No events this month (minimal amount)
        
        # Professional Services  
        '9000': {'base': 4000, 'variance': 3000, 'nov_amount': 5200},    # Back to normal
        '9100': {'base': 4000, 'variance': 1500, 'nov_amount': 4500},    # Normal
        '9200': {'base': 6000, 'variance': 3000, 'nov_amount': 7500},    # Normal
        
        # Other
        '3000': {'base': 20000, 'variance': 5000, 'nov_amount': 22000},  # Normal
        '3100': {'base': 11000, 'variance': 2000, 'nov_amount': 12500},  # Normal
        '9500': {'base': 1200, 'variance': 600, 'nov_amount': 250},      # 🟡 ANOMALY: Switched to lower-fee bank
        '9600': {'base': 3500, 'variance': 0, 'nov_amount': 3500},       # Fixed
        '9700': {'base': 1000, 'variance': 1000, 'nov_amount': 15000}    # 🔴 ANOMALY: Large charity donation
    }
    
    # Anomaly descriptions
    descriptions = {
        '4100': "🔴 ANOMALY: Major enterprise consulting engagement - ABC Corp",
        '5100': "🔴 ANOMALY: Bulk order for new office expansion (chairs, desks, monitors)",
        '5200': "🔴 ANOMALY: Annual enterprise license renewals + new tools",
        '5530': "🟡 ANOMALY: 5 new contractors onboarded for Q4 projects",
        '6300': "🔴 ANOMALY: Company-wide team building retreat - Lake Tahoe",
        '8000': "🟡 ANOMALY: Black Friday / Cyber Monday marketing blitz",
        '8200': "🔴 ANOMALY: No trade shows scheduled this month (vs typical $5K) - minimal admin cost",
        '9500': "🟡 ANOMALY: Switched banks - lower merchant fees",
        '9700': "🔴 ANOMALY: Year-end corporate giving campaign ($15K to local schools)"
    }
    
    # November 15, 2025 (mid-month)
    transaction_date = "2025-11-15"
    
    for account_id, pattern in account_patterns.items():
        amount = pattern['nov_amount']
        
        # Get description
        if account_id in descriptions:
            desc = descriptions[account_id]
        else:
            desc = f"Monthly {account_id} - November 2025"
        
        transactions.append({
            'transaction_id': f"TXN{txn_id_counter:05d}",
            'gl_account_id': account_id,
            'date': transaction_date,
            'amount': round(amount, 2),
            'description': desc,
            'department': 'Operations',
            'vendor': f"Vendor_{account_id}"
        })
        
        txn_id_counter += 1
    
    return pd.DataFrame(transactions)


def create_november_file(output_dir: str = "./data/pl_reports"):
    """Create November 2025 P&L CSV file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🗓️  Generating November 2025 P&L Data")
    print("=" * 80)
    
    # Generate data
    nov_df = generate_november_2025_pl()
    
    # Save file
    filepath = output_path / "pl_2025-11.csv"
    nov_df.to_csv(filepath, index=False)
    
    print(f"\n✅ Created: {filepath}")
    print(f"   Transactions: {len(nov_df)}")
    
    # Show expected anomalies
    print("\n" + "=" * 80)
    print("📊 EXPECTED ANOMALIES vs OCTOBER 2025")
    print("=" * 80)
    print("""
HIGH SEVERITY (>30% variance):
1. GL 4100 (Consulting Revenue): +190% - Major enterprise client
2. GL 5100 (Office Supplies): +241% - Bulk purchase for expansion
3. GL 5200 (Software): +93% - Annual license renewals
4. GL 6300 (Training): +738% - Company retreat
5. GL 8200 (Trade Shows): -100% - No events this month
6. GL 9700 (Charitable): +702% - Year-end giving campaign

MEDIUM SEVERITY (15-30% variance):
7. GL 5530 (Contractors): +522% - New hires for Q4
8. GL 8000 (Marketing): +19% - Black Friday campaign
9. GL 9500 (Bank Fees): -70% - Switched to lower-fee bank

EXPECTED (Should NOT flag):
- GL 5000 (Rent): Back to $12,500 (Oct spike was one-time)
- GL 6200 (Travel): Back to $4,200 (Oct conference was one-time)
- GL 9000 (Legal): Back to $5,200 (Oct litigation was one-time)

Total Expected Detections: ~9 anomalies
""")
    
    print("\n" + "=" * 80)
    print("🚀 NEXT STEPS")
    print("=" * 80)
    print("""
Test incremental monthly workflow:

1. Run analysis on November data:
   python main.py analyze data/pl_reports/pl_2025-11.csv --month 2025-11

2. System will:
   ✅ Load only November transactions (no re-processing of Oct)
   ✅ Aggregate November balances
   ✅ Calculate Nov vs Oct variance
   ✅ Detect ~9 new anomalies
   ✅ Generate explanations
   ✅ Append to database (Oct data preserved)

3. View report:
   cat reports/anomaly_report_2025-11_*.md

4. Check database growth:
   python main.py stats
   # Should show 351 + 27 = 378 transactions total
""")


if __name__ == "__main__":
    create_november_file()

