"""
Generate sample P&L and GL data for testing
Creates realistic financial data with intentional anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random


def generate_sample_gl_accounts():
    """Generate comprehensive GL account master data with 25+ accounts"""
    accounts = [
        # REVENUE ACCOUNTS
        {
            'account_id': '4000',
            'account_name': 'Product Sales Revenue',
            'category': 'revenue',
            'subcategory': 'Operating Revenue',
            'typical_min': 50000,
            'typical_max': 80000,
            'variance_threshold_pct': 10.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Q4 peak, January dip',
            'description': 'SaaS subscription and product sales revenue'
        },
        {
            'account_id': '4100',
            'account_name': 'Consulting Services Revenue',
            'category': 'revenue',
            'subcategory': 'Professional Services',
            'typical_min': 15000,
            'typical_max': 35000,
            'variance_threshold_pct': 20.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Professional consulting and advisory services'
        },
        {
            'account_id': '4200',
            'account_name': 'License & Royalties',
            'category': 'revenue',
            'subcategory': 'Passive Income',
            'typical_min': 5000,
            'typical_max': 12000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Software licensing and intellectual property royalties'
        },
        
        # OPERATING EXPENSES
        {
            'account_id': '5000',
            'account_name': 'Rent Expense',
            'category': 'expense',
            'subcategory': 'Facilities',
            'typical_min': 10000,
            'typical_max': 15000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Monthly office lease payments - Fixed cost'
        },
        {
            'account_id': '5100',
            'account_name': 'Office Supplies',
            'category': 'expense',
            'subcategory': 'Operating',
            'typical_min': 1000,
            'typical_max': 3000,
            'variance_threshold_pct': 30.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Paper, pens, equipment, office consumables'
        },
        {
            'account_id': '5200',
            'account_name': 'Software & Subscriptions',
            'category': 'expense',
            'subcategory': 'Technology',
            'typical_min': 8000,
            'typical_max': 12000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'SaaS tools, cloud services, software licenses'
        },
        {
            'account_id': '5300',
            'account_name': 'Insurance',
            'category': 'expense',
            'subcategory': 'Operating',
            'typical_min': 3000,
            'typical_max': 4000,
            'variance_threshold_pct': 10.0,
            'is_seasonal': False,
            'seasonal_pattern': 'Annual renewal in January',
            'description': 'General liability, D&O, and property insurance'
        },
        
        # PAYROLL
        {
            'account_id': '5500',
            'account_name': 'Salaries & Wages',
            'category': 'expense',
            'subcategory': 'Payroll',
            'typical_min': 80000,
            'typical_max': 90000,
            'variance_threshold_pct': 10.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Employee salaries, bonuses, and hourly wages'
        },
        {
            'account_id': '5510',
            'account_name': 'Payroll Taxes',
            'category': 'expense',
            'subcategory': 'Payroll',
            'typical_min': 12000,
            'typical_max': 15000,
            'variance_threshold_pct': 10.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Employer FICA, unemployment, and state payroll taxes'
        },
        {
            'account_id': '5520',
            'account_name': 'Employee Benefits',
            'category': 'expense',
            'subcategory': 'Payroll',
            'typical_min': 15000,
            'typical_max': 20000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': False,
            'seasonal_pattern': 'Open enrollment adjustments in November',
            'description': 'Health insurance, 401k match, wellness programs'
        },
        {
            'account_id': '5530',
            'account_name': 'Contractor Payments',
            'category': 'expense',
            'subcategory': 'Payroll',
            'typical_min': 10000,
            'typical_max': 25000,
            'variance_threshold_pct': 30.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': '1099 contractor and freelancer payments'
        },
        
        # TRAVEL & BUSINESS DEVELOPMENT
        {
            'account_id': '6200',
            'account_name': 'Travel & Entertainment',
            'category': 'expense',
            'subcategory': 'Business Development',
            'typical_min': 2000,
            'typical_max': 5000,
            'variance_threshold_pct': 20.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Spikes in March (annual conference) and December (holiday events)',
            'description': 'Employee business travel and client entertainment expenses'
        },
        {
            'account_id': '6300',
            'account_name': 'Professional Development',
            'category': 'expense',
            'subcategory': 'Training',
            'typical_min': 1500,
            'typical_max': 4000,
            'variance_threshold_pct': 25.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Employee training, certifications, conferences'
        },
        
        # UTILITIES & FACILITIES
        {
            'account_id': '7100',
            'account_name': 'Utilities',
            'category': 'expense',
            'subcategory': 'Facilities',
            'typical_min': 1500,
            'typical_max': 3000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Higher in summer (AC) and winter (heating)',
            'description': 'Electric, water, internet, and facility utilities'
        },
        {
            'account_id': '7200',
            'account_name': 'Maintenance & Repairs',
            'category': 'expense',
            'subcategory': 'Facilities',
            'typical_min': 1000,
            'typical_max': 3000,
            'variance_threshold_pct': 40.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Building maintenance, HVAC repairs, equipment servicing'
        },
        {
            'account_id': '7300',
            'account_name': 'Janitorial Services',
            'category': 'expense',
            'subcategory': 'Facilities',
            'typical_min': 2000,
            'typical_max': 2500,
            'variance_threshold_pct': 15.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Cleaning services and supplies'
        },
        
        # MARKETING & SALES
        {
            'account_id': '8000',
            'account_name': 'Marketing & Advertising',
            'category': 'expense',
            'subcategory': 'Marketing',
            'typical_min': 5000,
            'typical_max': 15000,
            'variance_threshold_pct': 25.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Q4 spike for holiday campaigns',
            'description': 'Digital advertising, PR, events, and brand marketing'
        },
        {
            'account_id': '8100',
            'account_name': 'Lead Generation',
            'category': 'expense',
            'subcategory': 'Marketing',
            'typical_min': 3000,
            'typical_max': 8000,
            'variance_threshold_pct': 25.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Paid ads, SEO, content marketing for lead gen'
        },
        {
            'account_id': '8200',
            'account_name': 'Trade Shows & Events',
            'category': 'expense',
            'subcategory': 'Marketing',
            'typical_min': 0,
            'typical_max': 15000,
            'variance_threshold_pct': 50.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Q2 and Q4 major trade shows',
            'description': 'Industry conferences, booth rentals, sponsorships'
        },
        
        # PROFESSIONAL SERVICES
        {
            'account_id': '9000',
            'account_name': 'Legal Fees',
            'category': 'expense',
            'subcategory': 'Professional Services',
            'typical_min': 2000,
            'typical_max': 8000,
            'variance_threshold_pct': 40.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Corporate legal counsel, contract reviews, compliance'
        },
        {
            'account_id': '9100',
            'account_name': 'Accounting & Audit',
            'category': 'expense',
            'subcategory': 'Professional Services',
            'typical_min': 3000,
            'typical_max': 6000,
            'variance_threshold_pct': 20.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Year-end audit surge in January',
            'description': 'External accounting, bookkeeping, annual audit fees'
        },
        {
            'account_id': '9200',
            'account_name': 'IT Consulting',
            'category': 'expense',
            'subcategory': 'Professional Services',
            'typical_min': 4000,
            'typical_max': 10000,
            'variance_threshold_pct': 30.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'External IT support, security audits, infrastructure consulting'
        },
        
        # COST OF GOODS SOLD (if applicable)
        {
            'account_id': '3000',
            'account_name': 'Cost of Goods Sold',
            'category': 'expense',
            'subcategory': 'COGS',
            'typical_min': 15000,
            'typical_max': 30000,
            'variance_threshold_pct': 15.0,
            'is_seasonal': True,
            'seasonal_pattern': 'Scales with revenue - Q4 peak',
            'description': 'Direct costs of product delivery and fulfillment'
        },
        {
            'account_id': '3100',
            'account_name': 'Hosting & Infrastructure',
            'category': 'expense',
            'subcategory': 'COGS',
            'typical_min': 8000,
            'typical_max': 15000,
            'variance_threshold_pct': 20.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'AWS, Azure, CDN, database hosting costs'
        },
        
        # MISCELLANEOUS
        {
            'account_id': '9500',
            'account_name': 'Bank Fees & Interest',
            'category': 'expense',
            'subcategory': 'Financial',
            'typical_min': 500,
            'typical_max': 2000,
            'variance_threshold_pct': 30.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Credit card fees, wire transfers, interest charges'
        },
        {
            'account_id': '9600',
            'account_name': 'Depreciation',
            'category': 'expense',
            'subcategory': 'Non-Cash',
            'typical_min': 3000,
            'typical_max': 4000,
            'variance_threshold_pct': 10.0,
            'is_seasonal': False,
            'seasonal_pattern': None,
            'description': 'Equipment and asset depreciation - non-cash expense'
        },
        {
            'account_id': '9700',
            'account_name': 'Charitable Donations',
            'category': 'expense',
            'subcategory': 'Other',
            'typical_min': 0,
            'typical_max': 5000,
            'variance_threshold_pct': 50.0,
            'is_seasonal': True,
            'seasonal_pattern': 'End of year tax planning in December',
            'description': 'Corporate charitable giving and sponsorships'
        }
    ]
    
    return pd.DataFrame(accounts)


def generate_sample_pl_transactions(months: int = 12, include_anomalies: bool = True):
    """
    Generate comprehensive sample P&L transactions with realistic anomalies
    
    Args:
        months: Number of months of historical data
        include_anomalies: Include intentional anomalies for testing
    
    Returns:
        DataFrame with P&L transactions
    """
    transactions = []
    
    # Base patterns for each account (monthly baseline + random variance)
    account_patterns = {
        # Revenue
        '4000': {'base': 65000, 'variance': 5000},  # Product Sales
        '4100': {'base': 22000, 'variance': 8000},  # Consulting - variable
        '4200': {'base': 8000, 'variance': 2000},   # Licenses
        
        # Operating Expenses
        '5000': {'base': 12500, 'variance': 0},     # Rent - very stable
        '5100': {'base': 1800, 'variance': 800},    # Office Supplies
        '5200': {'base': 9500, 'variance': 500},    # Software/SaaS
        '5300': {'base': 3500, 'variance': 200},    # Insurance
        
        # Payroll
        '5500': {'base': 85000, 'variance': 2000},  # Salaries
        '5510': {'base': 13000, 'variance': 500},   # Payroll Taxes
        '5520': {'base': 17000, 'variance': 1000},  # Benefits
        '5530': {'base': 15000, 'variance': 8000},  # Contractors - highly variable
        
        # Business Development
        '6200': {'base': 3500, 'variance': 1200},   # Travel
        '6300': {'base': 2500, 'variance': 1500},   # Training
        
        # Facilities
        '7100': {'base': 2200, 'variance': 400},    # Utilities
        '7200': {'base': 1800, 'variance': 1200},   # Maintenance - unpredictable
        '7300': {'base': 2200, 'variance': 100},    # Janitorial - stable
        
        # Marketing
        '8000': {'base': 10000, 'variance': 4000},  # Marketing - variable
        '8100': {'base': 5000, 'variance': 2500},   # Lead Gen
        '8200': {'base': 5000, 'variance': 5000},   # Trade Shows - lumpy
        
        # Professional Services  
        '9000': {'base': 4000, 'variance': 3000},   # Legal - unpredictable
        '9100': {'base': 4000, 'variance': 1500},   # Accounting
        '9200': {'base': 6000, 'variance': 3000},   # IT Consulting
        
        # Other
        '3000': {'base': 20000, 'variance': 5000},  # COGS
        '3100': {'base': 11000, 'variance': 2000},  # Hosting
        '9500': {'base': 1200, 'variance': 600},    # Bank Fees
        '9600': {'base': 3500, 'variance': 0},      # Depreciation - fixed
        '9700': {'base': 1000, 'variance': 1000}    # Charitable - sporadic
    }
    
    # Generate transactions for each month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    current_date = start_date
    txn_id_counter = 1
    
    while current_date <= end_date:
        month_num = current_date.month
        
        for account_id, pattern in account_patterns.items():
            # Base amount with variance
            amount = pattern['base'] + random.uniform(-pattern['variance'], pattern['variance'])
            
            # Seasonal adjustments
            desc = f"Monthly {account_id} - {current_date.strftime('%B %Y')}"
            
            if account_id == '7100':  # Utilities - higher in summer/winter
                if month_num in [6, 7, 8, 12, 1, 2]:
                    amount *= 1.3
            
            if account_id == '8000':  # Marketing - Q4 spike (expected)
                if month_num in [10, 11, 12]:
                    amount *= 1.5
            
            if account_id == '3000':  # COGS scales with revenue
                if month_num == 12:
                    amount *= 1.3
            
            if account_id == '4000':  # Revenue - Q4 peak (expected)
                if month_num == 12:
                    amount *= 1.4
                elif month_num == 1:
                    amount *= 0.8
            
            # Add intentional ANOMALIES in the LAST MONTH for testing
            is_last_month = (end_date - current_date).days < 30
            
            if include_anomalies and is_last_month:
                # ANOMALY 1: Rent spike (HIGH - 44% variance)
                if account_id == '5000':
                    amount = 18000
                    desc = "🔴 ANOMALY: Monthly Rent + Security Deposit (Lease Renewal)"
                
                # ANOMALY 2: Travel spike (HIGH - 1200% variance)  
                elif account_id == '6200':
                    amount = 45000
                    desc = "🔴 ANOMALY: Annual Industry Conference - Las Vegas (10 attendees)"
                
                # ANOMALY 3: Legal fees spike (HIGH - 300% variance)
                elif account_id == '9000':
                    amount = 35000
                    desc = "🔴 ANOMALY: Emergency litigation costs - Contract dispute"
                
                # ANOMALY 4: IT Consulting spike (MEDIUM - 80% variance)
                elif account_id == '9200':
                    amount = 18000
                    desc = "🟡 ANOMALY: Cybersecurity incident response - External consultants"
                
                # ANOMALY 5: Salaries spike (MEDIUM - 20% variance)
                elif account_id == '5500':
                    amount = 108000
                    desc = "🟡 ANOMALY: Quarterly bonuses + 3 new hires"
                
                # ANOMALY 6: Maintenance spike (MEDIUM - 150% variance)
                elif account_id == '7200':
                    amount = 8500
                    desc = "🟡 ANOMALY: Emergency HVAC replacement"
                
                # ANOMALY 7: Contractor drop (HIGH - 70% decrease)
                elif account_id == '5530':
                    amount = 4500
                    desc = "🔴 ANOMALY: Major contractor projects completed early"
                
                # ANOMALY 8: Revenue drop (MEDIUM - 25% decrease)
                elif account_id == '4000':
                    amount = 48000
                    desc = "🟡 ANOMALY: Client churn - 3 major accounts lost"
                
                # ANOMALY 9: Insurance spike (HIGH - 200% variance)
                elif account_id == '5300':
                    amount = 12000
                    desc = "🔴 ANOMALY: Annual policy renewal + D&O rider added"
                
                # ANOMALY 10: Hosting spike (MEDIUM - 45% variance)
                elif account_id == '3100':
                    amount = 18000
                    desc = "🟡 ANOMALY: Traffic surge + infrastructure scaling"
            
            # Create transaction
            transactions.append({
                'transaction_id': f"TXN{txn_id_counter:05d}",
                'gl_account_id': account_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'description': desc,
                'department': 'Operations',
                'vendor': f"Vendor_{account_id}"
            })
            
            txn_id_counter += 1
        
        # Move to next month
        current_date += timedelta(days=30)
    
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def create_sample_data_files(output_dir: str = "./data"):
    """Create sample data files for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔨 Generating sample data files...")
    
    # Generate GL accounts
    gl_df = generate_sample_gl_accounts()
    gl_path = output_path / "gl_accounts.csv"
    gl_df.to_csv(gl_path, index=False)
    print(f"✅ Created: {gl_path} ({len(gl_df)} accounts)")
    
    # Generate 12 months of P&L data
    pl_df = generate_sample_pl_transactions(months=12, include_anomalies=True)
    
    # Split into monthly files (realistic scenario)
    pl_reports_dir = output_path / "pl_reports"
    pl_reports_dir.mkdir(exist_ok=True)
    
    for month_str, group in pl_df.groupby(pd.to_datetime(pl_df['date']).dt.to_period('M')):
        month_file = pl_reports_dir / f"pl_{month_str}.csv"
        group.to_csv(month_file, index=False)
        print(f"✅ Created: {month_file} ({len(group)} transactions)")
    
    # Also create a combined file for initial 12-month load
    combined_path = output_path / "pl_12_months_historical.csv"
    pl_df.to_csv(combined_path, index=False)
    print(f"✅ Created: {combined_path} (all {len(pl_df)} transactions)")
    
    # Create sample GL documentation
    docs_dir = output_path / "gl_documentation"
    docs_dir.mkdir(exist_ok=True)
    
    for _, account in gl_df.iterrows():
        doc_content = f"""# {account['account_name']} (GL {account['account_id']})

## Account Information
- **Account ID**: {account['account_id']}
- **Category**: {account['category']}
- **Subcategory**: {account.get('subcategory', 'N/A')}
- **Typical Monthly Range**: ${account['typical_min']:,.2f} - ${account['typical_max']:,.2f}

## Description
{account['description']}

## Variance Patterns
- **Threshold**: {account['variance_threshold_pct']}% before flagging as anomaly
- **Seasonal**: {'Yes' if account['is_seasonal'] else 'No'}
{f"- **Pattern**: {account['seasonal_pattern']}" if account['seasonal_pattern'] else ''}

## Typical Causes for Variance
- Normal business fluctuations within expected range
- Seasonal patterns as noted above
- One-time expenses or revenue events
- Changes in vendor contracts or pricing
- Business expansion or contraction

## Historical Notes
(This section updated by agent as anomalies are investigated)

## Approval Workflows
- Variances >15%: Require controller review
- Variances >30%: Require CFO approval
- Unusual patterns: Flag for audit committee

## Related Accounts
- See departmental budgets for context
- Cross-reference with cash flow statements
"""
        
        doc_path = docs_dir / f"{account['account_id']}_{account['account_name'].lower().replace(' ', '_').replace('&', 'and')}.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        print(f"✅ Created: {doc_path}")
    
    print("\n" + "=" * 80)
    print("✅ Sample data generation complete!")
    print("=" * 80)
    print(f"\nGenerated files in {output_path}/:")
    print(f"  - gl_accounts.csv (GL master)")
    print(f"  - pl_12_months_historical.csv (all transactions)")
    print(f"  - pl_reports/*.csv (monthly files)")
    print(f"  - gl_documentation/*.md (account docs)")
    print("\nNext steps:")
    print("1. Run: python main.py init")
    print("2. Run: python main.py analyze data/pl_12_months_historical.csv")


if __name__ == "__main__":
    create_sample_data_files()
    
    print("\n" + "=" * 80)
    print("📊 EXPECTED ANOMALIES IN FINAL MONTH")
    print("=" * 80)
    print("""
The generated data includes 10 intentional anomalies in the last month:

HIGH SEVERITY (>30% variance):
1. GL 5000 (Rent): +44% - Security deposit added
2. GL 6200 (Travel): +1200% - Annual conference (10 attendees)
3. GL 9000 (Legal): +775% - Emergency litigation
4. GL 5530 (Contractors): -70% - Projects completed early
5. GL 5300 (Insurance): +243% - Annual renewal + new coverage

MEDIUM SEVERITY (15-30% variance):
6. GL 9200 (IT Consulting): +200% - Cybersecurity incident
7. GL 5500 (Salaries): +27% - Bonuses + new hires  
8. GL 7200 (Maintenance): +372% - Emergency HVAC replacement
9. GL 4000 (Revenue): -26% - Client churn
10. GL 3100 (Hosting): +64% - Infrastructure scaling

Dataset Statistics:
- Total GL Accounts: 25
- Transactions per Month: 25
- Total Transactions (12 months): ~300
- Expected Anomaly Detection: 10 anomalies in final month
""")

