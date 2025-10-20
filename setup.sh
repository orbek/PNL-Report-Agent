#!/bin/bash
# Quick setup script for Financial P&L Anomaly Detection Agent

set -e  # Exit on error

echo "============================================================================"
echo "  Financial P&L Anomaly Detection Agent - Setup"
echo "============================================================================"

# Check Python version
echo ""
echo "🐍 Checking Python version..."

# Try to find Python 3.10
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
    echo "   ✅ Found: Python $PYTHON_VERSION"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "   Found: Python $PYTHON_VERSION"
    
    # Check if it's 3.10
    if ! python3 -c 'import sys; exit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then
        echo "   ⚠️  Warning: Python 3.10 is recommended. You have $PYTHON_VERSION"
        echo "   Install Python 3.10: brew install python@3.10 (macOS) or apt install python3.10 (Linux)"
    fi
else
    echo "   ❌ Python 3 not found. Please install Python 3.10"
    exit 1
fi

# Create virtual environment with Python 3.10
echo ""
echo "📦 Creating virtual environment with $PYTHON_CMD..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "📥 Installing dependencies..."
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "   ✅ Dependencies installed"

# Check for .env file
echo ""
echo "🔑 Checking environment configuration..."
if [ ! -f "../.env" ]; then
    echo "   ⚠️  .env file not found"
    echo "   Creating from template..."
    cp ../.env.example ../.env
    echo "   ✅ Created ../.env - EDIT THIS FILE WITH YOUR API KEYS"
    echo ""
    echo "   📝 Required: Add your OPENAI_API_KEY to ../.env"
    echo "   Edit with: nano ../.env"
else
    echo "   ✅ .env file exists"
    
    # Check if API key is set
    if grep -q "sk-proj-...your-key-here..." ../.env; then
        echo "   ⚠️  WARNING: Update OPENAI_API_KEY in ../.env with your actual key"
    else
        echo "   ✅ OPENAI_API_KEY appears to be configured"
    fi
fi

# Generate sample data
echo ""
echo "🔨 Generating sample data..."
python main.py --generate-sample

# Initialize database and vector store
echo ""
echo "📊 Initializing database and vector store..."
python main.py init

# Run test analysis
echo ""
echo "🧪 Running test analysis with sample data..."
python main.py analyze data/pl_12_months_historical.csv

echo ""
echo "============================================================================"
echo "✅ Setup Complete!"
echo "============================================================================"
echo ""
echo "📄 Check your anomaly report in: reports/"
echo ""
echo "Next steps:"
echo "1. Review the generated report"
echo "2. Replace sample data with your real P&L files"
echo "3. Run: python main.py analyze <your-pl-file.csv>"
echo ""
echo "For help: python main.py --help"
echo ""
echo "Deactivate virtual environment: deactivate"
echo "============================================================================"

