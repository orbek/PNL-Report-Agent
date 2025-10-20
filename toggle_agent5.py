#!/usr/bin/env python3
"""
Agent 5 Toggle Utility

This script makes it easy to enable/disable Agent 5 (Report Formatting) in the workflow.

Usage:
    python toggle_agent5.py enable   # Enable Agent 5
    python toggle_agent5.py disable  # Disable Agent 5
    python toggle_agent5.py status   # Check current status
"""

import sys
import os
import re
from pathlib import Path

def get_workflow_file():
    """Get the path to workflow.py"""
    return Path(__file__).parent / "workflow.py"

def read_workflow_file():
    """Read the workflow.py file"""
    workflow_file = get_workflow_file()
    if not workflow_file.exists():
        print(f"❌ Error: {workflow_file} not found!")
        return None
    
    with open(workflow_file, 'r') as f:
        return f.read()

def write_workflow_file(content):
    """Write the workflow.py file"""
    workflow_file = get_workflow_file()
    with open(workflow_file, 'w') as f:
        f.write(content)

def check_status():
    """Check if Agent 5 is enabled or disabled"""
    content = read_workflow_file()
    if content is None:
        return None
    
    # Check if Agent 5 node is commented out (look for uncommented version)
    node_pattern = r'^\s*workflow\.add_node\("format"'
    node_match = re.search(node_pattern, content, re.MULTILINE)
    
    # Check if Agent 5 edges are commented out (look for uncommented version)
    edge_pattern = r'^\s*workflow\.add_edge\("report", "format"\)'
    edge_match = re.search(edge_pattern, content, re.MULTILINE)
    
    if node_match and edge_match:
        return "enabled"
    else:
        return "disabled"

def enable_agent5():
    """Enable Agent 5 by uncommenting the relevant lines"""
    content = read_workflow_file()
    if content is None:
        return False
    
    # Uncomment Agent 5 node (handle various comment patterns)
    content = re.sub(
        r'^\s*#\s*workflow\.add_node\("format", self\.agents\.format_report\)',
        r'        workflow.add_node("format", self.agents.format_report)',
        content,
        flags=re.MULTILINE
    )
    
    # Uncomment Agent 5 edges
    content = re.sub(
        r'^\s*#\s*workflow\.add_edge\("report", "format"\)',
        r'        workflow.add_edge("report", "format")',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^\s*#\s*workflow\.add_edge\("format", END\)',
        r'        workflow.add_edge("format", END)',
        content,
        flags=re.MULTILINE
    )
    
    # Comment out the direct edge from report to END
    content = re.sub(
        r'^\s*workflow\.add_edge\("report", END\)\s*# Skip formatting step',
        r'        # workflow.add_edge("report", END)  # Skip formatting step',
        content,
        flags=re.MULTILINE
    )
    
    write_workflow_file(content)
    return True

def disable_agent5():
    """Disable Agent 5 by commenting out the relevant lines"""
    content = read_workflow_file()
    if content is None:
        return False
    
    # Comment out Agent 5 node
    content = re.sub(
        r'^\s*workflow\.add_node\("format", self\.agents\.format_report\)',
        r'        # workflow.add_node("format", self.agents.format_report)  # Temporarily disabled',
        content,
        flags=re.MULTILINE
    )
    
    # Comment out Agent 5 edges
    content = re.sub(
        r'^\s*workflow\.add_edge\("report", "format"\)',
        r'        # workflow.add_edge("report", "format")  # Temporarily disabled',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^\s*workflow\.add_edge\("format", END\)',
        r'        # workflow.add_edge("format", END)  # Temporarily disabled',
        content,
        flags=re.MULTILINE
    )
    
    # Uncomment the direct edge from report to END
    content = re.sub(
        r'^\s*#\s*workflow\.add_edge\("report", END\)\s*# Skip formatting step',
        r'        workflow.add_edge("report", END)  # Skip formatting step',
        content,
        flags=re.MULTILINE
    )
    
    write_workflow_file(content)
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python toggle_agent5.py [enable|disable|status]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        status = check_status()
        if status is None:
            sys.exit(1)
        
        if status == "enabled":
            print("✅ Agent 5 (Report Formatting) is ENABLED")
            print("   - Report formatting will be processed by GPT-4o-mini")
            print("   - Additional cost: ~$0.0001 - $0.001 per analysis")
        else:
            print("⚠️  Agent 5 (Report Formatting) is DISABLED")
            print("   - Currency formatting handled directly by Agent 4")
            print("   - Faster execution, lower cost")
    
    elif command == "enable":
        current_status = check_status()
        if current_status == "enabled":
            print("✅ Agent 5 is already enabled!")
            return
        
        if enable_agent5():
            print("✅ Agent 5 (Report Formatting) has been ENABLED")
            print("   - Restart your analysis to use Agent 5")
            print("   - Additional formatting will be applied to reports")
        else:
            print("❌ Failed to enable Agent 5")
            sys.exit(1)
    
    elif command == "disable":
        current_status = check_status()
        if current_status == "disabled":
            print("⚠️  Agent 5 is already disabled!")
            return
        
        if disable_agent5():
            print("⚠️  Agent 5 (Report Formatting) has been DISABLED")
            print("   - Currency formatting handled by Agent 4")
            print("   - Faster execution, lower cost")
        else:
            print("❌ Failed to disable Agent 5")
            sys.exit(1)
    
    else:
        print("❌ Invalid command. Use: enable, disable, or status")
        sys.exit(1)

if __name__ == "__main__":
    main()
