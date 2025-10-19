"""
API Cost Tracking for Financial P&L Anomaly Detection Agent
Tracks token usage and calculates costs based on OpenAI pricing
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to format small decimals properly"""
    def encode(self, obj):
        if isinstance(obj, float):
            # Format small numbers as proper decimals instead of scientific notation
            if 0 < abs(obj) < 0.001:
                return f"{obj:.8f}"
        return super().encode(obj)

@dataclass
class APICall:
    """Individual API call cost tracking"""
    timestamp: str
    model: str
    agent: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    call_duration: float
    query_preview: str = ""  # First 100 chars of query for debugging

@dataclass
class RunCostSummary:
    """Complete run cost summary"""
    run_id: str
    start_time: str
    end_time: str
    total_duration: float
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    cost_breakdown: Dict[str, float]  # by agent
    calls: List[APICall]

class CostTracker:
    """Track and calculate API costs for each run"""
    
    # OpenAI Pricing (per 1M tokens) - Updated December 2024
    PRICING = {
        # GPT-4 Models
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        
        # GPT-5 Models (when available)
        "gpt-5": {"input": 1.25, "output": 10.00},
        
        # Embedding Models
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0}
    }
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.start_time = datetime.now()
        self.calls: List[APICall] = []
        self.agent_costs: Dict[str, float] = {}
    
    def track_call(self, 
                   model: str, 
                   agent: str, 
                   input_tokens: int, 
                   output_tokens: int,
                   call_duration: float,
                   query_preview: str = "") -> APICall:
        """Track a single API call and calculate cost"""
        
        # Get pricing for model (fallback to gpt-4 if not found)
        pricing = self.PRICING.get(model, self.PRICING["gpt-4"])
        
        # Calculate costs (convert to per-token pricing)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Create call record
        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            call_duration=call_duration,
            query_preview=query_preview[:100] if query_preview else ""
        )
        
        # Track call
        self.calls.append(call)
        
        # Update agent costs
        if agent not in self.agent_costs:
            self.agent_costs[agent] = 0.0
        self.agent_costs[agent] += total_cost
        
        return call
    
    def track_embedding_call(self, 
                           model: str, 
                           agent: str, 
                           input_tokens: int,
                           call_duration: float,
                           query_preview: str = "") -> APICall:
        """Track embedding API call (no output tokens)"""
        return self.track_call(model, agent, input_tokens, 0, call_duration, query_preview)
    
    def get_summary(self) -> RunCostSummary:
        """Generate complete cost summary for the run"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        total_input_tokens = sum(call.input_tokens for call in self.calls)
        total_output_tokens = sum(call.output_tokens for call in self.calls)
        total_cost = sum(call.total_cost for call in self.calls)
        
        return RunCostSummary(
            run_id=self.run_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=total_duration,
            total_calls=len(self.calls),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            cost_breakdown=self.agent_costs.copy(),
            calls=self.calls.copy()
        )
    
    def save_report(self, filepath: str):
        """Save cost report to JSON file"""
        summary = self.get_summary()
        
        # Convert to dict and format costs as proper decimals
        report_data = asdict(summary)
        
        # Format all cost fields to 8 decimal places as strings
        def format_costs(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if (('cost' in key.lower() or key == 'total_cost') and isinstance(value, float)) or (key == 'cost_breakdown' and isinstance(value, dict)):
                        if key == 'cost_breakdown':
                            # Format cost_breakdown values
                            for agent, cost in value.items():
                                if isinstance(cost, float):
                                    value[agent] = f"{cost:.8f}"
                        else:
                            # Convert to string with 8 decimal places to avoid scientific notation
                            obj[key] = f"{value:.8f}"
                    elif isinstance(value, (dict, list)):
                        format_costs(value)
            elif isinstance(obj, list):
                for item in obj:
                    format_costs(item)
        
        format_costs(report_data)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print cost summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("💰 API COST SUMMARY")
        print("="*70)
        print(f"Run ID: {summary.run_id}")
        print(f"Duration: {summary.total_duration:.1f}s")
        print(f"Total Calls: {summary.total_calls}")
        print(f"Total Tokens: {summary.total_input_tokens:,} input + {summary.total_output_tokens:,} output")
        print(f"Total Cost: ${summary.total_cost:.8f}")
        print("\nCost by Agent:")
        for agent, cost in summary.cost_breakdown.items():
            percentage = (cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            print(f"  {agent:15}: ${cost:.8f} ({percentage:.1f}%)")
        
        # Show detailed call breakdown if there are calls
        if summary.calls:
            print(f"\n📊 DETAILED CALL BREAKDOWN:")
            print("-" * 80)
            print(f"{'#':<3} {'Agent':<12} {'Model':<12} {'Input':<8} {'Output':<8} {'Input Cost':<12} {'Output Cost':<12} {'Total Cost':<12} {'Duration':<8} {'Query Preview'}")
            print("-" * 80)
            
            for i, call in enumerate(summary.calls, 1):
                input_cost_str = f"${call.input_cost:.8f}"
                output_cost_str = f"${call.output_cost:.8f}"
                total_cost_str = f"${call.total_cost:.8f}"
                print(f"{i:<3} {call.agent:<12} {call.model:<12} {call.input_tokens:<8} {call.output_tokens:<8} {input_cost_str:<12} {output_cost_str:<12} {total_cost_str:<12} {call.call_duration:<7.2f}s {call.query_preview[:25]}...")
        
        print("="*80)
    
    def get_cost_per_anomaly(self, anomaly_count: int) -> float:
        """Calculate average cost per anomaly analyzed"""
        if anomaly_count == 0:
            return 0.0
        summary = self.get_summary()
        return summary.total_cost / anomaly_count
