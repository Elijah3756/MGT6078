"""
Report Generator
Generate comprehensive reports with results and analysis
"""

import json
import os
from datetime import datetime
from typing import Dict


class ReportGenerator:
    """Generate portfolio analysis reports"""
    
    def __init__(self, output_dir='output/reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_text_report(self, backtest_results: Dict,
                            benchmark_results: Dict,
                            all_metrics: Dict,
                            portfolio_results: list) -> str:
        """
        Generate comprehensive text report
        
        Args:
            backtest_results: Strategy backtest results
            benchmark_results: Benchmark backtest results
            all_metrics: Performance metrics
            portfolio_results: List of quarterly portfolio results
            
        Returns:
            Report text
        """
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append("LLM-POWERED BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
        lines.append("BACKTEST REPORT")
        lines.append("="*80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Summary
        lines.append("\n" + "="*80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("="*80)
        
        strategy_summary = backtest_results['summary']
        benchmark_summary = benchmark_results['summary']
        
        lines.append(f"\nStrategy Performance:")
        lines.append(f"  Total Return:         {strategy_summary['total_return']:>10.2%}")
        lines.append(f"  Annualized Return:    {strategy_summary['annualized_return']:>10.2%}")
        lines.append(f"  Volatility:           {strategy_summary['volatility']:>10.2%}")
        lines.append(f"  Sharpe Ratio:         {strategy_summary['sharpe_ratio']:>10.4f}")
        lines.append(f"  Maximum Drawdown:     {strategy_summary['max_drawdown']:>10.2%}")
        
        lines.append(f"\nBenchmark (Equal Weight) Performance:")
        lines.append(f"  Total Return:         {benchmark_summary['total_return']:>10.2%}")
        lines.append(f"  Annualized Return:    {benchmark_summary['annualized_return']:>10.2%}")
        lines.append(f"  Volatility:           {benchmark_summary['volatility']:>10.2%}")
        lines.append(f"  Sharpe Ratio:         {benchmark_summary['sharpe_ratio']:>10.4f}")
        lines.append(f"  Maximum Drawdown:     {benchmark_summary['max_drawdown']:>10.2%}")
        
        lines.append(f"\nOutperformance:")
        outperf = strategy_summary['annualized_return'] - benchmark_summary['annualized_return']
        lines.append(f"  Annualized:           {outperf:>10.2%}")
        
        # Detailed Metrics
        lines.append("\n" + "="*80)
        lines.append("DETAILED PERFORMANCE METRICS")
        lines.append("="*80)
        
        lines.append(f"\nRisk-Adjusted Returns:")
        lines.append(f"  Sharpe Ratio:         {all_metrics['sharpe_ratio']:>10.4f}")
        lines.append(f"  Sortino Ratio:        {all_metrics['sortino_ratio']:>10.4f}")
        lines.append(f"  Calmar Ratio:         {all_metrics['calmar_ratio']:>10.4f}")
        if 'information_ratio' in all_metrics:
            lines.append(f"  Information Ratio:    {all_metrics['information_ratio']:>10.4f}")
        
        lines.append(f"\nRisk Metrics:")
        lines.append(f"  Volatility (annual):  {all_metrics['volatility']:>10.2%}")
        lines.append(f"  VaR (95%):           {all_metrics['var_95']:>10.2%}")
        lines.append(f"  CVaR (95%):          {all_metrics['cvar_95']:>10.2%}")
        lines.append(f"  Maximum Drawdown:     {all_metrics['max_drawdown']:>10.2%}")
        
        lines.append(f"\nTurnover:")
        lines.append(f"  Average per Quarter:  {all_metrics['avg_turnover']:>10.2%}")
        
        # Quarterly Performance
        lines.append("\n" + "="*80)
        lines.append("QUARTERLY PERFORMANCE")
        lines.append("="*80)
        
        lines.append(f"\n{'Quarter':<12} {'Strategy':>10} {'Benchmark':>10} {'Diff':>10}")
        lines.append("-" * 45)
        
        for i, h in enumerate(backtest_results['history']):
            quarter = h['quarter']
            strat_ret = h['return']
            bench_ret = benchmark_results['history'][i]['return']
            diff = strat_ret - bench_ret
            
            lines.append(f"{quarter:<12} {strat_ret:>9.2%} {bench_ret:>9.2%} {diff:>9.2%}")
        
        # Portfolio Allocations
        lines.append("\n" + "="*80)
        lines.append("PORTFOLIO ALLOCATIONS")
        lines.append("="*80)
        
        for result in portfolio_results:
            lines.append(f"\n{result['quarter']}:")
            lines.append(f"  Sharpe Ratio: {result['sharpe']:.4f}")
            lines.append(f"  Allocations:")
            for ticker, weight in sorted(result['ticker_weights'].items()):
                lines.append(f"    {ticker}: {weight:>6.2%}")
        
        # Methodology
        lines.append("\n" + "="*80)
        lines.append("METHODOLOGY")
        lines.append("="*80)
        
        lines.append("""
This portfolio optimization system uses a novel approach combining:

1. Multi-Source Data Analysis
   - Political news from GDELT database
   - Federal Reserve FOMC meeting minutes
   - Quarterly investment research reports
   - Bloomberg financial headlines
   - Historical ETF price data

2. LLM-Powered Views Generation
   - Large Language Model analyzes all data sources
   - Generates expected returns and confidence levels
   - Produces structured investment views per sector

3. Black-Litterman Portfolio Optimization
   - Combines market equilibrium (prior) with LLM views
   - Uncertainty weighting based on confidence levels
   - Long-only optimization with realistic constraints

4. Quarterly Rebalancing
   - Portfolio rebalanced each quarter based on new data
   - Transaction costs not modeled (conservative estimate)
   - Benchmark: Equal-weight portfolio of same 6 ETFs

Sector ETFs:
  XLK - Technology
  XLY - Consumer Discretionary
  ITA - Aerospace & Defense
  XLE - Energy
  XLV - Healthcare
  XLF - Financials
""")
        
        # Conclusion
        lines.append("\n" + "="*80)
        lines.append("CONCLUSION")
        lines.append("="*80)
        
        if outperf > 0:
            lines.append(f"""
The LLM-powered Black-Litterman strategy outperformed the equal-weight benchmark
by {outperf:.2%} annually over the test period. This demonstrates the potential
value of incorporating LLM-analyzed qualitative data into quantitative portfolio
optimization frameworks.

Key success factors:
- Systematic integration of multiple data sources
- Structured LLM output for consistent views
- Risk-aware optimization through Black-Litterman framework
- Quarterly rebalancing discipline
""")
        else:
            lines.append(f"""
The LLM-powered Black-Litterman strategy underperformed the equal-weight benchmark
by {abs(outperf):.2%} annually over the test period. This highlights challenges in:
- LLM interpretation of complex financial data
- Parameter sensitivity in the Black-Litterman framework
- Potential overfitting to recent market patterns

Areas for improvement:
- Enhanced LLM prompting strategies
- Hyperparameter optimization
- Longer backtesting periods
- Alternative view formulations
""")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def save_json_results(self, backtest_results: Dict,
                         benchmark_results: Dict,
                         all_metrics: Dict,
                         portfolio_results: list,
                         filename: str = 'results.json'):
        """
        Save all results as JSON
        
        Args:
            backtest_results: Strategy results
            benchmark_results: Benchmark results
            all_metrics: Performance metrics
            portfolio_results: Portfolio results
            filename: Output filename
        """
        import numpy as np
        
        def convert_to_json_serializable(obj):
            """Convert numpy types to Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        output = {
            'generated_at': datetime.now().isoformat(),
            'strategy': convert_to_json_serializable(backtest_results),
            'benchmark': convert_to_json_serializable(benchmark_results),
            'metrics': convert_to_json_serializable(all_metrics),
            'portfolios': convert_to_json_serializable(portfolio_results)
        }
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved JSON results to {output_path}")
    
    def generate_full_report(self, backtest_results: Dict,
                            benchmark_results: Dict,
                            all_metrics: Dict,
                            portfolio_results: list):
        """
        Generate complete report suite
        
        Args:
            backtest_results: Strategy results
            benchmark_results: Benchmark results
            all_metrics: Performance metrics
            portfolio_results: Portfolio results
        """
        print(f"\n{'='*80}")
        print("GENERATING REPORTS")
        print(f"{'='*80}\n")
        
        # Text report
        report_text = self.generate_text_report(
            backtest_results, benchmark_results, all_metrics, portfolio_results
        )
        
        text_path = os.path.join(self.output_dir, 'full_report.txt')
        with open(text_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Saved text report to {text_path}")
        
        # JSON results
        self.save_json_results(
            backtest_results, benchmark_results, all_metrics, portfolio_results
        )
        
        print(f"\n✓ All reports generated in {self.output_dir}/")


def main():
    """Test report generator"""
    pass


if __name__ == "__main__":
    main()

