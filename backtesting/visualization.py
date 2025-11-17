"""
Visualization
Create charts for backtest results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os


class Visualizer:
    """Create visualizations for portfolio performance"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-darkgrid')
        
        sns.set_palette("husl")
        self.figsize = (12, 6)
    
    def plot_cumulative_returns(self, backtest_results: Dict, 
                               benchmark_results: Dict = None,
                               output_path: str = None):
        """
        Plot cumulative returns over time
        
        Args:
            backtest_results: Strategy backtest results
            benchmark_results: Optional benchmark results
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Strategy cumulative returns
        history = backtest_results['history']
        quarters = [h['quarter'] for h in history]
        values = [backtest_results['summary']['initial_capital']] + [h['end_value'] for h in history]
        
        ax.plot(range(len(values)), values, marker='o', linewidth=2, label='LLM-BL Strategy')
        
        # Benchmark
        if benchmark_results:
            bench_history = benchmark_results['history']
            bench_values = [benchmark_results['summary']['initial_capital']] + \
                          [h['end_value'] for h in bench_history]
            ax.plot(range(len(bench_values)), bench_values, marker='s', linewidth=2, 
                   label='Equal Weight Benchmark', linestyle='--')
        
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Cumulative Portfolio Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Set x-axis labels
        x_labels = ['Start'] + quarters
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved cumulative returns chart to {output_path}")
        
        plt.close()
    
    def plot_quarterly_returns(self, backtest_results: Dict,
                              benchmark_results: Dict = None,
                              output_path: str = None):
        """
        Plot quarterly returns comparison
        
        Args:
            backtest_results: Strategy results
            benchmark_results: Benchmark results
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        quarters = [h['quarter'] for h in backtest_results['history']]
        strategy_returns = [h['return'] * 100 for h in backtest_results['history']]
        
        x = np.arange(len(quarters))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strategy_returns, width, label='LLM-BL Strategy')
        
        if benchmark_results:
            bench_returns = [h['return'] * 100 for h in benchmark_results['history']]
            bars2 = ax.bar(x + width/2, bench_returns, width, label='Equal Weight Benchmark')
        
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Quarterly Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved quarterly returns chart to {output_path}")
        
        plt.close()
    
    def plot_allocation_heatmap(self, backtest_results: Dict,
                               output_path: str = None):
        """
        Plot portfolio allocation heatmap over time
        
        Args:
            backtest_results: Backtest results
            output_path: Path to save figure
        """
        history = backtest_results['history']
        quarters = [h['quarter'] for h in history]
        tickers = sorted(list(history[0]['weights'].keys()))
        
        # Build allocation matrix
        allocation_matrix = []
        for h in history:
            weights = h['weights']
            allocation_matrix.append([weights[ticker] * 100 for ticker in tickers])
        
        allocation_matrix = np.array(allocation_matrix).T
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], 6))
        
        sns.heatmap(allocation_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                   xticklabels=quarters, yticklabels=tickers,
                   cbar_kws={'label': 'Allocation (%)'}, ax=ax)
        
        ax.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Ticker', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved allocation heatmap to {output_path}")
        
        plt.close()
    
    def plot_drawdown(self, backtest_results: Dict,
                     output_path: str = None):
        """
        Plot drawdown over time
        
        Args:
            backtest_results: Backtest results
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        history = backtest_results['history']
        initial_capital = backtest_results['summary']['initial_capital']
        
        values = [initial_capital] + [h['end_value'] for h in history]
        quarters = ['Start'] + [h['quarter'] for h in history]
        
        # Calculate drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100
        
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(range(len(drawdown)), drawdown, color='darkred', linewidth=2)
        
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels(quarters, rotation=45, ha='right')
        
        # Highlight max drawdown
        max_dd_idx = np.argmin(drawdown)
        ax.plot(max_dd_idx, drawdown[max_dd_idx], 'r*', markersize=15,
               label=f'Max Drawdown: {drawdown[max_dd_idx]:.2f}%')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved drawdown chart to {output_path}")
        
        plt.close()
    
    def plot_risk_return_scatter(self, backtest_results: Dict,
                                benchmark_results: Dict = None,
                                output_path: str = None):
        """
        Plot risk-return scatter
        
        Args:
            backtest_results: Strategy results
            benchmark_results: Benchmark results
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        strategy_return = backtest_results['summary']['annualized_return'] * 100
        strategy_vol = backtest_results['summary']['volatility'] * 100
        
        ax.scatter(strategy_vol, strategy_return, s=200, marker='o', 
                  label='LLM-BL Strategy', zorder=3)
        
        if benchmark_results:
            bench_return = benchmark_results['summary']['annualized_return'] * 100
            bench_vol = benchmark_results['summary']['volatility'] * 100
            ax.scatter(bench_vol, bench_return, s=200, marker='s',
                      label='Equal Weight Benchmark', zorder=3)
        
        ax.set_xlabel('Volatility (% annual)', fontsize=12)
        ax.set_ylabel('Return (% annual)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved risk-return scatter to {output_path}")
        
        plt.close()
    
    def create_all_charts(self, backtest_results: Dict,
                         benchmark_results: Dict = None,
                         output_dir: str = 'output/charts'):
        """
        Create all visualization charts
        
        Args:
            backtest_results: Strategy results
            benchmark_results: Benchmark results
            output_dir: Directory to save charts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        self.plot_cumulative_returns(
            backtest_results, benchmark_results,
            os.path.join(output_dir, 'cumulative_returns.png')
        )
        
        self.plot_quarterly_returns(
            backtest_results, benchmark_results,
            os.path.join(output_dir, 'quarterly_returns.png')
        )
        
        self.plot_allocation_heatmap(
            backtest_results,
            os.path.join(output_dir, 'allocation_heatmap.png')
        )
        
        self.plot_drawdown(
            backtest_results,
            os.path.join(output_dir, 'drawdown.png')
        )
        
        self.plot_risk_return_scatter(
            backtest_results, benchmark_results,
            os.path.join(output_dir, 'risk_return.png')
        )
        
        print(f"\n✓ All charts saved to {output_dir}/")


def main():
    """Test visualizer"""
    pass


if __name__ == "__main__":
    main()

