#!/usr/bin/env python3
"""
Main Pipeline
LLM-Powered Black-Litterman Portfolio Optimization System
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import DataLoader
from llm.analyzer import LLMAnalyzer
from optimization.portfolio_optimizer import PortfolioOptimizer
from backtesting.backtester import Backtester
from backtesting.metrics import PerformanceMetrics
from backtesting.visualization import Visualizer
from reports.report_generator import ReportGenerator
from config_loader import get_config

import argparse
import json


def main():
    """Main execution pipeline"""
    
    # Load configuration
    config = get_config()
    
    # Parse arguments (CLI args override config)
    parser = argparse.ArgumentParser(description='LLM-Powered Black-Litterman Portfolio Optimization')
    parser.add_argument('--llm-type', type=str, default=None,
                       choices=['finance-llm', 'simulated'],
                       help='LLM model type (overrides config.yaml)')
    parser.add_argument('--quarters', type=str, nargs='+', default=None,
                       help='Quarters to analyze (overrides config.yaml)')
    parser.add_argument('--skip-charts', action='store_true',
                       help='Skip chart generation')
    parser.add_argument('--allow-shorts', action='store_true', default=None,
                       help='Allow short positions (overrides config.yaml)')
    parser.add_argument('--long-only', action='store_true',
                       help='Restrict to long-only positions (no shorting)')
    
    args = parser.parse_args()
    
    # Use config values with CLI overrides
    llm_type = args.llm_type or config['llm']['model_type']
    quarters = args.quarters or config['data']['quarters']
    allow_shorts = config['black_litterman']['allow_shorts'] if args.allow_shorts is None else args.allow_shorts
    
    # Handle long-only flag (overrides allow-shorts)
    if args.long_only:
        allow_shorts = False
    
    initial_capital = config['portfolio']['initial_capital']
    
    print("\n" + "="*80)
    print("LLM-POWERED BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  LLM Type: {llm_type}")
    print(f"  Quarters: {', '.join(quarters)}")
    print(f"  Allow Shorts: {allow_shorts}")
    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"="*80)
    
    # Initialize components
    print("\n" + "="*80)
    print("INITIALIZING COMPONENTS")
    print("="*80)
    
    loader = DataLoader()
    
    # Initialize analyzer with config values (temperature, top_p)
    llm_config = config.get('llm', {})
    analyzer = LLMAnalyzer(
        model_type=llm_type,
        temperature=llm_config.get('temperature', 0.7),
        top_p=llm_config.get('top_p', 0.9)
    )
    
    # Initialize optimizer with config values
    optimizer = PortfolioOptimizer(
        allow_shorts=allow_shorts,
        min_weight=config['black_litterman']['min_weight'],
        max_weight=config['black_litterman']['max_weight']
    )
    optimizer.set_hyperparameters(
        risk_aversion=config['black_litterman']['risk_aversion'],
        tau_for_covariance=config['black_litterman']['tau_for_covariance'],
        tau_omega=config['black_litterman']['tau_omega'],
        relative_confidence=config['black_litterman']['relative_confidence']
    )
    
    backtester = Backtester(initial_capital=initial_capital)
    visualizer = Visualizer()
    report_gen = ReportGenerator()
    
    print("✓ All components initialized")
    
    # Process each quarter
    print("\n" + "="*80)
    print("PROCESSING QUARTERS")
    print("="*80)
    
    portfolio_results = []
    
    for quarter in quarters:
        print(f"\n{'#'*80}")
        print(f"# QUARTER: {quarter}")
        print(f"{'#'*80}")
        
        try:
            # 1. Load data
            print(f"\nStep 1/4: Loading data for {quarter}...")
            quarter_data = loader.load_quarterly_data(quarter)
            
            # 2. Generate LLM views
            print(f"\nStep 2/4: Generating LLM views...")
            llm_views = analyzer.generate_views(quarter_data)
            
            # 3. Optimize portfolio
            print(f"\nStep 3/4: Optimizing portfolio...")
            portfolio = optimizer.optimize_quarter(
                quarter_data, 
                llm_views,
                lookback_days=config['data']['lookback_days']
            )
            portfolio_results.append(portfolio)
            
            # 4. Save results
            print(f"\nStep 4/4: Saving results...")
            optimizer.save_results(portfolio)
            
            print(f"\n✓ Completed processing for {quarter}")
            
        except Exception as e:
            print(f"\n✗ Error processing {quarter}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(portfolio_results) == 0:
        print("\n✗ No quarters were successfully processed. Exiting.")
        return
    
    print(f"\n✓ Successfully processed {len(portfolio_results)}/{len(quarters)} quarters")
    
    # Backtesting
    print("\n" + "="*80)
    print("BACKTESTING")
    print("="*80)
    
    # Get stock data and quarter dates
    stock_data = loader.load_stock_data()
    quarter_dates_map = loader.quarter_dates
    
    # Run strategy backtest
    backtest_results = backtester.run_backtest(
        portfolio_results,
        stock_data,
        quarter_dates_map
    )
    
    # Run benchmark
    equal_weights = {ticker: 1/6 for ticker in optimizer.tickers}
    benchmark_results = backtester.run_benchmark(
        equal_weights,
        stock_data,
        quarter_dates_map,
        [p['quarter'] for p in portfolio_results]
    )
    
    # Calculate all metrics
    print("\n" + "="*80)
    print("CALCULATING PERFORMANCE METRICS")
    print("="*80)
    
    all_metrics = PerformanceMetrics.calculate_all_metrics(
        backtest_results,
        benchmark_results
    )
    
    print("\n✓ Performance metrics calculated")
    
    # Generate visualizations
    if not args.skip_charts:
        visualizer.create_all_charts(backtest_results, benchmark_results)
    
    # Generate reports
    report_gen.generate_full_report(
        backtest_results,
        benchmark_results,
        all_metrics,
        portfolio_results
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    strategy_return = backtest_results['summary']['annualized_return']
    benchmark_return = benchmark_results['summary']['annualized_return']
    outperformance = strategy_return - benchmark_return
    
    print(f"\nStrategy Annualized Return:    {strategy_return:>10.2%}")
    print(f"Benchmark Annualized Return:   {benchmark_return:>10.2%}")
    print(f"Outperformance:                {outperformance:>10.2%}")
    
    print(f"\nStrategy Sharpe Ratio:         {backtest_results['summary']['sharpe_ratio']:>10.4f}")
    print(f"Benchmark Sharpe Ratio:        {benchmark_results['summary']['sharpe_ratio']:>10.4f}")
    
    print(f"\nStrategy Max Drawdown:         {backtest_results['summary']['max_drawdown']:>10.2%}")
    print(f"Benchmark Max Drawdown:        {benchmark_results['summary']['max_drawdown']:>10.2%}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\nOutput Files:")
    print("  - output/portfolios/         : Quarterly portfolio allocations")
    print("  - output/views/              : LLM-generated views")
    if not args.skip_charts:
        print("  - output/charts/             : Performance visualizations")
    print("  - output/reports/            : Full analysis reports")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

