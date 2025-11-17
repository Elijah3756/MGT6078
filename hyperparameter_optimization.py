#!/usr/bin/env python3
"""
Hyperparameter Optimization
Grid search and random search for optimal hyperparameters
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from itertools import product
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import DataLoader
from data.formatter import DataFormatter
from llm.analyzer import LLMAnalyzer
from optimization.portfolio_optimizer import PortfolioOptimizer
from backtesting.backtester import Backtester
from backtesting.metrics import PerformanceMetrics


class HyperparameterOptimizer:
    """Optimize hyperparameters using grid search or random search"""
    
    def __init__(self, quarters: List[str] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            quarters: List of quarters to use for optimization
        """
        self.quarters = quarters or ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
        self.loader = DataLoader()
        self.formatter = DataFormatter()
        self.backtester = Backtester(initial_capital=100000)
        
    def define_search_space(self) -> Dict:
        """
        Define hyperparameter search space
        
        Returns:
            Dictionary with parameter ranges
        """
        return {
            # LLM hyperparameters (for finance-llm local model)
            'llm_temperature': [0.3, 0.5, 0.7, 0.9],  # Lower = more deterministic
            'llm_top_p': [0.7, 0.9, 0.95],  # Nucleus sampling
            
            # Black-Litterman hyperparameters
            'risk_aversion': [1.5, 2.0, 2.5, 3.0, 3.5],  # Higher = more conservative
            'tau_for_covariance': [0.01, 0.025, 0.05, 0.075, 0.1],  # Prior uncertainty
            'tau_omega': [0.025, 0.05, 0.075, 0.1],  # Views uncertainty
            'relative_confidence': [0.5, 0.75, 1.0, 1.25, 1.5],  # Confidence scaling
            
            # Portfolio constraints
            'allow_shorts': [True, False],
            'max_weight': [1.0, 1.5, 2.0],  # Max position size
            
            # Data settings
            'lookback_days': [126, 189, 252, 315]  # 6 months to 15 months
        }
    
    def generate_random_config(self, search_space: Dict) -> Dict:
        """
        Generate random hyperparameter configuration
        
        Args:
            search_space: Parameter search space
            
        Returns:
            Random configuration dictionary
        """
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        return config
    
    def evaluate_config(self, config: Dict, llm_type: str = 'simulated') -> Dict:
        """
        Evaluate a hyperparameter configuration
        
        Args:
            config: Hyperparameter configuration
            llm_type: LLM model type to use
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Initialize components with config
            analyzer = LLMAnalyzer(
                model_type=llm_type,
                temperature=config.get('llm_temperature', 0.7),
                top_p=config.get('llm_top_p', 0.9)
            )
            optimizer = PortfolioOptimizer(
                allow_shorts=config.get('allow_shorts', True),
                max_weight=config.get('max_weight', 2.0)
            )
            
            # Set Black-Litterman hyperparameters
            optimizer.set_hyperparameters(
                risk_aversion=config.get('risk_aversion', 2.5),
                tau_for_covariance=config.get('tau_for_covariance', 0.025),
                tau_omega=config.get('tau_omega', 0.05),
                relative_confidence=config.get('relative_confidence', 1.0)
            )
            
            # Process quarters
            portfolio_results = []
            for quarter in self.quarters:
                try:
                    quarter_data = self.loader.load_quarterly_data(quarter)
                    llm_views = analyzer.generate_views(quarter_data)
                    
                    portfolio = optimizer.optimize_quarter(
                        quarter_data, 
                        llm_views,
                        lookback_days=config.get('lookback_days', 252)
                    )
                    portfolio_results.append(portfolio)
                except Exception as e:
                    print(f"  Error processing {quarter}: {e}")
                    continue
            
            if len(portfolio_results) == 0:
                return {'error': 'No successful portfolios'}
            
            # Run backtest
            stock_data = self.loader.load_stock_data()
            quarter_dates_map = self.loader.quarter_dates
            
            backtest_results = self.backtester.run_backtest(
                portfolio_results,
                stock_data,
                quarter_dates_map
            )
            
            # Calculate metrics
            equal_weights = {ticker: 1/6 for ticker in optimizer.tickers}
            benchmark_results = self.backtester.run_benchmark(
                equal_weights,
                stock_data,
                quarter_dates_map,
                [p['quarter'] for p in portfolio_results]
            )
            
            metrics = PerformanceMetrics.calculate_all_metrics(
                backtest_results,
                benchmark_results
            )
            
            # Return key metrics
            return {
                'config': config,
                'annualized_return': metrics['annualized_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics['max_drawdown'],
                'outperformance': metrics.get('outperformance', 0),
                'information_ratio': metrics.get('information_ratio', 0),
                'total_return': metrics['total_return'],
                'volatility': metrics['volatility'],
                'win_rate': metrics['win_rate']
            }
            
        except Exception as e:
            return {'error': str(e), 'config': config}
    
    def grid_search(self, search_space: Dict, llm_type: str = 'simulated',
                   max_combinations: int = 50) -> List[Dict]:
        """
        Perform grid search over hyperparameter space
        
        Args:
            search_space: Parameter search space
            llm_type: LLM model type
            max_combinations: Maximum number of combinations to test
            
        Returns:
            List of evaluation results
        """
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[p] for p in param_names]
        
        all_combinations = list(product(*param_values))
        
        # Limit if too many
        if len(all_combinations) > max_combinations:
            print(f"Too many combinations ({len(all_combinations)}). "
                  f"Randomly sampling {max_combinations}...")
            all_combinations = random.sample(all_combinations, max_combinations)
        
        results = []
        print(f"\n{'='*80}")
        print(f"GRID SEARCH: Testing {len(all_combinations)} configurations")
        print(f"{'='*80}\n")
        
        for i, combination in enumerate(all_combinations, 1):
            config = dict(zip(param_names, combination))
            print(f"[{i}/{len(all_combinations)}] Testing: {config}")
            
            result = self.evaluate_config(config, llm_type)
            if 'error' not in result:
                print(f"  ✓ Return: {result['annualized_return']:.2%}, "
                      f"Sharpe: {result['sharpe_ratio']:.4f}")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        return results
    
    def random_search(self, search_space: Dict, n_trials: int = 20,
                     llm_type: str = 'simulated') -> List[Dict]:
        """
        Perform random search over hyperparameter space
        
        Args:
            search_space: Parameter search space
            n_trials: Number of random configurations to test
            llm_type: LLM model type
            
        Returns:
            List of evaluation results
        """
        results = []
        print(f"\n{'='*80}")
        print(f"RANDOM SEARCH: Testing {n_trials} random configurations")
        print(f"{'='*80}\n")
        
        for i in range(n_trials):
            config = self.generate_random_config(search_space)
            print(f"[{i+1}/{n_trials}] Testing: {config}")
            
            result = self.evaluate_config(config, llm_type)
            if 'error' not in result:
                print(f"  ✓ Return: {result['annualized_return']:.2%}, "
                      f"Sharpe: {result['sharpe_ratio']:.4f}")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict], top_n: int = 10) -> Dict:
        """
        Analyze optimization results and find best configurations
        
        Args:
            results: List of evaluation results
            top_n: Number of top configurations to return
            
        Returns:
            Analysis dictionary
        """
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        
        if len(valid_results) == 0:
            return {'error': 'No valid results'}
        
        # Sort by Sharpe ratio (primary) and return (secondary)
        sorted_results = sorted(
            valid_results,
            key=lambda x: (x['sharpe_ratio'], x['annualized_return']),
            reverse=True
        )
        
        top_configs = sorted_results[:top_n]
        
        # Calculate statistics
        returns = [r['annualized_return'] for r in valid_results]
        sharpes = [r['sharpe_ratio'] for r in valid_results]
        
        analysis = {
            'total_trials': len(results),
            'valid_trials': len(valid_results),
            'best_config': top_configs[0]['config'],
            'best_metrics': {
                'annualized_return': top_configs[0]['annualized_return'],
                'sharpe_ratio': top_configs[0]['sharpe_ratio'],
                'sortino_ratio': top_configs[0]['sortino_ratio'],
                'max_drawdown': top_configs[0]['max_drawdown'],
                'outperformance': top_configs[0]['outperformance']
            },
            'top_configs': top_configs,
            'statistics': {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'min_sharpe': np.min(sharpes),
                'max_sharpe': np.max(sharpes)
            }
        }
        
        return analysis
    
    def save_results(self, results: List[Dict], analysis: Dict, 
                    output_dir: str = 'output/hyperparameter_optimization'):
        """
        Save optimization results
        
        Args:
            results: All evaluation results
            analysis: Analysis dictionary
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all results
        results_path = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, f'analysis_{timestamp}.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Saved results to {results_path}")
        print(f"✓ Saved analysis to {analysis_path}")
        
        # Generate visualizations
        try:
            from hyperparameter_visualization import HyperparameterVisualizer
            print(f"\n{'='*80}")
            print("GENERATING VISUALIZATIONS")
            print(f"{'='*80}")
            visualizer = HyperparameterVisualizer(results_dir=output_dir)
            visualizer.create_summary_report({'results': results, 'analysis': analysis})
        except Exception as e:
            print(f"\nWarning: Could not generate visualizations: {e}")
            print("You can generate them later with: python hyperparameter_visualization.py")
        
        # Print summary
        print(f"\n{'='*80}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nBest Configuration:")
        for param, value in analysis['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nBest Metrics:")
        for metric, value in analysis['best_metrics'].items():
            if isinstance(value, float):
                if 'ratio' in metric.lower() or 'drawdown' in metric.lower():
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")
        print(f"\nStatistics across all trials:")
        stats = analysis['statistics']
        print(f"  Mean Return: {stats['mean_return']:.2%} ± {stats['std_return']:.2%}")
        print(f"  Mean Sharpe: {stats['mean_sharpe']:.4f} ± {stats['std_sharpe']:.4f}")
        print(f"  Return Range: [{stats['min_return']:.2%}, {stats['max_return']:.2%}]")
        print(f"  Sharpe Range: [{stats['min_sharpe']:.4f}, {stats['max_sharpe']:.4f}]")


def main():
    """Run hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--method', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Search method: grid or random')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of trials for random search')
    parser.add_argument('--max-combinations', type=int, default=50,
                       help='Max combinations for grid search')
    parser.add_argument('--llm-type', type=str, default='finance-llm',
                       choices=['finance-llm', 'simulated'],
                       help='LLM model type: finance-llm (local AdaptLLM model, default) or simulated (fallback only)')
    parser.add_argument('--quarters', type=str, nargs='+',
                       default=['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'],
                       help='Quarters to use for optimization')
    
    args = parser.parse_args()
    
    optimizer = HyperparameterOptimizer(quarters=args.quarters)
    search_space = optimizer.define_search_space()
    
    if args.method == 'grid':
        results = optimizer.grid_search(
            search_space,
            llm_type=args.llm_type,
            max_combinations=args.max_combinations
        )
    else:
        results = optimizer.random_search(
            search_space,
            n_trials=args.n_trials,
            llm_type=args.llm_type
        )
    
    analysis = optimizer.analyze_results(results)
    optimizer.save_results(results, analysis)


if __name__ == "__main__":
    main()

