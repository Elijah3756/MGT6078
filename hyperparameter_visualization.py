#!/usr/bin/env python3
"""
Hyperparameter Optimization Visualization
Create graphs and analysis from hyperparameter optimization results
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import glob


class HyperparameterVisualizer:
    """Visualize hyperparameter optimization results"""
    
    def __init__(self, results_dir: str = 'output/hyperparameter_optimization'):
        """
        Initialize visualizer
        
        Args:
            results_dir: Directory containing optimization results
        """
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('ggplot')
        sns.set_palette("husl")
    
    def load_latest_results(self) -> Optional[Dict]:
        """
        Load the most recent optimization results
        
        Returns:
            Dictionary with results and analysis, or None if not found
        """
        # Find latest results file
        results_files = glob.glob(os.path.join(self.results_dir, 'results_*.json'))
        analysis_files = glob.glob(os.path.join(self.results_dir, 'analysis_*.json'))
        
        if not results_files or not analysis_files:
            print(f"No results found in {self.results_dir}")
            return None
        
        # Get most recent files
        latest_results = max(results_files, key=os.path.getctime)
        latest_analysis = max(analysis_files, key=os.path.getctime)
        
        print(f"Loading results from: {os.path.basename(latest_results)}")
        print(f"Loading analysis from: {os.path.basename(latest_analysis)}")
        
        with open(latest_results, 'r') as f:
            results = json.load(f)
        
        with open(latest_analysis, 'r') as f:
            analysis = json.load(f)
        
        return {'results': results, 'analysis': analysis}
    
    def load_results_from_file(self, results_path: str, analysis_path: str) -> Dict:
        """
        Load results from specific files
        
        Args:
            results_path: Path to results JSON file
            analysis_path: Path to analysis JSON file
            
        Returns:
            Dictionary with results and analysis
        """
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
        
        return {'results': results, 'analysis': analysis}
    
    def create_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for analysis
        
        Args:
            results: List of result dictionaries
            
        Returns:
            DataFrame with hyperparameters and metrics
        """
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        
        if len(valid_results) == 0:
            raise ValueError("No valid results found")
        
        # Extract data
        data = []
        for result in valid_results:
            row = {}
            
            # Add hyperparameters
            config = result.get('config', {})
            for key, value in config.items():
                row[key] = value
            
            # Add metrics
            row['annualized_return'] = result.get('annualized_return', 0)
            row['sharpe_ratio'] = result.get('sharpe_ratio', 0)
            row['sortino_ratio'] = result.get('sortino_ratio', 0)
            row['max_drawdown'] = result.get('max_drawdown', 0)
            row['outperformance'] = result.get('outperformance', 0)
            row['information_ratio'] = result.get('information_ratio', 0)
            row['total_return'] = result.get('total_return', 0)
            row['volatility'] = result.get('volatility', 0)
            row['win_rate'] = result.get('win_rate', 0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_hyperparameter_importance(self, df: pd.DataFrame, 
                                      metric: str = 'sharpe_ratio',
                                      top_n: int = 10):
        """
        Plot importance of each hyperparameter
        
        Args:
            df: DataFrame with results
            metric: Metric to analyze
            top_n: Number of top configurations to use
        """
        # Get top configurations
        top_df = df.nlargest(top_n, metric)
        
        # Calculate correlation/importance for each hyperparameter
        hyperparams = [col for col in df.columns 
                      if col not in ['annualized_return', 'sharpe_ratio', 
                                    'sortino_ratio', 'max_drawdown', 
                                    'outperformance', 'information_ratio',
                                    'total_return', 'volatility', 'win_rate']]
        
        importances = []
        for param in hyperparams:
            if df[param].dtype in ['object', 'bool']:
                # For categorical, use difference between top and bottom
                top_mean = top_df[metric].mean()
                bottom_mean = df.nsmallest(top_n, metric)[metric].mean()
                importance = abs(top_mean - bottom_mean)
            else:
                # For numerical, use correlation
                importance = abs(df[param].corr(df[metric]))
            
            importances.append({
                'parameter': param,
                'importance': importance
            })
        
        importance_df = pd.DataFrame(importances).sort_values('importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='parameter', ax=ax)
        ax.set_xlabel('Importance (Correlation with ' + metric + ')')
        ax.set_title(f'Hyperparameter Importance for {metric}')
        ax.set_ylabel('')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'hyperparameter_importance_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved hyperparameter importance plot to {output_path}")
    
    def plot_hyperparameter_vs_metric(self, df: pd.DataFrame, 
                                     hyperparam: str,
                                     metric: str = 'sharpe_ratio'):
        """
        Plot relationship between a hyperparameter and metric
        
        Args:
            df: DataFrame with results
            hyperparam: Hyperparameter to plot
            metric: Metric to plot against
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if df[hyperparam].dtype in ['object', 'bool']:
            # Categorical: box plot
            sns.boxplot(data=df, x=hyperparam, y=metric, ax=ax)
            ax.set_xlabel(hyperparam)
        else:
            # Numerical: scatter plot with regression line
            sns.scatterplot(data=df, x=hyperparam, y=metric, ax=ax, alpha=0.6)
            sns.regplot(data=df, x=hyperparam, y=metric, ax=ax, scatter=False, color='red')
            ax.set_xlabel(hyperparam)
        
        ax.set_ylabel(metric)
        ax.set_title(f'{hyperparam} vs {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 
                                  f'{hyperparam}_vs_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {hyperparam} vs {metric} plot")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Plot correlation heatmap between hyperparameters and metrics
        
        Args:
            df: DataFrame with results
        """
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_df = df[numerical_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Hyperparameter and Metric Correlations')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved correlation heatmap to {output_path}")
    
    def plot_metric_distributions(self, df: pd.DataFrame):
        """
        Plot distributions of key metrics
        
        Args:
            df: DataFrame with results
        """
        metrics = ['sharpe_ratio', 'annualized_return', 'max_drawdown', 
                  'outperformance', 'volatility']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i]
                df[metric].hist(bins=20, ax=ax, edgecolor='black')
                ax.axvline(df[metric].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df[metric].mean():.4f}')
                ax.axvline(df[metric].median(), color='green', linestyle='--', 
                          label=f'Median: {df[metric].median():.4f}')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {metric}')
                ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'metric_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metric distributions to {output_path}")
    
    def plot_2d_hyperparameter_space(self, df: pd.DataFrame,
                                    param1: str, param2: str,
                                    metric: str = 'sharpe_ratio'):
        """
        Plot 2D hyperparameter space colored by metric
        
        Args:
            df: DataFrame with results
            param1: First hyperparameter
            param2: Second hyperparameter
            metric: Metric to color by
        """
        if param1 not in df.columns or param2 not in df.columns:
            print(f"Warning: {param1} or {param2} not in results")
            return
        
        # Convert categorical to numeric if needed
        plot_df = df.copy()
        if plot_df[param1].dtype == 'object':
            plot_df[param1] = pd.Categorical(plot_df[param1]).codes
        if plot_df[param2].dtype == 'object':
            plot_df[param2] = pd.Categorical(plot_df[param2]).codes
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(plot_df[param1], plot_df[param2], 
                           c=plot_df[metric], cmap='viridis', 
                           s=100, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'{param1} vs {param2} (colored by {metric})')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(metric)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 
                                  f'2d_space_{param1}_{param2}_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved 2D space plot to {output_path}")
    
    def plot_pareto_frontier(self, df: pd.DataFrame,
                            x_metric: str = 'volatility',
                            y_metric: str = 'annualized_return'):
        """
        Plot Pareto frontier (risk vs return)
        
        Args:
            df: DataFrame with results
            x_metric: X-axis metric (typically risk)
            y_metric: Y-axis metric (typically return)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(df[x_metric], df[y_metric], 
                           c=df['sharpe_ratio'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')
        
        # Highlight top 5 by Sharpe ratio
        top_5 = df.nlargest(5, 'sharpe_ratio')
        ax.scatter(top_5[x_metric], top_5[y_metric], 
                  s=300, marker='*', color='red', 
                  edgecolors='black', linewidth=2,
                  label='Top 5 by Sharpe Ratio', zorder=5)
        
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f'Pareto Frontier: {x_metric} vs {y_metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 
                                  f'pareto_frontier_{x_metric}_{y_metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Pareto frontier plot to {output_path}")
    
    def plot_top_configurations(self, analysis: Dict, top_n: int = 10):
        """
        Plot comparison of top configurations
        
        Args:
            analysis: Analysis dictionary
            top_n: Number of top configurations to show
        """
        top_configs = analysis.get('top_configs', [])[:top_n]
        
        if len(top_configs) == 0:
            print("No top configurations found")
            return
        
        # Extract metrics
        metrics = ['sharpe_ratio', 'annualized_return', 'sortino_ratio', 
                  'max_drawdown', 'outperformance']
        
        config_names = [f"Config {i+1}" for i in range(len(top_configs))]
        metric_data = {metric: [c.get(metric, 0) for c in top_configs] 
                      for metric in metrics}
        
        # Create grouped bar chart
        x = np.arange(len(config_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, metric in enumerate(metrics):
            # Normalize metrics to 0-1 for comparison
            values = np.array(metric_data[metric])
            if metric == 'max_drawdown':
                # Invert drawdown (lower is better)
                values = -values
                values = (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                values = (values - values.min()) / (values.max() - values.min() + 1e-10)
            
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Normalized Score')
        ax.set_title(f'Top {top_n} Configurations Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'top_{top_n}_configurations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved top configurations plot to {output_path}")
    
    def create_summary_report(self, data: Dict):
        """
        Create comprehensive visualization report
        
        Args:
            data: Dictionary with results and analysis
        """
        results = data['results']
        analysis = data['analysis']
        
        print(f"\n{'='*80}")
        print("CREATING HYPERPARAMETER OPTIMIZATION VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        # Create DataFrame
        df = self.create_dataframe(results)
        print(f"✓ Loaded {len(df)} valid configurations\n")
        
        # 1. Hyperparameter importance
        print("1. Analyzing hyperparameter importance...")
        self.plot_hyperparameter_importance(df, metric='sharpe_ratio')
        self.plot_hyperparameter_importance(df, metric='annualized_return')
        
        # 2. Key hyperparameter relationships
        print("\n2. Plotting hyperparameter relationships...")
        key_params = ['risk_aversion', 'tau_for_covariance', 'tau_omega', 
                     'relative_confidence', 'llm_temperature', 'lookback_days']
        
        for param in key_params:
            if param in df.columns:
                self.plot_hyperparameter_vs_metric(df, param, 'sharpe_ratio')
                self.plot_hyperparameter_vs_metric(df, param, 'annualized_return')
        
        # 3. Correlation heatmap
        print("\n3. Creating correlation heatmap...")
        self.plot_correlation_heatmap(df)
        
        # 4. Metric distributions
        print("\n4. Plotting metric distributions...")
        self.plot_metric_distributions(df)
        
        # 5. 2D hyperparameter spaces
        print("\n5. Plotting 2D hyperparameter spaces...")
        self.plot_2d_hyperparameter_space(df, 'risk_aversion', 'tau_for_covariance')
        self.plot_2d_hyperparameter_space(df, 'llm_temperature', 'risk_aversion')
        
        # 6. Pareto frontier
        print("\n6. Creating Pareto frontier...")
        self.plot_pareto_frontier(df, 'volatility', 'annualized_return')
        
        # 7. Top configurations
        print("\n7. Plotting top configurations...")
        self.plot_top_configurations(analysis, top_n=10)
        
        print(f"\n{'='*80}")
        print(f"✓ All visualizations saved to {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Print summary statistics
        print("Summary Statistics:")
        print(f"  Total trials: {analysis.get('total_trials', 0)}")
        print(f"  Valid trials: {analysis.get('valid_trials', 0)}")
        print(f"\nBest Configuration:")
        best_config = analysis.get('best_config', {})
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"\nBest Metrics:")
        best_metrics = analysis.get('best_metrics', {})
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                if 'ratio' in metric.lower() or 'drawdown' in metric.lower():
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:.2%}")


def main():
    """Create visualizations from latest optimization results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize hyperparameter optimization results')
    parser.add_argument('--results-dir', type=str, 
                       default='output/hyperparameter_optimization',
                       help='Directory containing optimization results')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Specific results JSON file (optional)')
    parser.add_argument('--analysis-file', type=str, default=None,
                       help='Specific analysis JSON file (optional)')
    
    args = parser.parse_args()
    
    visualizer = HyperparameterVisualizer(results_dir=args.results_dir)
    
    if args.results_file and args.analysis_file:
        data = visualizer.load_results_from_file(args.results_file, args.analysis_file)
    else:
        data = visualizer.load_latest_results()
    
    if data is None:
        print("No results found. Run hyperparameter optimization first.")
        return
    
    visualizer.create_summary_report(data)


if __name__ == "__main__":
    main()

