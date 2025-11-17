"""
Portfolio Optimizer
Wrapper around BlackLitterman model for 6 sector ETFs
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import tempfile

# Add parent and BlackLitterman directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BlackLitterman'))
from BlackLitterman import BlackLitterman

from optimization.views_converter import ViewsConverter
from data.stock_processor import StockProcessor


class PortfolioOptimizer:
    """Optimize portfolio using Black-Litterman model with LLM views"""
    
    def __init__(self, tickers=None, allow_shorts=True):
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
        self.converter = ViewsConverter(self.tickers)
        self.processor = StockProcessor(self.tickers)
        self.allow_shorts = allow_shorts  # Allow short positions
        
        # Black-Litterman hyperparameters
        self.risk_aversion = 2.5  # Typical for equity portfolios
        self.tau_for_covariance = 0.025  # Uncertainty in prior
        self.tau_omega = 0.05  # Uncertainty in views
        self.relative_confidence = 1.0
    
    def set_hyperparameters(self, risk_aversion=None, tau_for_covariance=None,
                           tau_omega=None, relative_confidence=None):
        """
        Set Black-Litterman hyperparameters
        
        Args:
            risk_aversion: Risk aversion coefficient (2-4 typical)
            tau_for_covariance: Tau for covariance (0.025-0.1)
            tau_omega: Tau for omega (0.025-0.1)
            relative_confidence: Relative confidence (0.5-1.0)
        """
        if risk_aversion is not None:
            self.risk_aversion = risk_aversion
        if tau_for_covariance is not None:
            self.tau_for_covariance = tau_for_covariance
        if tau_omega is not None:
            self.tau_omega = tau_omega
        if relative_confidence is not None:
            self.relative_confidence = relative_confidence
    
    def prepare_returns_data(self, stock_data: Dict[str, pd.DataFrame], 
                            quarter_end_date: str,
                            lookback_days: int = 252) -> str:
        """
        Prepare returns CSV file for Black-Litterman model
        
        Args:
            stock_data: Dictionary of stock DataFrames
            quarter_end_date: End date of the quarter (YYYY-MM-DD format)
            lookback_days: Number of days of historical data to use
            
        Returns:
            Path to temporary CSV file
        """
        # Calculate returns filtered up to quarter end date
        returns_df = self.processor.calculate_returns_up_to_date(
            stock_data, quarter_end_date, lookback_days
        )
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        returns_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def optimize_quarter(self, quarter_data: Dict, llm_views: Dict, 
                        lookback_days: int = 252) -> Dict:
        """
        Optimize portfolio for a quarter using LLM views
        
        Args:
            quarter_data: Dictionary with quarterly data including stock_data
            llm_views: Dictionary with LLM-generated views
            lookback_days: Days of historical data for covariance
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}")
        print(f"OPTIMIZING PORTFOLIO - {quarter_data['quarter']}")
        print(f"{'='*80}")
        
        # 1. Prepare returns data filtered up to quarter end date
        stock_data = quarter_data['stock_data']
        quarter_end_date = quarter_data['date_range']['end']
        returns_csv = self.prepare_returns_data(stock_data, quarter_end_date, lookback_days)
        
        print(f"✓ Prepared returns data ({lookback_days} days up to {quarter_end_date})")
        
        # 2. Convert LLM views to P, Q, Omega matrices
        P, Q, Omega = self.converter.convert_to_matrices(llm_views)
        
        # Validate matrices before optimization
        try:
            self.converter.validate_matrices(P, Q, Omega)
            print(f"✓ Converted views to matrices ({len(Q)} views)")
            print(f"✓ Matrix validation passed")
        except Exception as e:
            print(f"✗ Matrix validation failed: {e}")
            raise
        
        # 3. Get market weights (equal weight as starting point)
        market_weights = self.processor.get_market_weights(method='equal')
        
        print(f"✓ Using equal market weights: {market_weights}")
        
        # 4. Initialize Black-Litterman model
        bl = BlackLitterman(returns_csv)
        
        # 5. Run optimization with optional short constraints
        try:
            # Temporarily modify Black-Litterman to add constraints
            original_optimize = bl.optimize_portfolio
            
            def constrained_optimize(initial_weights):
                from scipy.optimize import minimize
                import math
                
                # Constraint: weights must sum to 1
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                ]
                
                # Set bounds based on allow_shorts flag
                if self.allow_shorts:
                    # Allow short positions: no lower bound, but reasonable upper bound
                    # Typically allow shorts up to -100% and longs up to 200% per asset
                    bounds = [(-1.0, 2.0) for _ in range(len(initial_weights))]
                    constraint_type = "long/short"
                else:
                    # Long-only: weights between 0% and 100%
                    bounds = [(0.0, 1.0) for _ in range(len(initial_weights))]
                    constraint_type = "long-only"
                
                result = minimize(
                    bl.neg_sharpe_ratio, 
                    initial_weights,
                    method='SLSQP', 
                    constraints=constraints,
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                
                opt_weights = result.x
                max_sharpe = -result.fun
                
                return opt_weights, max_sharpe
            
            bl.optimize_portfolio = constrained_optimize
            
            # Pass Omega_matrix from LLM confidences to BlackLitterman
            opt_weights, sharpe = bl.black_litterman_weights(
                risk_aversion=self.risk_aversion,
                tau_for_covariance=self.tau_for_covariance,
                market_weights=list(market_weights),
                P_matrix=P,
                Q_vector=Q,
                tau_omega=self.tau_omega,
                relative_confidence=self.relative_confidence,
                Omega_matrix=Omega  # Pass LLM-derived Omega
            )
            
            constraint_msg = "long/short" if self.allow_shorts else "long-only"
            print(f"\n✓ Optimization completed ({constraint_msg})")
            print(f"  Sharpe Ratio: {sharpe:.4f}")
            
        except Exception as e:
            print(f"Error in Black-Litterman optimization: {e}")
            print("Using equal weights as fallback")
            opt_weights = market_weights
            sharpe = 0.0
        
        finally:
            # Clean up temporary file
            os.unlink(returns_csv)
        
        # 6. Format results
        results = {
            'quarter': quarter_data['quarter'],
            'weights': opt_weights,
            'sharpe': sharpe,
            'ticker_weights': {ticker: float(weight) 
                              for ticker, weight in zip(self.tickers, opt_weights)},
            'views_used': llm_views,
            'hyperparameters': {
                'risk_aversion': self.risk_aversion,
                'tau_for_covariance': self.tau_for_covariance,
                'tau_omega': self.tau_omega,
                'relative_confidence': self.relative_confidence
            }
        }
        
        # Print allocation
        self.print_allocation(results)
        
        return results
    
    def print_allocation(self, results: Dict):
        """
        Print portfolio allocation
        
        Args:
            results: Optimization results dictionary
        """
        print(f"\n{'='*80}")
        print(f"PORTFOLIO ALLOCATION - {results['quarter']}")
        print(f"{'='*80}")
        
        print(f"\nSharpe Ratio: {results['sharpe']:.4f}\n")
        
        # Separate longs and shorts for clarity
        longs = {t: w for t, w in results['ticker_weights'].items() if w >= 0}
        shorts = {t: w for t, w in results['ticker_weights'].items() if w < 0}
        
        print("Long Positions:")
        if longs:
            for ticker in self.tickers:
                if ticker in longs:
                    weight = longs[ticker]
                    print(f"  {ticker}: {weight:6.2%}")
        else:
            print("  (none)")
        
        if shorts:
            print("\nShort Positions:")
            for ticker in self.tickers:
                if ticker in shorts:
                    weight = shorts[ticker]
                    print(f"  {ticker}: {weight:6.2%}")
        
        total = sum(results['ticker_weights'].values())
        print(f"\nNet Exposure: {total:6.2%}")
        
        if shorts:
            gross_exposure = sum(abs(w) for w in results['ticker_weights'].values())
            print(f"Gross Exposure: {gross_exposure:6.2%}")
    
    def save_results(self, results: Dict, output_dir: str = 'output/portfolios'):
        """
        Save optimization results
        
        Args:
            results: Optimization results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        output_path = os.path.join(output_dir, f"{results['quarter']}_portfolio.json")
        
        # Make results JSON serializable
        json_results = results.copy()
        json_results['weights'] = json_results['weights'].tolist() if isinstance(json_results['weights'], np.ndarray) else list(json_results['weights'])
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Saved portfolio to {output_path}")


def main():
    """Test portfolio optimizer"""
    from data.loader import DataLoader
    from llm.analyzer import LLMAnalyzer
    
    # Load data
    loader = DataLoader()
    analyzer = LLMAnalyzer(model_type='simulated')
    optimizer = PortfolioOptimizer()
    
    # Load Q1 2024
    print("Loading Q1 2024 data...")
    data = loader.load_quarterly_data('Q1_2024')
    
    # Generate views
    print("\nGenerating investment views...")
    views = analyzer.generate_views(data)
    
    # Optimize portfolio
    print("\nOptimizing portfolio...")
    results = optimizer.optimize_quarter(data, views, lookback_days=252)
    
    # Save results
    optimizer.save_results(results)


if __name__ == "__main__":
    main()

