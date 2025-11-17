"""
Portfolio Optimizer
Wrapper around BlackLitterman model for 6 sector ETFs
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import tempfile
import logging
from contextlib import contextmanager

# Add parent and BlackLitterman directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BlackLitterman'))
from BlackLitterman import BlackLitterman

from optimization.views_converter import ViewsConverter
from data.stock_processor import StockProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Optimize portfolio using Black-Litterman model with LLM views"""
    
    def __init__(self, tickers: Optional[List[str]] = None, 
                 allow_shorts: bool = True,
                 min_weight: float = -1.0,
                 max_weight: float = 2.0,
                 max_iterations: int = 1000):
        """
        Initialize Portfolio Optimizer
        
        Args:
            tickers: List of ticker symbols (default: 6 sector ETFs)
            allow_shorts: Whether to allow short positions
            min_weight: Minimum weight per asset (for shorts, typically -1.0)
            max_weight: Maximum weight per asset (for leveraged longs, typically 2.0)
            max_iterations: Maximum iterations for optimization solver
        """
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
        self.converter = ViewsConverter(self.tickers)
        self.processor = StockProcessor(self.tickers)
        self.allow_shorts = allow_shorts
        self.min_weight = min_weight if allow_shorts else 0.0
        self.max_weight = max_weight
        self.max_iterations = max_iterations
        
        # Black-Litterman hyperparameters
        self.risk_aversion = 2.5  # Typical for equity portfolios
        self.tau_for_covariance = 0.025  # Uncertainty in prior
        self.tau_omega = 0.05  # Uncertainty in views
        self.relative_confidence = 1.0
        
        # Validation thresholds
        self.min_data_points = 60  # Minimum days of data required
        self.min_sharpe_threshold = -10.0  # Minimum acceptable Sharpe ratio
    
    def set_hyperparameters(self, risk_aversion: Optional[float] = None, 
                           tau_for_covariance: Optional[float] = None,
                           tau_omega: Optional[float] = None, 
                           relative_confidence: Optional[float] = None):
        """
        Set Black-Litterman hyperparameters
        
        Args:
            risk_aversion: Risk aversion coefficient (2-4 typical)
            tau_for_covariance: Tau for covariance (0.025-0.1)
            tau_omega: Tau for omega (0.025-0.1)
            relative_confidence: Relative confidence (0.5-1.0)
            
        Raises:
            ValueError: If hyperparameters are outside reasonable ranges
        """
        if risk_aversion is not None:
            if not (0.5 <= risk_aversion <= 10.0):
                raise ValueError(f"risk_aversion should be between 0.5 and 10.0, got {risk_aversion}")
            self.risk_aversion = risk_aversion
            
        if tau_for_covariance is not None:
            if not (0.001 <= tau_for_covariance <= 0.5):
                raise ValueError(f"tau_for_covariance should be between 0.001 and 0.5, got {tau_for_covariance}")
            self.tau_for_covariance = tau_for_covariance
            
        if tau_omega is not None:
            if not (0.001 <= tau_omega <= 0.5):
                raise ValueError(f"tau_omega should be between 0.001 and 0.5, got {tau_omega}")
            self.tau_omega = tau_omega
            
        if relative_confidence is not None:
            if not (0.1 <= relative_confidence <= 2.0):
                raise ValueError(f"relative_confidence should be between 0.1 and 2.0, got {relative_confidence}")
            self.relative_confidence = relative_confidence
    
    @contextmanager
    def _temporary_returns_file(self, stock_data: Dict[str, pd.DataFrame], 
                                 quarter_end_date: str,
                                 lookback_days: int = 252):
        """
        Context manager for temporary returns CSV file
        
        Args:
            stock_data: Dictionary of stock DataFrames
            quarter_end_date: End date of the quarter (YYYY-MM-DD format)
            lookback_days: Number of days of historical data to use
            
        Yields:
            Path to temporary CSV file
            
        Raises:
            ValueError: If insufficient data is available
        """
        # Calculate returns filtered up to quarter end date
        returns_df = self.processor.calculate_returns_up_to_date(
            stock_data, quarter_end_date, lookback_days
        )
        
        # Validate data quality
        if len(returns_df) < self.min_data_points:
            raise ValueError(
                f"Insufficient data: only {len(returns_df)} days available, "
                f"need at least {self.min_data_points} days"
            )
        
        # Check for missing tickers
        missing_tickers = set(self.tickers) - set(returns_df.columns)
        if missing_tickers:
            raise ValueError(f"Missing data for tickers: {missing_tickers}")
        
        # Check for excessive missing values
        missing_pct = returns_df.isna().sum().sum() / (len(returns_df) * len(self.tickers))
        if missing_pct > 0.1:
            logger.warning(f"High percentage of missing values: {missing_pct:.1%}")
        
        # Fill any remaining NaN values with forward fill then backward fill
        returns_df = returns_df.ffill().bfill().fillna(0)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        try:
            returns_df.to_csv(temp_file.name, index=False)
            temp_file.close()
            yield temp_file.name
        finally:
            # Clean up
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def _validate_views(self, llm_views: Dict) -> None:
        """
        Validate LLM views before optimization
        
        Args:
            llm_views: Dictionary with LLM-generated views
            
        Raises:
            ValueError: If views are invalid
        """
        if 'views' not in llm_views:
            raise ValueError("llm_views must contain 'views' key")
        
        views = llm_views['views']
        if not isinstance(views, list) or len(views) == 0:
            raise ValueError("views must be a non-empty list")
        
        # Check all tickers are covered
        view_tickers = {view['ticker'] for view in views if 'ticker' in view}
        missing_tickers = set(self.tickers) - view_tickers
        if missing_tickers:
            logger.warning(f"Missing views for tickers: {missing_tickers}")
        
        # Validate each view
        for i, view in enumerate(views):
            if 'ticker' not in view:
                raise ValueError(f"View {i} missing 'ticker' field")
            if view['ticker'] not in self.tickers:
                raise ValueError(f"View {i} has unknown ticker: {view['ticker']}")
            if 'expected_return' not in view:
                raise ValueError(f"View {i} missing 'expected_return' field")
            if not isinstance(view['expected_return'], (int, float)):
                raise ValueError(f"View {i} expected_return must be numeric")
            if abs(view['expected_return']) > 1.0:
                logger.warning(
                    f"View {i} has extreme expected_return: {view['expected_return']:.2%} "
                    "(expected returns should typically be between -50% and +50%)"
                )
            if 'confidence' not in view:
                raise ValueError(f"View {i} missing 'confidence' field")
            confidence = view['confidence']
            if not (0 <= confidence <= 100):
                raise ValueError(f"View {i} confidence must be between 0 and 100, got {confidence}")
    
    def _optimize_with_constraints(self, bl: BlackLitterman, 
                                   initial_weights: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Optimize portfolio with constraints using scipy.optimize
        
        Args:
            bl: BlackLitterman instance with posterior computed
            initial_weights: Initial weight vector
            
        Returns:
            Tuple of (optimal_weights, sharpe_ratio, optimization_info)
        """
        from scipy.optimize import minimize
        
        # Constraint: weights must sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Set bounds based on allow_shorts flag
        bounds = [(self.min_weight, self.max_weight) for _ in range(len(initial_weights))]
        
        # Run optimization
        result = minimize(
            bl.neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': 1e-9}
        )
        
        # Check convergence
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
            if result.fun is None or np.isnan(result.fun):
                raise RuntimeError(f"Optimization failed: {result.message}")
        
        opt_weights = result.x
        sharpe = -result.fun
        
        # Validate results
        if sharpe < self.min_sharpe_threshold:
            logger.warning(
                f"Very low Sharpe ratio: {sharpe:.4f}. "
                "This may indicate optimization issues or extreme market conditions."
            )
        
        # Check weight sum
        weight_sum = np.sum(opt_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights do not sum to 1.0: {weight_sum:.6f}")
            # Normalize weights
            opt_weights = opt_weights / weight_sum
        
        optimization_info = {
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
            'converged': result.success
        }
        
        return opt_weights, sharpe, optimization_info
    
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
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If optimization fails critically
        """
        quarter = quarter_data.get('quarter', 'Unknown')
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZING PORTFOLIO - {quarter}")
        logger.info(f"{'='*80}")
        
        # 1. Validate inputs
        if 'stock_data' not in quarter_data:
            raise ValueError("quarter_data must contain 'stock_data'")
        if 'date_range' not in quarter_data or 'end' not in quarter_data['date_range']:
            raise ValueError("quarter_data must contain 'date_range' with 'end' key")
        
        self._validate_views(llm_views)
        
        stock_data = quarter_data['stock_data']
        quarter_end_date = quarter_data['date_range']['end']
        
        # Initialize default values in case of early errors
        market_weights = self.processor.get_market_weights(method='equal')
        opt_weights = market_weights.copy()
        sharpe = 0.0
        opt_info = {'success': False, 'message': 'Not optimized', 'iterations': 0, 'converged': False}
        
        # 2. Prepare returns data using context manager
        try:
            with self._temporary_returns_file(stock_data, quarter_end_date, lookback_days) as returns_csv:
                logger.info(f"✓ Prepared returns data ({lookback_days} days up to {quarter_end_date})")
                
                # 3. Convert LLM views to P, Q, Omega matrices
                P, Q, Omega = self.converter.convert_to_matrices(llm_views)
                
                # Validate matrices before optimization
                try:
                    self.converter.validate_matrices(P, Q, Omega)
                    logger.info(f"✓ Converted views to matrices ({len(Q)} views)")
                    logger.info(f"✓ Matrix validation passed")
                except Exception as e:
                    logger.error(f"✗ Matrix validation failed: {e}")
                    raise ValueError(f"Matrix validation failed: {e}") from e
                
                # 4. Get market weights (equal weight as starting point)
                market_weights = self.processor.get_market_weights(method='equal')
                opt_weights = market_weights.copy()  # Update in case we need fallback
                logger.info(f"✓ Using equal market weights: {market_weights}")
                
                # 5. Initialize Black-Litterman model
                bl = BlackLitterman(returns_csv)
                
                # 6. Compute posterior returns and covariance
                try:
                    # Set up views and compute posterior
                    bl._set_hyperparameters(
                        risk_aversion=self.risk_aversion,
                        tau_for_covariance=self.tau_for_covariance,
                        tau_omega=self.tau_omega,
                        relative_confidence=self.relative_confidence,
                        market_weights=list(market_weights)
                    )
                    
                    bl_data = bl._load_data()
                    bl._compute_prior_from_market(bl_data)
                    bl._setup_views(P, Q, Omega_matrix=Omega, 
                                   relative_confidence=self.relative_confidence)
                    bl._compute_posterior()
                    
                    # Check for numerical issues
                    if np.any(np.isnan(bl.posterior_returns)) or np.any(np.isinf(bl.posterior_returns)):
                        raise RuntimeError("Posterior returns contain NaN or Inf values")
                    if np.any(np.isnan(bl.posterior_cov_matrix)) or np.any(np.isinf(bl.posterior_cov_matrix)):
                        raise RuntimeError("Posterior covariance contains NaN or Inf values")
                    
                except Exception as e:
                    logger.error(f"Error computing posterior: {e}")
                    raise RuntimeError(f"Failed to compute posterior: {e}") from e
                
                # 7. Run optimization with constraints
                try:
                    initial_weights = np.array([1/len(self.tickers)] * len(self.tickers))
                    opt_weights, sharpe, opt_info = self._optimize_with_constraints(bl, initial_weights)
                    
                    constraint_msg = "long/short" if self.allow_shorts else "long-only"
                    logger.info(f"\n✓ Optimization completed ({constraint_msg})")
                    logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
                    if not opt_info['converged']:
                        logger.warning(f"  Optimization warning: {opt_info['message']}")
                    
                except Exception as e:
                    logger.error(f"Error in optimization: {e}")
                    logger.warning("Using equal weights as fallback")
                    opt_weights = market_weights.copy()
                    sharpe = 0.0
                    opt_info = {'success': False, 'message': str(e), 'iterations': 0, 'converged': False}
        
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in optimization: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e
        
        # 8. Format results
        results = {
            'quarter': quarter,
            'weights': opt_weights.tolist() if isinstance(opt_weights, np.ndarray) else list(opt_weights),
            'sharpe': float(sharpe),
            'ticker_weights': {ticker: float(weight) 
                              for ticker, weight in zip(self.tickers, opt_weights)},
            'views_used': llm_views,
            'hyperparameters': {
                'risk_aversion': self.risk_aversion,
                'tau_for_covariance': self.tau_for_covariance,
                'tau_omega': self.tau_omega,
                'relative_confidence': self.relative_confidence
            },
            'optimization_info': opt_info,
            'constraints': {
                'allow_shorts': self.allow_shorts,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight
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
        logger.info(f"\n{'='*80}")
        logger.info(f"PORTFOLIO ALLOCATION - {results['quarter']}")
        logger.info(f"{'='*80}")
        
        logger.info(f"\nSharpe Ratio: {results['sharpe']:.4f}")
        
        # Show optimization status
        opt_info = results.get('optimization_info', {})
        if not opt_info.get('converged', True):
            logger.warning(f"  Optimization Status: {opt_info.get('message', 'Unknown')}")
        
        logger.info("")
        
        # Separate longs and shorts for clarity
        longs = {t: w for t, w in results['ticker_weights'].items() if w >= 0}
        shorts = {t: w for t, w in results['ticker_weights'].items() if w < 0}
        
        logger.info("Long Positions:")
        if longs:
            # Sort by weight descending
            sorted_longs = sorted(longs.items(), key=lambda x: x[1], reverse=True)
            for ticker, weight in sorted_longs:
                logger.info(f"  {ticker}: {weight:6.2%}")
        else:
            logger.info("  (none)")
        
        if shorts:
            logger.info("\nShort Positions:")
            # Sort by weight ascending (most negative first)
            sorted_shorts = sorted(shorts.items(), key=lambda x: x[1])
            for ticker, weight in sorted_shorts:
                logger.info(f"  {ticker}: {weight:6.2%}")
        
        total = sum(results['ticker_weights'].values())
        logger.info(f"\nNet Exposure: {total:6.2%}")
        
        if shorts:
            gross_exposure = sum(abs(w) for w in results['ticker_weights'].values())
            logger.info(f"Gross Exposure: {gross_exposure:6.2%}")
        
        # Show top holdings
        sorted_all = sorted(results['ticker_weights'].items(), 
                          key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"\nTop 3 Holdings (by absolute weight):")
        for ticker, weight in sorted_all[:3]:
            logger.info(f"  {ticker}: {weight:6.2%}")
    
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
        if 'weights' in json_results:
            if isinstance(json_results['weights'], np.ndarray):
                json_results['weights'] = json_results['weights'].tolist()
            elif not isinstance(json_results['weights'], list):
                json_results['weights'] = list(json_results['weights'])
        
        # Ensure all numpy types are converted
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            return obj
        
        json_results = convert_numpy_types(json_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\n✓ Saved portfolio to {output_path}")


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

