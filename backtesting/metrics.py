"""
Performance Metrics
Calculate additional performance metrics for portfolio analysis
"""

import numpy as np
from typing import Dict, List, Optional

# Import config loader
try:
    from config_loader import get_config
except ImportError:
    # Fallback if config_loader not available
    def get_config():
        return {'performance': {'risk_free_rate': 0.02}}


class PerformanceMetrics:
    """Calculate performance metrics for portfolio"""
    
    @staticmethod
    def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate"""
        if years <= 0 or initial_value <= 0:
            return 0.0
        return (final_value / initial_value) ** (1 / years) - 1
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate (if None, uses config.yaml)
            
        Returns:
            Sortino ratio
        """
        # Get risk-free rate from config if not provided
        if risk_free_rate is None:
            config = get_config()
            risk_free_rate = config.get('performance', {}).get('risk_free_rate', 0.02)
        quarterly_rf = risk_free_rate / 4
        excess_returns = returns - quarterly_rf
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns) * np.sqrt(4)  # Annualized
        
        avg_excess_return = np.mean(excess_returns) * 4  # Annualized
        
        return avg_excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    @staticmethod
    def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)
        
        Args:
            annualized_return: Annualized return
            max_drawdown: Maximum drawdown (as negative number)
            
        Returns:
            Calmar ratio
        """
        if max_drawdown >= 0:
            return 0.0
        return annualized_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_rolling_sharpe(returns: np.ndarray, window: int = 4) -> List[float]:
        """
        Calculate rolling Sharpe ratio
        
        Args:
            returns: Array of quarterly returns
            window: Rolling window size (number of quarters)
            
        Returns:
            List of rolling Sharpe ratios
        """
        rolling_sharpe = []
        risk_free_rate = 0.02 / 4  # Quarterly risk-free rate
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            excess_returns = window_returns - risk_free_rate
            
            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) * np.sqrt(4) / (np.std(excess_returns) * np.sqrt(4))
                rolling_sharpe.append(sharpe)
            else:
                rolling_sharpe.append(0.0)
        
        return rolling_sharpe
    
    @staticmethod
    def calculate_information_ratio(strategy_returns: np.ndarray, 
                                    benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio (active return / tracking error)
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        active_returns = strategy_returns - benchmark_returns
        
        if len(active_returns) == 0:
            return 0.0
        
        mean_active_return = np.mean(active_returns) * 4  # Annualized
        tracking_error = np.std(active_returns) * np.sqrt(4)  # Annualized
        
        return mean_active_return / tracking_error if tracking_error > 0 else 0.0
    
    @staticmethod
    def calculate_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR at given confidence level
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            
        Returns:
            CVaR at given confidence level
        """
        if len(returns) == 0:
            return 0.0
        
        var = PerformanceMetrics.calculate_value_at_risk(returns, confidence)
        # Average of returns below VaR
        tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    @staticmethod
    def calculate_turnover(weights_history: List[Dict[str, float]]) -> float:
        """
        Calculate average portfolio turnover
        
        Args:
            weights_history: List of weight dictionaries over time
            
        Returns:
            Average turnover
        """
        if len(weights_history) <= 1:
            return 0.0
        
        turnovers = []
        tickers = list(weights_history[0].keys())
        
        for i in range(1, len(weights_history)):
            prev_weights = weights_history[i-1]
            curr_weights = weights_history[i]
            
            turnover = sum(abs(curr_weights.get(ticker, 0) - prev_weights.get(ticker, 0)) 
                          for ticker in tickers)
            turnovers.append(turnover / 2)  # Divide by 2 to avoid double counting
        
        return np.mean(turnovers)
    
    @staticmethod
    def calculate_all_metrics(backtest_results: Dict, 
                             benchmark_results: Dict = None) -> Dict:
        """
        Calculate comprehensive set of performance metrics
        
        Args:
            backtest_results: Backtest results dictionary
            benchmark_results: Optional benchmark results for comparison
            
        Returns:
            Dictionary of all metrics
        """
        summary = backtest_results['summary']
        returns = np.array(backtest_results['quarterly_returns'])
        history = backtest_results['history']
        
        # Basic metrics from summary
        metrics = {
            'total_return': summary['total_return'],
            'annualized_return': summary['annualized_return'],
            'volatility': summary['volatility'],
            'sharpe_ratio': summary['sharpe_ratio'],
            'max_drawdown': summary['max_drawdown'],
            'win_rate': summary['win_rate']
        }
        
        # Additional metrics
        metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(
            summary['annualized_return'], 
            summary['max_drawdown']
        )
        metrics['var_95'] = PerformanceMetrics.calculate_value_at_risk(returns, 0.95)
        metrics['cvar_95'] = PerformanceMetrics.calculate_conditional_var(returns, 0.95)
        
        # Turnover
        weights_history = [h['weights'] for h in history]
        metrics['avg_turnover'] = PerformanceMetrics.calculate_turnover(weights_history)
        
        # Benchmark comparison metrics
        if benchmark_results:
            benchmark_returns = np.array(benchmark_results['quarterly_returns'])
            metrics['information_ratio'] = PerformanceMetrics.calculate_information_ratio(
                returns, benchmark_returns
            )
            
            # Outperformance
            metrics['outperformance'] = (
                summary['annualized_return'] - 
                benchmark_results['summary']['annualized_return']
            )
        
        return metrics


def main():
    """Test metrics"""
    # Sample returns
    returns = np.array([0.05, -0.02, 0.03, 0.08, -0.01, 0.04, 0.02])
    
    print("Sample Performance Metrics:")
    print(f"Sortino Ratio: {PerformanceMetrics.calculate_sortino_ratio(returns):.4f}")
    print(f"VaR (95%): {PerformanceMetrics.calculate_value_at_risk(returns, 0.95):.4f}")
    print(f"CVaR (95%): {PerformanceMetrics.calculate_conditional_var(returns, 0.95):.4f}")


if __name__ == "__main__":
    main()

