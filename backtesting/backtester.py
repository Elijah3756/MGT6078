"""
Backtester
Simulate portfolio performance with quarterly rebalancing
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime


class Backtester:
    """Backtest portfolio strategy with quarterly rebalancing"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.tickers = ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
    
    def get_quarterly_returns(self, stock_data: Dict[str, pd.DataFrame],
                              quarter_dates: tuple) -> Dict[str, float]:
        """
        Calculate returns for each ticker in a quarter
        
        Args:
            stock_data: Dictionary of stock DataFrames
            quarter_dates: Tuple of (start_date, end_date)
            
        Returns:
            Dictionary mapping ticker to return
        """
        start_date = pd.to_datetime(quarter_dates[0])
        end_date = pd.to_datetime(quarter_dates[1])
        
        returns = {}
        for ticker in self.tickers:
            if ticker in stock_data:
                df = stock_data[ticker]
                quarter_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if len(quarter_df) > 0:
                    start_price = quarter_df.iloc[0]['Close']
                    end_price = quarter_df.iloc[-1]['Close']
                    returns[ticker] = (end_price - start_price) / start_price
                else:
                    print(f"Warning: No data for {ticker} in {quarter_dates}")
                    returns[ticker] = 0.0
            else:
                returns[ticker] = 0.0
        
        return returns
    
    def calculate_portfolio_return(self, weights: Dict[str, float], 
                                   returns: Dict[str, float]) -> float:
        """
        Calculate portfolio return for a period
        
        Args:
            weights: Dictionary of ticker weights
            returns: Dictionary of ticker returns
            
        Returns:
            Portfolio return
        """
        portfolio_return = 0.0
        for ticker in self.tickers:
            portfolio_return += weights.get(ticker, 0.0) * returns.get(ticker, 0.0)
        
        return portfolio_return
    
    def run_backtest(self, portfolio_results: List[Dict], 
                    stock_data: Dict[str, pd.DataFrame],
                    quarter_dates_map: Dict[str, tuple]) -> Dict:
        """
        Run backtest simulation
        
        Args:
            portfolio_results: List of quarterly portfolio optimization results
            stock_data: Dictionary of stock price DataFrames
            quarter_dates_map: Mapping of quarter to date ranges
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST")
        print(f"{'='*80}")
        
        portfolio_value = self.initial_capital
        history = []
        
        for i, result in enumerate(portfolio_results):
            quarter = result['quarter']
            weights = result['ticker_weights']
            
            print(f"\nQuarter {i+1}: {quarter}")
            print(f"  Starting value: ${portfolio_value:,.2f}")
            
            # Get actual returns for this quarter
            quarter_dates = quarter_dates_map[quarter]
            returns = self.get_quarterly_returns(stock_data, quarter_dates)
            
            # Calculate portfolio return
            portfolio_return = self.calculate_portfolio_return(weights, returns)
            
            # Update portfolio value
            new_value = portfolio_value * (1 + portfolio_return)
            
            # Record history
            history_entry = {
                'quarter': quarter,
                'quarter_num': i + 1,
                'start_value': portfolio_value,
                'end_value': new_value,
                'return': portfolio_return,
                'weights': weights,
                'ticker_returns': returns
            }
            history.append(history_entry)
            
            print(f"  Quarterly return: {portfolio_return:+.2%}")
            print(f"  Ending value: ${new_value:,.2f}")
            
            portfolio_value = new_value
        
        # Calculate summary statistics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        n_quarters = len(history)
        
        # Annualized return (compound)
        years = n_quarters / 4
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate quarterly returns array for stats
        quarterly_returns = np.array([h['return'] for h in history])
        
        # Volatility (annualized)
        volatility = np.std(quarterly_returns) * np.sqrt(4)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        values = [self.initial_capital] + [h['end_value'] for h in history]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        wins = sum(1 for r in quarterly_returns if r > 0)
        win_rate = wins / n_quarters if n_quarters > 0 else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_quarters': n_quarters,
            'best_quarter': max(quarterly_returns),
            'worst_quarter': min(quarterly_returns),
            'avg_quarterly_return': np.mean(quarterly_returns)
        }
        
        results = {
            'history': history,
            'summary': summary,
            'quarterly_returns': quarterly_returns.tolist()
        }
        
        self.print_summary(summary)
        
        return results
    
    def run_benchmark(self, benchmark_weights: Dict[str, float],
                     stock_data: Dict[str, pd.DataFrame],
                     quarter_dates_map: Dict[str, tuple],
                     quarters: List[str]) -> Dict:
        """
        Run benchmark strategy (e.g., equal weight buy and hold)
        
        Args:
            benchmark_weights: Fixed weights for benchmark
            stock_data: Stock price data
            quarter_dates_map: Quarter date mappings
            quarters: List of quarters to simulate
            
        Returns:
            Benchmark backtest results
        """
        print(f"\n{'='*80}")
        print("RUNNING BENCHMARK (Equal Weight)")
        print(f"{'='*80}")
        
        portfolio_value = self.initial_capital
        history = []
        
        for i, quarter in enumerate(quarters):
            quarter_dates = quarter_dates_map[quarter]
            returns = self.get_quarterly_returns(stock_data, quarter_dates)
            
            portfolio_return = self.calculate_portfolio_return(benchmark_weights, returns)
            new_value = portfolio_value * (1 + portfolio_return)
            
            history.append({
                'quarter': quarter,
                'quarter_num': i + 1,
                'start_value': portfolio_value,
                'end_value': new_value,
                'return': portfolio_return,
                'weights': benchmark_weights,
                'ticker_returns': returns
            })
            
            portfolio_value = new_value
        
        # Calculate summary
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        n_quarters = len(history)
        years = n_quarters / 4
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        quarterly_returns = np.array([h['return'] for h in history])
        volatility = np.std(quarterly_returns) * np.sqrt(4)
        
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        values = [self.initial_capital] + [h['end_value'] for h in history]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        wins = sum(1 for r in quarterly_returns if r > 0)
        win_rate = wins / n_quarters if n_quarters > 0 else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_quarters': n_quarters,
            'best_quarter': max(quarterly_returns),
            'worst_quarter': min(quarterly_returns),
            'avg_quarterly_return': np.mean(quarterly_returns)
        }
        
        self.print_summary(summary)
        
        return {
            'history': history,
            'summary': summary,
            'quarterly_returns': quarterly_returns.tolist()
        }
    
    def print_summary(self, summary: Dict):
        """
        Print backtest summary statistics
        
        Args:
            summary: Summary statistics dictionary
        """
        print(f"\n{'='*80}")
        print("BACKTEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital:      ${summary['initial_capital']:>12,.2f}")
        print(f"  Final Value:          ${summary['final_value']:>12,.2f}")
        print(f"  Total Return:         {summary['total_return']:>12.2%}")
        print(f"  Annualized Return:    {summary['annualized_return']:>12.2%}")
        
        print(f"\nRisk Metrics:")
        print(f"  Volatility (annual):  {summary['volatility']:>12.2%}")
        print(f"  Sharpe Ratio:         {summary['sharpe_ratio']:>12.4f}")
        print(f"  Maximum Drawdown:     {summary['max_drawdown']:>12.2%}")
        
        print(f"\nQuarterly Statistics:")
        print(f"  Number of Quarters:   {summary['n_quarters']:>12}")
        print(f"  Win Rate:             {summary['win_rate']:>12.2%}")
        print(f"  Best Quarter:         {summary['best_quarter']:>12.2%}")
        print(f"  Worst Quarter:        {summary['worst_quarter']:>12.2%}")
        print(f"  Avg Quarterly Return: {summary['avg_quarterly_return']:>12.2%}")


def main():
    """Test backtester"""
    # This will be tested from the main pipeline
    pass


if __name__ == "__main__":
    main()

