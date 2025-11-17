"""
Stock Data Processor
Processes stock price data to calculate returns for Black-Litterman model
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import os


class StockProcessor:
    """Process stock data for Black-Litterman optimization"""
    
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
    
    def calculate_returns(self, stock_data: Dict[str, pd.DataFrame], 
                         lookback_days: int = 252) -> pd.DataFrame:
        """
        Calculate daily returns for all ETFs
        
        Args:
            stock_data: Dictionary mapping ticker to price DataFrame
            lookback_days: Number of days to use for return calculation
            
        Returns:
            DataFrame with returns for each ticker
        """
        returns_dict = {}
        
        for ticker in self.tickers:
            if ticker in stock_data:
                df = stock_data[ticker].copy()
                df = df.sort_values('Date')
                
                # Calculate daily returns
                df['Return'] = df['Close'].pct_change()
                
                # Take last N days
                df = df.tail(lookback_days)
                
                returns_dict[ticker] = df['Return'].values
        
        # Create DataFrame with aligned returns
        min_length = min(len(v) for v in returns_dict.values())
        
        aligned_returns = {}
        for ticker, returns in returns_dict.items():
            aligned_returns[ticker] = returns[-min_length:]
        
        returns_df = pd.DataFrame(aligned_returns)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def save_returns_csv(self, returns_df: pd.DataFrame, output_path: str):
        """
        Save returns data as CSV for Black-Litterman model
        
        Args:
            returns_df: DataFrame with returns
            output_path: Path to save CSV file
        """
        returns_df.to_csv(output_path, index=False)
        print(f"Saved returns data to {output_path}")
    
    def get_covariance_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix from returns
        
        Args:
            returns_df: DataFrame with returns
            
        Returns:
            Covariance matrix
        """
        return returns_df.cov()
    
    def get_market_weights(self, method: str = 'equal') -> np.ndarray:
        """
        Get market weights for the portfolio
        
        Args:
            method: 'equal' for equal weights, 'market_cap' for market cap weights
            
        Returns:
            Array of weights
        """
        if method == 'equal':
            return np.array([1/len(self.tickers)] * len(self.tickers))
        elif method == 'market_cap':
            # For now, use approximate market cap weights
            # These are rough approximations - can be updated with actual market cap data
            weights_dict = {
                'XLK': 0.30,  # Technology - largest
                'XLV': 0.20,  # Healthcare
                'XLF': 0.15,  # Financials
                'XLY': 0.15,  # Consumer Discretionary
                'XLE': 0.10,  # Energy
                'ITA': 0.10,  # Aerospace & Defense - smallest
            }
            return np.array([weights_dict[ticker] for ticker in self.tickers])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def prepare_bl_data(self, stock_data: Dict[str, pd.DataFrame], 
                       lookback_days: int = 252) -> tuple:
        """
        Prepare all data needed for Black-Litterman
        
        Args:
            stock_data: Dictionary of stock DataFrames
            lookback_days: Days of historical data to use
            
        Returns:
            Tuple of (returns_df, covariance_matrix, market_weights)
        """
        returns_df = self.calculate_returns(stock_data, lookback_days)
        cov_matrix = self.get_covariance_matrix(returns_df)
        market_weights = self.get_market_weights(method='equal')
        
        return returns_df, cov_matrix, market_weights
    
    def get_quarterly_returns_array(self, stock_data: Dict[str, pd.DataFrame],
                                   quarter_dates: tuple) -> np.ndarray:
        """
        Get returns for a specific quarter for backtesting
        
        Args:
            stock_data: Dictionary of stock DataFrames
            quarter_dates: Tuple of (start_date, end_date)
            
        Returns:
            Array of quarterly returns for each ticker
        """
        start_date = pd.to_datetime(quarter_dates[0])
        end_date = pd.to_datetime(quarter_dates[1])
        
        returns = []
        for ticker in self.tickers:
            if ticker in stock_data:
                df = stock_data[ticker]
                quarter_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if len(quarter_df) > 0:
                    start_price = quarter_df.iloc[0]['Close']
                    end_price = quarter_df.iloc[-1]['Close']
                    ret = (end_price - start_price) / start_price
                    returns.append(ret)
                else:
                    returns.append(0.0)
            else:
                returns.append(0.0)
        
        return np.array(returns)


def main():
    """Test stock processor"""
    from data.loader import DataLoader
    
    loader = DataLoader()
    stock_data = loader.load_stock_data()
    
    processor = StockProcessor()
    returns_df, cov_matrix, market_weights = processor.prepare_bl_data(stock_data)
    
    print("Returns DataFrame shape:", returns_df.shape)
    print("\nCovariance Matrix:")
    print(cov_matrix)
    print("\nMarket Weights:")
    print(market_weights)
    
    # Test saving
    processor.save_returns_csv(returns_df, 'output/returns_test.csv')


if __name__ == "__main__":
    main()

