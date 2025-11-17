"""
Data Loader Module
Loads all data sources: political news, Bloomberg headlines, FOMC PDFs, investor research, and stock data
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import PyPDF2
import re

# Import config loader
try:
    from config_loader import get_config, get_project_root
except ImportError:
    # Fallback if config_loader not available
    def get_project_root():
        return Path(__file__).resolve().parents[1]
    def get_config():
        return {'project_root': str(get_project_root())}


class DataLoader:
    """Unified data loader for all project data sources"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            base_path: Optional base path. If None, uses project root from config or derives from file location.
        """
        if base_path is None:
            config = get_config()
            self.base_path = config.get('project_root', str(get_project_root()))
        else:
            self.base_path = base_path
        
        # Ensure base_path is absolute
        self.base_path = str(Path(self.base_path).resolve())
        
        # Load tickers from config
        try:
            config = get_config()
            self.tickers = config.get('portfolio', {}).get('tickers', ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF'])
        except Exception:
            self.tickers = ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
        
        # Define quarter date ranges
        self.quarter_dates = {
            'Q1_2024': ('2024-01-01', '2024-03-31'),
            'Q2_2024': ('2024-04-01', '2024-06-30'),
            'Q3_2024': ('2024-07-01', '2024-09-30'),
            'Q4_2024': ('2024-10-01', '2024-12-31'),
            'Q1_2025': ('2025-01-01', '2025-03-31'),
            'Q2_2025': ('2025-04-01', '2025-06-30'),
            'Q3_2025': ('2025-07-01', '2025-09-30'),
        }
        
        # FOMC files mapping to quarters
        self.fomc_mapping = {
            'Q1_2024': ['jan24.pdf', 'march24.pdf'],
            'Q2_2024': ['may24.pdf', 'june24.pdf'],
            'Q3_2024': ['july24.pdf', 'sept24.pdf'],
            'Q4_2024': ['nov24.pdf', 'dec24.pdf'],
            'Q1_2025': ['jan25.pdf', 'march25.pdf'],
            'Q2_2025': ['may25.pdf', 'june25.pdf'],
            'Q3_2025': ['july25.pdf', 'august25.pdf', 'sept25.pdf', 'october25.pdf'],
        }
    
    def load_political_news(self, quarter: str) -> str:
        """
        Load pre-formatted political news for a quarter
        
        Args:
            quarter: Quarter string like 'Q1_2024'
            
        Returns:
            Formatted political news text
        """
        file_path = os.path.join(
            self.base_path, 
            'political_news_data', 
            f'{quarter}_llm_input.txt'
        )
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Political news file not found for {quarter}")
            return f"No political news data available for {quarter}"
    
    def load_bloomberg_headlines(self, ticker: str, quarter: str) -> List[str]:
        """
        Load Bloomberg headlines for a specific ticker and filter by quarter
        
        Args:
            ticker: ETF ticker symbol
            quarter: Quarter string like 'Q1_2024'
            
        Returns:
            List of headlines for the quarter
        """
        # Map ticker to file name
        file_mapping = {
            'XLK': 'XLK Data.txt',
            'XLY': 'XLY Data.txt',
            'ITA': 'ITA Data.txt',
            'XLE': 'XLE.txt',
            'XLV': 'XLV.txt',
            'XLF': 'XLF.txt'
        }
        
        file_path = os.path.join(
            self.base_path,
            'Finance and Investments Group Final Project',
            'Bloomberg Headlines',
            file_mapping[ticker]
        )
        
        if not os.path.exists(file_path):
            print(f"Warning: Bloomberg headlines file not found for {ticker}")
            return []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse headlines and dates
        headlines = []
        quarter_year = quarter.split('_')
        quarter_num = int(quarter_year[0][1])
        year = int(quarter_year[1])
        
        # Get month range for the quarter
        quarter_months = {
            1: ['01', '02', '03'],
            2: ['04', '05', '06'],
            3: ['07', '08', '09'],
            4: ['10', '11', '12']
        }
        
        target_months = quarter_months[quarter_num]
        
        for line in lines:
            # Extract date (format: MM/DD)
            date_match = re.search(r'\t(\d{2})/(\d{2})\t', line)
            if date_match:
                month = date_match.group(1)
                if month in target_months:
                    # Extract headline (text before first tab)
                    parts = line.split('\t')
                    if len(parts) > 1:
                        headline = parts[0].strip().lstrip('0123456789)').strip()
                        if headline:
                            headlines.append(headline)
        
        return headlines[:50]  # Limit to top 50 headlines
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def load_fomc_minutes(self, quarter: str) -> str:
        """
        Load and combine FOMC meeting minutes for a quarter
        
        Args:
            quarter: Quarter string like 'Q1_2024'
            
        Returns:
            Combined FOMC text
        """
        fomc_files = self.fomc_mapping.get(quarter, [])
        combined_text = []
        
        fomc_dir = os.path.join(
            self.base_path,
            'Finance and Investments Group Final Project',
            'FOMC data'
        )
        
        for filename in fomc_files:
            file_path = os.path.join(fomc_dir, filename)
            if os.path.exists(file_path):
                text = self.extract_pdf_text(file_path)
                # Summarize key sections (first 3000 chars as summary)
                if text:
                    combined_text.append(f"--- {filename} ---\n{text[:3000]}")
            else:
                print(f"Warning: FOMC file not found: {filename}")
        
        return "\n\n".join(combined_text) if combined_text else "No FOMC data available"
    
    def load_investor_research(self, quarter: str) -> str:
        """
        Load quarterly investor research report
        
        Args:
            quarter: Quarter string like 'Q1_2024'
            
        Returns:
            Research report text
        """
        # Map quarter to research file
        file_mapping = {
            'Q1_2024': 'Investment-Research-Update-Q1-2024.pdf',
            'Q2_2024': 'Investment-Research-Update-Q2-2024.pdf',
            'Q3_2024': 'Investment-Research-Update-Q3-2024.pdf',
            'Q4_2024': 'Investment-Research-Update-Q4-2024.pdf',
            'Q1_2025': 'Quarterly-Sector-and-Investment-Research-Update-Q1-2025.pdf',
            'Q2_2025': 'Quarterly-Sector-and-Investment-Research-Update-Q2-2025.pdf',
            'Q3_2025': 'Quarterly-Sector-and-Investment-Research-Update-Q3-2025.pdf',
        }
        
        file_path = os.path.join(
            self.base_path,
            'Finance and Investments Group Final Project',
            'Investor Research',
            file_mapping.get(quarter, '')
        )
        
        if os.path.exists(file_path):
            text = self.extract_pdf_text(file_path)
            # Return first 5000 characters as summary
            return text[:5000] if text else "No research data extracted"
        else:
            print(f"Warning: Investor research file not found for {quarter}")
            return "No investor research available"
    
    def load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical stock data for all 6 ETFs
        
        Returns:
            Dictionary mapping ticker to DataFrame with price data
        """
        stock_file = os.path.join(
            self.base_path,
            'Finance and Investments Group Final Project',
            'Stock Data.xlsx'
        )
        
        stock_data = {}
        for ticker in self.tickers:
            try:
                df = pd.read_excel(stock_file, sheet_name=ticker)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                stock_data[ticker] = df
            except Exception as e:
                print(f"Error loading stock data for {ticker}: {e}")
        
        return stock_data
    
    def calculate_quarterly_returns(self, stock_data: Dict[str, pd.DataFrame], 
                                   quarter: str) -> Dict[str, float]:
        """
        Calculate quarterly returns for each ETF
        
        Args:
            stock_data: Dictionary of stock DataFrames
            quarter: Quarter string
            
        Returns:
            Dictionary mapping ticker to quarterly return
        """
        start_date, end_date = self.quarter_dates[quarter]
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        returns = {}
        for ticker, df in stock_data.items():
            # Filter data for the quarter
            quarter_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if len(quarter_df) > 0:
                start_price = quarter_df.iloc[0]['Close']
                end_price = quarter_df.iloc[-1]['Close']
                returns[ticker] = (end_price - start_price) / start_price
            else:
                print(f"Warning: No data for {ticker} in {quarter}")
                returns[ticker] = 0.0
        
        return returns
    
    def load_quarterly_data(self, quarter: str) -> Dict:
        """
        Load all data sources for a specific quarter
        
        Args:
            quarter: Quarter string like 'Q1_2024'
            
        Returns:
            Dictionary containing all data for the quarter
        """
        print(f"\n{'='*80}")
        print(f"Loading data for {quarter}")
        print(f"{'='*80}")
        
        # Load stock data first
        stock_data = self.load_stock_data()
        quarterly_returns = self.calculate_quarterly_returns(stock_data, quarter)
        
        # Load political news (already aggregated)
        political_news = self.load_political_news(quarter)
        
        # Load FOMC minutes
        fomc_text = self.load_fomc_minutes(quarter)
        
        # Load investor research
        research_text = self.load_investor_research(quarter)
        
        # Load Bloomberg headlines for each ticker
        ticker_data = {}
        for ticker in self.tickers:
            headlines = self.load_bloomberg_headlines(ticker, quarter)
            ticker_data[ticker] = {
                'bloomberg_headlines': headlines,
                'historical_return': quarterly_returns.get(ticker, 0.0)
            }
        
        # Combine all data
        quarter_data = {
            'quarter': quarter,
            'date_range': {
                'start': self.quarter_dates[quarter][0],
                'end': self.quarter_dates[quarter][1]
            },
            'political_news': political_news,
            'fomc_summary': fomc_text,
            'research_summary': research_text,
            'tickers': ticker_data,
            'stock_data': stock_data
        }
        
        print(f"\n✓ Loaded political news: {len(political_news)} characters")
        print(f"✓ Loaded FOMC data: {len(fomc_text)} characters")
        print(f"✓ Loaded research: {len(research_text)} characters")
        print(f"✓ Loaded Bloomberg headlines for {len(ticker_data)} tickers")
        print(f"✓ Calculated quarterly returns")
        
        return quarter_data


def main():
    """Test the data loader"""
    loader = DataLoader()
    
    # Test loading Q1 2024 data
    data = loader.load_quarterly_data('Q1_2024')
    
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(f"\nQuarter: {data['quarter']}")
    print(f"Date Range: {data['date_range']}")
    print(f"\nXLK Headlines (first 3):")
    for i, headline in enumerate(data['tickers']['XLK']['bloomberg_headlines'][:3]):
        print(f"  {i+1}. {headline}")
    print(f"\nQuarterly Returns:")
    for ticker, info in data['tickers'].items():
        print(f"  {ticker}: {info['historical_return']:.2%}")


if __name__ == "__main__":
    main()

