"""
Data Formatter
Formats all data sources into structured format for LLM consumption
"""

import json
from typing import Dict


class DataFormatter:
    """Format quarterly data for LLM analysis"""
    
    def __init__(self, tickers=None):
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
        self.ticker_names = {
            'XLK': 'Technology Select Sector SPDR',
            'XLY': 'Consumer Discretionary Select Sector SPDR',
            'ITA': 'iShares U.S. Aerospace & Defense ETF',
            'XLE': 'Energy Select Sector SPDR',
            'XLV': 'Health Care Select Sector SPDR',
            'XLF': 'Financial Select Sector SPDR'
        }
    
    def format_for_llm(self, quarter_data: Dict) -> str:
        """
        Format quarterly data into text for LLM analysis
        
        Args:
            quarter_data: Dictionary with all quarterly data
            
        Returns:
            Formatted text string
        """
        output = []
        
        # Header
        output.append(f"INVESTMENT ANALYSIS DATA - {quarter_data['quarter']}")
        output.append("=" * 80)
        output.append(f"Date Range: {quarter_data['date_range']['start']} to {quarter_data['date_range']['end']}")
        output.append("\n")
        
        # FOMC Summary
        output.append("FEDERAL RESERVE POLICY (FOMC MINUTES SUMMARY)")
        output.append("-" * 80)
        fomc_preview = quarter_data['fomc_summary'][:2000]  # Limit for context
        output.append(fomc_preview)
        output.append("\n" + "=" * 80 + "\n")
        
        # Research Summary
        output.append("QUARTERLY INVESTMENT RESEARCH SUMMARY")
        output.append("-" * 80)
        research_preview = quarter_data['research_summary'][:2000]
        output.append(research_preview)
        output.append("\n" + "=" * 80 + "\n")
        
        # Ticker-specific data
        for ticker in self.tickers:
            ticker_info = quarter_data['tickers'].get(ticker, {})
            
            output.append(f"TICKER: {ticker} - {self.ticker_names[ticker]}")
            output.append("-" * 80)
            
            # Historical return
            hist_return = ticker_info.get('historical_return', 0.0)
            output.append(f"Previous Quarter Return: {hist_return:.2%}")
            output.append("")
            
            # Bloomberg headlines
            headlines = ticker_info.get('bloomberg_headlines', [])
            if headlines:
                output.append(f"Top Bloomberg Headlines ({len(headlines)} total):")
                for i, headline in enumerate(headlines[:15], 1):  # Limit to 15 headlines
                    output.append(f"  {i}. {headline}")
            else:
                output.append("No Bloomberg headlines available")
            
            output.append("\n" + "=" * 80 + "\n")
        
        # Political news (already formatted)
        output.append("POLITICAL NEWS ANALYSIS")
        output.append("-" * 80)
        political_preview = quarter_data['political_news'][:3000]  # Limit for context
        output.append(political_preview)
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)
    
    def save_formatted_data(self, quarter_data: Dict, output_path: str):
        """
        Save formatted data to file
        
        Args:
            quarter_data: Dictionary with quarterly data
            output_path: Path to save formatted text
        """
        formatted_text = self.format_for_llm(quarter_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        print(f"Saved formatted data to {output_path}")
    
    def save_json(self, quarter_data: Dict, output_path: str):
        """
        Save quarterly data as JSON
        
        Args:
            quarter_data: Dictionary with quarterly data
            output_path: Path to save JSON
        """
        # Create a serializable copy (remove DataFrame objects)
        json_data = {
            'quarter': quarter_data['quarter'],
            'date_range': quarter_data['date_range'],
            'fomc_summary': quarter_data['fomc_summary'][:5000],
            'research_summary': quarter_data['research_summary'][:5000],
            'political_news': quarter_data['political_news'][:5000],
            'tickers': {}
        }
        
        for ticker, info in quarter_data['tickers'].items():
            json_data['tickers'][ticker] = {
                'bloomberg_headlines': info.get('bloomberg_headlines', []),
                'historical_return': info.get('historical_return', 0.0)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved JSON data to {output_path}")


def main():
    """Test formatter"""
    from data.loader import DataLoader
    
    loader = DataLoader()
    formatter = DataFormatter()
    
    # Load Q1 2024 data
    data = loader.load_quarterly_data('Q1_2024')
    
    # Format for LLM
    formatted_text = formatter.format_for_llm(data)
    
    print("Formatted text length:", len(formatted_text))
    print("\nFirst 1000 characters:")
    print(formatted_text[:1000])
    
    # Save formatted data
    formatter.save_formatted_data(data, 'output/Q1_2024_formatted.txt')
    formatter.save_json(data, 'output/Q1_2024_data.json')


if __name__ == "__main__":
    main()

