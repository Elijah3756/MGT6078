"""
Political News Collector for Black-Litterman Optimization
Collects political news articles related to key ETF tickers using GDELT
"""

from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re


class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass


class PoliticalNewsCollector:
    """
    Collects political news articles for specified tickers using GDELT
    """
    
    def __init__(self):
        self.gd = GdeltDoc()
        # Map tickers to their full names and general topic keywords for broader political news search
        self.ticker_info = {
            'XLK': {
                'name': 'Technology Select Sector SPDR Fund',
                'topic_keywords': ['technology', 'tech industry', 'tech sector', 'silicon valley', 'big tech'],
                'industry': 'Technology'
            },
            'XLY': {
                'name': 'Consumer Discretionary Select Sector SPDR Fund',
                'topic_keywords': ['retail', 'consumer spending', 'consumer economy', 'retail industry', 'consumer sector'],
                'industry': 'Consumer Discretionary'
            },
            'ITA': {
                'name': 'iShares U.S. Aerospace & Defense ETF',
                'topic_keywords': ['defense', 'aerospace', 'military', 'defense industry', 'defense contractors'],
                'industry': 'Aerospace & Defense'
            },
            'XLE': {
                'name': 'Energy Select Sector SPDR Fund',
                'topic_keywords': ['energy', 'oil', 'gas', 'energy industry', 'fossil fuels', 'renewable energy'],
                'industry': 'Energy'
            },
            'XLV': {
                'name': 'Health Care Select Sector SPDR Fund',
                'topic_keywords': ['healthcare', 'health care', 'pharmaceuticals', 'pharma', 'medical', 'biotech'],
                'industry': 'Healthcare'
            },
            'XLF': {
                'name': 'Financial Select Sector SPDR Fund',
                'topic_keywords': ['banking', 'finance', 'financial sector', 'banks', 'wall street', 'financial industry'],
                'industry': 'Financial'
            }
        }
    
    def get_quarter_dates(self, year: int, quarter: int) -> tuple:
        """
        Get start and end dates for a given quarter
        
        Args:
            year: Year (e.g., 2024)
            quarter: Quarter number (1, 2, 3, or 4)
        
        Returns:
            Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
        """
        if quarter == 1:
            start_date = f"{year}-01-01"
            end_date = f"{year}-03-31"
        elif quarter == 2:
            start_date = f"{year}-04-01"
            end_date = f"{year}-06-30"
        elif quarter == 3:
            start_date = f"{year}-07-01"
            end_date = f"{year}-09-30"
        elif quarter == 4:
            start_date = f"{year}-10-01"
            end_date = f"{year}-12-31"
        else:
            raise ValueError("Quarter must be 1, 2, 3, or 4")
        
        return start_date, end_date
    
    def search_political_news(self, 
                             ticker: str, 
                             start_date: str, 
                             end_date: str,
                             max_results: int = 250) -> pd.DataFrame:
        """
        Search for top political news articles related to a ticker's general topic
        
        Uses broader searches focused on general industry topics with political context,
        rather than restrictive keyword combinations.
        
        Args:
            ticker: ETF ticker symbol (e.g., 'XLK')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_results: Maximum number of results to return per search
        
        Returns:
            DataFrame containing article information
        """
        if ticker not in self.ticker_info:
            raise ValueError(f"Unknown ticker: {ticker}")
        
        ticker_data = self.ticker_info[ticker]
        topic_keywords = ticker_data['topic_keywords']
        
        all_articles = []
        
        # Search for top political news on the general topic of each ETF
        # Use broader searches that focus on the industry topic with political context
        # This approach is less restrictive and captures more relevant political news
        
        # Primary search: main industry topic with political context
        # Use the first 2-3 most relevant topic keywords
        primary_keywords = topic_keywords[:3]
        
        for keyword in primary_keywords:
            try:
                # Search for articles about the topic that are likely political
                # GDELT will return articles that mention the keyword in political context
                f = Filters(
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
                    num_records=max_results
                )
                
                articles = self.gd.article_search(f)
                
                if articles is not None and len(articles) > 0:
                    # Filter for English language only
                    articles = articles[articles['language'] == 'English'].copy()
                    
                    if len(articles) > 0:
                        # GDELT already focuses on political/news events, so articles returned
                        # are likely to have political relevance. We'll keep all articles and
                        # prioritize by date to get top news.
                        
                        # Add ticker and industry info to each article
                        articles['ticker'] = ticker
                        articles['industry'] = ticker_data['industry']
                        articles['search_keyword'] = keyword
                        all_articles.append(articles)
                        print(f"    Found {len(articles)} articles for topic: '{keyword}'")
                    else:
                        print(f"    No English articles found for topic: '{keyword}'")
                else:
                    print(f"    No articles found for topic: '{keyword}'")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                    
            except RateLimitError:
                # Re-raise rate limit errors so they can be handled at a higher level
                raise
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit indicators
                rate_limit_indicators = [
                    'rate limit',
                    'too many requests',
                    '429',
                    'quota exceeded',
                    'quota limit',
                    'throttle',
                    'throttled',
                    'rate exceeded',
                    'too many',
                    'limit exceeded'
                ]
                
                if any(indicator in error_str for indicator in rate_limit_indicators):
                    print(f"\n{'!'*80}")
                    print(f"RATE LIMIT DETECTED for '{keyword}' (Ticker: {ticker})")
                    print(f"   Error: {str(e)}")
                    print(f"{'!'*80}\n")
                    raise RateLimitError(f"Rate limit hit while searching '{keyword}' for {ticker}: {str(e)}")
                
                # Log other errors but continue
                print(f"    Error searching '{keyword}': {str(e)}")
                continue
        
        if not all_articles:
            print(f"  No articles found for {ticker}. Trying broader search...")
            # Fallback: try searching with just the main industry term
            try:
                main_topic = topic_keywords[0]
                f = Filters(
                    keyword=main_topic,
                    start_date=start_date,
                    end_date=end_date,
                    num_records=max_results * 2  # Get more results for broader search
                )
                articles = self.gd.article_search(f)
                if articles is not None and len(articles) > 0:
                    articles = articles[articles['language'] == 'English'].copy()
                    if len(articles) > 0:
                        articles['ticker'] = ticker
                        articles['industry'] = ticker_data['industry']
                        articles['search_keyword'] = main_topic
                        all_articles.append(articles)
                        print(f"    Found {len(articles)} articles with broader search: '{main_topic}'")
            except RateLimitError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                rate_limit_indicators = [
                    'rate limit', 'too many requests', '429', 'quota exceeded',
                    'quota limit', 'throttle', 'throttled', 'rate exceeded'
                ]
                if any(indicator in error_str for indicator in rate_limit_indicators):
                    print(f"\n{'!'*80}")
                    print(f"RATE LIMIT DETECTED in broader search for {ticker}")
                    print(f"   Error: {str(e)}")
                    print(f"{'!'*80}\n")
                    raise RateLimitError(f"Rate limit hit in broader search for {ticker}: {str(e)}")
                print(f"    Error with broader search: {str(e)}")
        
        if not all_articles:
            return pd.DataFrame()
        
        # Combine all results and remove duplicates based on URL
        combined_df = pd.concat(all_articles, ignore_index=True)
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
        after_dedup_count = len(combined_df)
        
        if after_dedup_count < initial_count:
            print(f"  Removed {initial_count - after_dedup_count} duplicate articles")
        
        # Sort by date (most recent first) to prioritize top news
        if 'seendate' in combined_df.columns:
            combined_df = combined_df.sort_values('seendate', ascending=False)
            # Limit to top articles if we have too many
            if len(combined_df) > max_results * 2:
                combined_df = combined_df.head(max_results * 2)
                print(f"  Limited to top {len(combined_df)} articles by date")
        
        # Fetch article content for each article
        print(f"  Fetching article content for {len(combined_df)} articles...")
        combined_df['article_text'] = combined_df['url'].apply(self._fetch_article_content)
        
        # Count articles with content before filtering
        articles_with_content = combined_df[combined_df['article_text'].notna() & (combined_df['article_text'] != '')]
        articles_without_content = len(combined_df) - len(articles_with_content)
        
        if articles_without_content > 0:
            print(f"  Warning: Could not fetch content for {articles_without_content} articles (keeping metadata only)")
        
        # Remove articles where we couldn't fetch content
        combined_df = combined_df[combined_df['article_text'].notna() & (combined_df['article_text'] != '')]
        
        return combined_df
    
    def _fetch_article_content(self, url: str, timeout: int = 10) -> str:
        """
        Fetch the full text content of an article from its URL
        
        Args:
            url: Article URL
            timeout: Request timeout in seconds
        
        Returns:
            Article text content or empty string if fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find article content in common tags
            article_content = None
            
            # Common article content selectors
            selectors = [
                'article',
                '[role="article"]',
                '.article-content',
                '.article-body',
                '.post-content',
                '.entry-content',
                'main',
                '.content',
                '#content'
            ]
            
            for selector in selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    break
            
            # If no specific article tag found, try to get main content
            if not article_content:
                article_content = soup.find('main') or soup.find('body')
            
            if article_content:
                # Get text and clean it up
                text = article_content.get_text(separator=' ', strip=True)
                # Remove excessive whitespace
                text = re.sub(r'\s+', ' ', text)
                # Limit length to avoid extremely long articles
                if len(text) > 50000:
                    text = text[:50000] + "... [truncated]"
                return text.strip()
            else:
                return ""
                
        except Exception as e:
            # Silently fail and return empty string
            return ""
    
    def collect_quarterly_news(self, 
                               year: int, 
                               quarter: int,
                               tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect political news for all tickers for a specific quarter
        
        Args:
            year: Year (e.g., 2024)
            quarter: Quarter number (1, 2, 3, or 4)
            tickers: List of tickers to search. If None, searches all 6 tickers
        
        Returns:
            Dictionary mapping ticker to DataFrame of articles
        """
        if tickers is None:
            tickers = list(self.ticker_info.keys())
        
        start_date, end_date = self.get_quarter_dates(year, quarter)
        
        print(f"\n{'='*80}")
        print(f"Collecting political news for Q{quarter} {year}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"{'='*80}\n")
        
        results = {}
        
        for ticker in tickers:
            print(f"Searching for {ticker} ({self.ticker_info[ticker]['name']})...")
            try:
                articles_df = self.search_political_news(ticker, start_date, end_date)
                
                if len(articles_df) > 0:
                    print(f"  Found {len(articles_df)} articles")
                    results[ticker] = articles_df
                else:
                    print(f"  No articles found")
                    results[ticker] = pd.DataFrame()
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[ticker] = pd.DataFrame()
        
        return results
    
    def save_results(self, 
                    results: Dict[str, pd.DataFrame], 
                    year: int, 
                    quarter: int,
                    output_dir: str = "political_news_data") -> Dict[str, str]:
        """
        Save collected articles to files
        
        Args:
            results: Dictionary mapping ticker to DataFrame
            year: Year
            quarter: Quarter number
            output_dir: Directory to save files
        
        Returns:
            Dictionary mapping ticker to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for ticker, df in results.items():
            if len(df) == 0:
                continue
            
            # Save as CSV
            csv_filename = f"{output_dir}/{ticker}_Q{quarter}_{year}_political_news.csv"
            df.to_csv(csv_filename, index=False)
            saved_files[ticker] = csv_filename
            
            # Save as JSON for LLM consumption
            json_filename = f"{output_dir}/{ticker}_Q{quarter}_{year}_political_news.json"
            articles_json = df.to_dict('records')
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(articles_json, f, indent=2, ensure_ascii=False)
        
        # Create summary file
        summary_filename = f"{output_dir}/Q{quarter}_{year}_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"Political News Collection Summary - Q{quarter} {year}\n")
            f.write("="*80 + "\n\n")
            for ticker, df in results.items():
                f.write(f"{ticker} ({self.ticker_info[ticker]['name']}): {len(df)} articles\n")
        
        print(f"\nResults saved to {output_dir}/")
        print(f"Summary: {summary_filename}")
        
        return saved_files
    
    def format_for_llm(self, 
                      results: Dict[str, pd.DataFrame],
                      max_articles_per_ticker: int = 50) -> str:
        """
        Format collected articles into a text format suitable for LLM input
        
        Args:
            results: Dictionary mapping ticker to DataFrame
            max_articles_per_ticker: Maximum articles to include per ticker
        
        Returns:
            Formatted string ready for LLM consumption
        """
        llm_text = []
        llm_text.append("POLITICAL NEWS ARTICLES FOR BLACK-LITTERMAN VIEWS MATRIX\n")
        llm_text.append("="*80 + "\n\n")
        
        for ticker, df in results.items():
            if len(df) == 0:
                continue
            
            ticker_data = self.ticker_info[ticker]
            llm_text.append(f"TICKER: {ticker} - {ticker_data['name']}")
            llm_text.append(f"INDUSTRY: {ticker_data['industry']}")
            llm_text.append("-"*80 + "\n")
            
            # Limit articles and sort by date (most recent first)
            df_sorted = df.head(max_articles_per_ticker).copy()
            if 'seendate' in df_sorted.columns:
                df_sorted = df_sorted.sort_values('seendate', ascending=False)
            
            for idx, row in df_sorted.iterrows():
                llm_text.append(f"\nArticle {idx + 1}:")
                if 'title' in row:
                    llm_text.append(f"Title: {row['title']}")
                if 'seendate' in row:
                    llm_text.append(f"Date: {row['seendate']}")
                if 'url' in row:
                    llm_text.append(f"URL: {row['url']}")
                if 'domain' in row:
                    llm_text.append(f"Source: {row['domain']}")
                if 'article_text' in row and pd.notna(row['article_text']) and row['article_text']:
                    # Truncate very long articles for readability
                    article_content = row['article_text']
                    if len(article_content) > 10000:
                        article_content = article_content[:10000] + "... [truncated]"
                    llm_text.append(f"\nArticle Content:\n{article_content}")
                llm_text.append("\n" + "-"*80 + "\n")
            
            llm_text.append("\n" + "="*80 + "\n\n")
        
        return "\n".join(llm_text)


def main():
    """
    Example usage of the PoliticalNewsCollector
    """
    collector = PoliticalNewsCollector()
    
    # Example: Collect news for Q1 2024
    year = 2024
    quarter = 1
    
    print("Political News Collector for Black-Litterman Optimization")
    print("="*80)
    
    # Collect news for all tickers
    results = collector.collect_quarterly_news(year, quarter)
    
    # Save results
    saved_files = collector.save_results(results, year, quarter)
    
    # Format for LLM
    llm_text = collector.format_for_llm(results)
    
    # Save LLM-formatted text
    llm_filename = f"political_news_data/Q{quarter}_{year}_llm_input.txt"
    os.makedirs("political_news_data", exist_ok=True)
    with open(llm_filename, 'w', encoding='utf-8') as f:
        f.write(llm_text)
    
    print(f"\nLLM-formatted text saved to: {llm_filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    for ticker, df in results.items():
        print(f"{ticker}: {len(df)} articles collected")
    
    return results


if __name__ == "__main__":
    results = main()

