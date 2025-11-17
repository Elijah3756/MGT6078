"""
Test script for political news collection
"""

from political_news_collector import PoliticalNewsCollector

# Initialize the collector
collector = PoliticalNewsCollector()

# Example: Collect political news for Q1 2024
year = 2024
quarter = 1

print("Testing Political News Collector")
print("="*80)

# Collect news for a specific ticker
print("\nTesting single ticker search (XLK)...")
start_date, end_date = collector.get_quarter_dates(year, quarter)
xlk_articles = collector.search_political_news('XLK', start_date, end_date)
print(f"Found {len(xlk_articles)} articles for XLK")

if len(xlk_articles) > 0:
    print("\nFirst few articles:")
    print(xlk_articles[['title', 'url', 'seendate']].head())

# Collect news for all tickers in a quarter
print("\n\nCollecting news for all tickers...")
results = collector.collect_quarterly_news(year, quarter)

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for ticker, df in results.items():
    print(f"{ticker}: {len(df)} articles")