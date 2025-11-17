# Political News Collector for Black-Litterman Optimization

This module collects political news articles related to key ETF tickers using GDELT (Global Database of Events, Language, and Tone) for use in updating the views matrix for Black-Litterman portfolio optimization.

## Overview

The `PoliticalNewsCollector` class searches GDELT for political news articles related to 6 key tickers:
- **XLK** - Technology Select Sector SPDR Fund
- **XLY** - Consumer Discretionary Select Sector SPDR Fund
- **ITA** - iShares U.S. Aerospace & Defense ETF
- **XLE** - Energy Select Sector SPDR Fund
- **XLV** - Health Care Select Sector SPDR Fund
- **XLF** - Financial Select Sector SPDR Fund

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `gdeltdoc` - Python wrapper for GDELT API
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## Usage

### Basic Usage - Single Quarter

```python
from political_news_collector import PoliticalNewsCollector

# Initialize collector
collector = PoliticalNewsCollector()

# Collect news for Q1 2024
results = collector.collect_quarterly_news(year=2024, quarter=1)

# Save results
collector.save_results(results, year=2024, quarter=1)

# Format for LLM input
llm_text = collector.format_for_llm(results)
```

### Collect News for Specific Ticker

```python
# Get quarter dates
start_date, end_date = collector.get_quarter_dates(2024, 1)

# Search for specific ticker
xlk_articles = collector.search_political_news('XLK', start_date, end_date)
print(f"Found {len(xlk_articles)} articles")
```

### Collect Multiple Quarters

See `collect_multiple_quarters.py` for an example script that collects data for multiple quarters.

## Output Files

The collector saves data in multiple formats:

1. **CSV files**: `{TICKER}_Q{QUARTER}_{YEAR}_political_news.csv`
   - Contains all article metadata (title, URL, date, source, etc.)

2. **JSON files**: `{TICKER}_Q{QUARTER}_{YEAR}_political_news.json`
   - Same data in JSON format for programmatic access

3. **Summary file**: `Q{QUARTER}_{YEAR}_summary.txt`
   - Overview of articles collected per ticker

4. **LLM input file**: `Q{QUARTER}_{YEAR}_llm_input.txt`
   - Formatted text ready for LLM consumption to generate views

## Search Strategy

The collector uses a multi-keyword search approach:

1. For each ticker, searches using:
   - The ticker symbol itself
   - Related industry keywords
   - Combined with political terms (policy, politics, government, regulation, etc.)

2. Filters articles by:
   - Date range (quarterly)
   - Relevance to political/policy topics
   - Relevance to the ticker/industry

## Integration with Black-Litterman Pipeline

Based on the project whiteboard, the workflow is:

```
Political News (this module) → LLM (System Prompt) → Views Matrix (P, Q) → Black-Litterman Code → Trading Simulation
```

The collected articles should be fed to an LLM with a system prompt that:
1. Analyzes the political news
2. Generates views on expected returns for each ticker
3. Outputs the views matrix (P and Q) for the Black-Litterman model

## Quarterly Updates

According to the project timeline, data should be collected quarterly:
- 2024 Q3, Q4
- 2025 Q1, Q2
- Update range: March 24 - March 25 for each quarter

Use the `collect_multiple_quarters.py` script to automate quarterly collection.

## Notes

- GDELT API has rate limits - the collector includes error handling for API failures
- Some tickers may return fewer results depending on political news volume
- The search combines multiple keywords to maximize coverage
- Results are deduplicated by URL to avoid duplicate articles

## Troubleshooting

**No articles found for a ticker:**
- Try adjusting the date range
- Check if the ticker symbol is correct
- Verify GDELT API is accessible

**API errors:**
- GDELT API may have rate limits
- Try reducing `max_results` parameter
- Add delays between requests if needed

