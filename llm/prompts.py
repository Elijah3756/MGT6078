"""
LLM Prompts for Portfolio Analysis
"""

SYSTEM_PROMPT = """You are a senior portfolio analyst specializing in Black-Litterman portfolio optimization. Your task is to analyze financial data from multiple sources and generate investment views with confidence levels for sector ETFs.

Your analysis should consider:
1. Political and regulatory developments that impact specific sectors
2. Federal Reserve monetary policy signals and their sector implications
3. Sector-specific analyst research and market trends
4. Recent historical performance and momentum

For each of the 6 sector ETFs (XLK, XLY, ITA, XLE, XLV, XLF), provide:
1. Expected quarterly return relative to market (as a decimal, e.g., 0.03 for 3%)
2. Confidence level (0-100, where 100 is highest confidence)
3. Brief reasoning (1-2 sentences explaining your view)

Output your analysis in the following JSON format:
{
  "quarter": "Q1_2024",
  "views": [
    {
      "ticker": "XLK",
      "expected_return": 0.03,
      "confidence": 75,
      "reasoning": "Brief explanation"
    }
  ]
}

Be objective and balanced. Consider both positive and negative factors. Your views will be used in a Black-Litterman portfolio optimization framework."""

USER_PROMPT_TEMPLATE = """Analyze the following financial data for {quarter} and generate investment views for portfolio optimization.

================================================================================
FEDERAL RESERVE POLICY (FOMC MINUTES SUMMARY)
================================================================================
{fomc_summary}

================================================================================
QUARTERLY INVESTMENT RESEARCH
================================================================================
{research_summary}

================================================================================
SECTOR ETF ANALYSIS
================================================================================

{ticker_analyses}

================================================================================
POLITICAL NEWS SUMMARY
================================================================================
{political_news}

Based on this comprehensive data, generate your investment views for each of the 6 sector ETFs: XLK, XLY, ITA, XLE, XLV, XLF.

Remember to output in JSON format as specified."""

def create_ticker_analysis_section(ticker: str, ticker_name: str, 
                                   headlines: list, 
                                   historical_return: float) -> str:
    """
    Create analysis section for a single ticker
    """
    section = []
    section.append(f"TICKER: {ticker} - {ticker_name}")
    section.append("-" * 80)
    section.append(f"Previous Quarter Return: {historical_return:.2%}")
    section.append("")
    
    if headlines:
        section.append(f"Top Bloomberg Headlines ({len(headlines)} total):")
        for i, headline in enumerate(headlines[:10], 1):  # Top 10 headlines
            section.append(f"  {i}. {headline}")
    else:
        section.append("No recent headlines available")
    
    section.append("")
    return "\n".join(section)

def create_user_prompt(quarter_data: dict) -> str:
    """
    Create full user prompt from quarterly data
    
    Args:
        quarter_data: Dictionary with all quarterly data
        
    Returns:
        Formatted prompt string
    """
    ticker_names = {
        'XLK': 'Technology Select Sector SPDR',
        'XLY': 'Consumer Discretionary Select Sector SPDR',
        'ITA': 'iShares U.S. Aerospace & Defense ETF',
        'XLE': 'Energy Select Sector SPDR',
        'XLV': 'Health Care Select Sector SPDR',
        'XLF': 'Financial Select Sector SPDR'
    }
    
    # Limit text lengths for context window
    fomc_summary = quarter_data.get('fomc_summary', '')[:3000]
    research_summary = quarter_data.get('research_summary', '')[:3000]
    political_news = quarter_data.get('political_news', '')[:5000]
    
    # Build ticker analyses
    ticker_analyses = []
    for ticker in ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']:
        ticker_info = quarter_data['tickers'].get(ticker, {})
        headlines = ticker_info.get('bloomberg_headlines', [])
        hist_return = ticker_info.get('historical_return', 0.0)
        
        analysis = create_ticker_analysis_section(
            ticker, 
            ticker_names[ticker],
            headlines,
            hist_return
        )
        ticker_analyses.append(analysis)
    
    ticker_analyses_text = "\n".join(ticker_analyses)
    
    prompt = USER_PROMPT_TEMPLATE.format(
        quarter=quarter_data['quarter'],
        fomc_summary=fomc_summary,
        research_summary=research_summary,
        ticker_analyses=ticker_analyses_text,
        political_news=political_news
    )
    
    return prompt

