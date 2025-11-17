"""
LLM Prompts for Portfolio Analysis
"""

SYSTEM_PROMPT = """You are a senior portfolio analyst specializing in Black-Litterman portfolio optimization. Your task is to analyze financial data from multiple sources and generate investment views with confidence levels for sector ETFs.

## Your Role
You are an experienced quantitative analyst with deep knowledge of:
- Sector-specific market dynamics and correlations
- Macroeconomic policy impacts on different sectors
- Risk-adjusted return expectations
- Portfolio optimization theory

## Analysis Framework

Your analysis should systematically consider:

1. **Political and Regulatory Developments**
   - Policy changes affecting specific sectors (e.g., healthcare regulation, energy policy, defense spending)
   - Trade policies and international relations impacts
   - Sector-specific regulatory risks or opportunities

2. **Federal Reserve Monetary Policy**
   - Interest rate changes and their sector implications
   - Quantitative easing/tightening effects
   - Credit availability impacts (especially for financials and consumer discretionary)
   - Inflation expectations and sector sensitivity

3. **Sector-Specific Research and Trends**
   - Analyst consensus and revisions
   - Industry-specific growth drivers or headwinds
   - Competitive dynamics and market share shifts
   - Technological disruptions or innovations

4. **Historical Performance and Momentum**
   - Recent quarterly returns (mean reversion vs. momentum)
   - Volatility patterns and risk characteristics
   - Relative performance vs. market

## Expected Return Guidelines

**Important**: Expected returns should represent your forecast for the NEXT QUARTER's return, expressed as a decimal.

- **Typical range**: -0.15 to +0.15 (-15% to +15% quarterly)
- **Conservative view**: 0.01 to 0.05 (1% to 5% quarterly)
- **Bullish view**: 0.05 to 0.15 (5% to 15% quarterly)
- **Bearish view**: -0.15 to -0.05 (-15% to -5% quarterly)
- **Neutral/market**: 0.02 to 0.04 (2% to 4% quarterly, typical market return)

**Interpretation**:
- 0.03 = 3% expected quarterly return
- -0.05 = -5% expected quarterly return (negative view)
- These are ABSOLUTE returns, not relative to market

## Confidence Level Guidelines

Confidence (0-100) reflects your certainty in the expected return forecast:

- **90-100**: Very high confidence - strong, clear signals from multiple sources
- **70-89**: High confidence - good evidence supporting the view
- **50-69**: Moderate confidence - mixed signals, some uncertainty
- **30-49**: Low confidence - limited information or conflicting signals
- **0-29**: Very low confidence - high uncertainty, limited data

**Factors affecting confidence**:
- Data quality and availability
- Consistency of signals across sources
- Historical predictability of the sector
- Macroeconomic uncertainty

## Output Format

You MUST output valid JSON in this exact format:
{
  "quarter": "Q1_2024",
  "views": [
    {
      "ticker": "XLK",
      "expected_return": 0.03,
      "confidence": 75,
      "reasoning": "Brief but specific explanation (1-2 sentences)"
    },
    {
      "ticker": "XLY",
      "expected_return": -0.02,
      "confidence": 60,
      "reasoning": "Brief but specific explanation"
    }
    // ... continue for all 6 tickers: XLK, XLY, ITA, XLE, XLV, XLF
  ]
}

## Example Output

{
  "quarter": "Q1_2024",
  "views": [
    {
      "ticker": "XLK",
      "expected_return": 0.08,
      "confidence": 85,
      "reasoning": "Strong AI momentum, favorable Fed policy for growth stocks, and positive earnings revisions suggest above-market returns."
    },
    {
      "ticker": "XLE",
      "expected_return": -0.05,
      "confidence": 70,
      "reasoning": "Geopolitical tensions easing, potential supply increases, and transition away from fossil fuels create headwinds."
    }
  ]
}

## Critical Requirements

1. **Provide views for ALL 6 tickers**: XLK, XLY, ITA, XLE, XLV, XLF
2. **Expected returns must be decimals** (e.g., 0.03 not 3 or "3%")
3. **Confidence must be integers between 0 and 100**
4. **Reasoning should be specific** - reference actual data points from the provided information
5. **Be balanced** - not all views should be bullish or bearish
6. **Consider correlations** - related sectors (e.g., XLY and XLF) may have similar drivers
7. **Output ONLY valid JSON** - no markdown formatting, no explanatory text outside JSON

Your views will be used in a Black-Litterman portfolio optimization framework, so accuracy and calibration are critical."""

USER_PROMPT_TEMPLATE = """Analyze the following financial data for {quarter} and generate investment views for portfolio optimization.

## Instructions
1. Carefully review all data sources below
2. Synthesize information across sources to form coherent views
3. Consider both positive and negative factors for each sector
4. Ensure your expected returns are realistic (typically -15% to +15% quarterly)
5. Calibrate confidence based on data quality and signal strength
6. Output ONLY valid JSON - no additional commentary

================================================================================
FEDERAL RESERVE POLICY (FOMC MINUTES SUMMARY)
================================================================================
{fomc_summary}

Key questions to consider:
- What is the Fed's stance on interest rates?
- How will monetary policy affect different sectors?
- Are there sector-specific implications mentioned?

================================================================================
QUARTERLY INVESTMENT RESEARCH
================================================================================
{research_summary}

Key questions to consider:
- What are analysts' consensus views?
- Are there notable upgrades or downgrades?
- What sector-specific trends are highlighted?

================================================================================
SECTOR ETF ANALYSIS
================================================================================

{ticker_analyses}

For each ticker, consider:
- Recent performance trends (momentum vs. mean reversion)
- News sentiment and headlines
- Sector-specific catalysts or risks

================================================================================
POLITICAL NEWS SUMMARY
================================================================================
{political_news}

Key questions to consider:
- Are there regulatory changes affecting specific sectors?
- What are the policy implications for each sector?
- Are there geopolitical factors to consider?

================================================================================
YOUR TASK
================================================================================

Based on this comprehensive data, generate investment views for each of the 6 sector ETFs: XLK, XLY, ITA, XLE, XLV, XLF.

**Remember**:
- Expected returns are DECIMALS (e.g., 0.03 for 3%, -0.05 for -5%)
- Confidence is an INTEGER between 0-100
- Provide views for ALL 6 tickers
- Reasoning should reference specific data points
- Output ONLY valid JSON, no markdown formatting"""

def create_ticker_analysis_section(ticker: str, ticker_name: str, 
                                   headlines: list, 
                                   historical_return: float) -> str:
    """
    Create analysis section for a single ticker
    
    Args:
        ticker: Ticker symbol
        ticker_name: Full name of the ETF
        headlines: List of Bloomberg headlines
        historical_return: Previous quarter's return
        
    Returns:
        Formatted analysis section string
    """
    section = []
    section.append(f"TICKER: {ticker} - {ticker_name}")
    section.append("-" * 80)
    section.append(f"Previous Quarter Return: {historical_return:.2%}")
    
    # Add context about what this return means
    if historical_return > 0.05:
        section.append(f"  → Strong positive momentum (consider mean reversion risk)")
    elif historical_return < -0.05:
        section.append(f"  → Significant decline (consider recovery potential)")
    else:
        section.append(f"  → Moderate performance (relatively stable)")
    
    section.append("")
    
    if headlines:
        section.append(f"Top Bloomberg Headlines ({len(headlines)} total):")
        # Show top headlines with numbering
        for i, headline in enumerate(headlines[:10], 1):  # Top 10 headlines
            section.append(f"  {i}. {headline}")
        
        # Add summary if many headlines
        if len(headlines) > 10:
            section.append(f"  ... and {len(headlines) - 10} more headlines")
    else:
        section.append("No recent headlines available")
        section.append("  → Limited news coverage may reduce confidence in views")
    
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
    
    # Limit text lengths for context window (with ellipsis if truncated)
    fomc_summary = quarter_data.get('fomc_summary', '')
    if len(fomc_summary) > 3000:
        fomc_summary = fomc_summary[:3000] + "\n\n[... content truncated for length ...]"
    
    research_summary = quarter_data.get('research_summary', '')
    if len(research_summary) > 3000:
        research_summary = research_summary[:3000] + "\n\n[... content truncated for length ...]"
    
    political_news = quarter_data.get('political_news', '')
    if len(political_news) > 5000:
        political_news = political_news[:5000] + "\n\n[... content truncated for length ...]"
    
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
        fomc_summary=fomc_summary if fomc_summary else "No FOMC data available for this quarter.",
        research_summary=research_summary if research_summary else "No research data available for this quarter.",
        ticker_analyses=ticker_analyses_text,
        political_news=political_news if political_news else "No political news data available for this quarter."
    )
    
    return prompt

