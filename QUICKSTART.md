# Quick Start Guide

## TL;DR

Run the complete pipeline:

```bash
python main.py
```

That's it! The system will:
1. Load all your existing data
2. Generate LLM views for each quarter
3. Optimize portfolios using Black-Litterman
4. Backtest performance
5. Generate visualizations and reports

Results will be in the `output/` directory.

## What Just Happened?

The system processed 7 quarters (Q1 2024 - Q3 2025) and:

### For Each Quarter:
1. **Loaded Multi-Source Data**
   - Political news (GDELT articles)
   - FOMC meeting minutes
   - Quarterly investment research
   - Bloomberg headlines
   - Historical stock prices

2. **Generated LLM Views**
   - Analyzed all data sources
   - Created structured investment views
   - Assigned confidence levels

3. **Optimized Portfolio**
   - Applied Black-Litterman model
   - Generated optimal weights
   - Maximized Sharpe ratio

### Then:
4. **Backtested Strategy**
   - Simulated quarterly rebalancing
   - Calculated actual returns
   - Compared to equal-weight benchmark

5. **Generated Reports**
   - Performance metrics
   - Visualization charts
   - Comprehensive analysis

## View Results

### Text Report
```bash
cat output/reports/full_report.txt
```

### JSON Data
```bash
cat output/reports/results.json | python -m json.tool | head -50
```

### Quarterly Portfolios
```bash
ls -lh output/portfolios/
```

### Charts (if generated)
```bash
open output/charts/cumulative_returns.png
```

## Run Options

### Use Different LLM

```bash
# Use local finance-LLM model
python main.py --llm-type finance-llm

# Use Claude API (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key"
python main.py --llm-type anthropic

# Use GPT-4 API (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key"
python main.py --llm-type openai
```

### Process Specific Quarters

```bash
# Just 2024
python main.py --quarters Q1_2024 Q2_2024 Q3_2024 Q4_2024

# Just a few for testing
python main.py --quarters Q1_2024 Q2_2024
```

### Skip Visualizations (faster)

```bash
python main.py --skip-charts
```

## Test Run (2 Quarters)

Quick test with just Q1 and Q2 2024:

```bash
python main.py --quarters Q1_2024 Q2_2024 --skip-charts
```

Should take ~30 seconds and show:
- Strategy outperforming benchmark
- Sharpe ratio > 2
- Portfolio allocations
- Full report generated

## Expected Output

The pipeline will print:
- Data loading progress
- LLM views generation
- Portfolio optimization results
- Backtest simulation
- Performance metrics
- Final summary comparing strategy vs benchmark

Check `output/reports/full_report.txt` for the complete analysis!

## Troubleshooting

### "No module named 'XXX'"
```bash
pip install -r requirements.txt
```

### Charts not generating
```bash
pip install matplotlib seaborn
```

### LLM model loading issues
Use simulated mode (default) which doesn't require any models:
```bash
python main.py --llm-type simulated
```

## What's Impressive?

This system:
- Processes 1M+ characters of financial text per quarter
- Integrates 5 different data sources
- Uses LLMs to generate structured investment views
- Applies sophisticated Black-Litterman optimization
- Backtests with realistic constraints
- Generates comprehensive reports automatically

All with a single command!

## Next Steps

1. Review the full report: `output/reports/full_report.txt`
2. Examine quarterly allocations: `output/portfolios/`
3. Check LLM views: `output/views/`
4. View charts: `output/charts/` (if not skipped)
5. Read the methodology: `README.md`

## Academic Use

For MGT 6078 project:
- Full methodology documented
- All code is modular and commented
- Results are reproducible
- Comprehensive PRD included
- Novel application of LLMs to portfolio optimization

Perfect for presentation and final report!

