# Workflow Diagram Documentation

This directory contains visual representations of the project's workflow, showing how data flows from inputs through processing steps to outputs.

## Generated Diagrams

### 1. `workflow_diagram.png`
A high-level overview of the workflow showing:
- **Inputs**: Configuration file, stock data, political news, Bloomberg headlines, FOMC PDFs, and investor research
- **Processing Pipeline**: Data loading → LLM analysis → Portfolio optimization → Backtesting → Metrics calculation
- **Outputs**: Portfolio allocations, LLM views, charts, reports, and metrics
- **Optional**: Hyperparameter optimization loop

### 2. `workflow_diagram_detailed.png`
A more detailed view showing:
- Sub-processes within each major step
- Data flow between components
- Internal processing details

### 3. `workflow_diagram_graphviz.png` (if Graphviz installed)
A Graphviz-generated diagram from `workflow_diagram.dot`

## How to Generate/Regenerate Diagrams

### Python Script (Recommended)
```bash
python workflow_diagram.py
```

This generates both the standard and detailed workflow diagrams as PNG files in the `output/` directory.

### Mermaid Diagram
The `workflow_diagram.mmd` file contains a Mermaid diagram that can be:
- Rendered in GitHub (automatically renders `.mmd` files)
- Rendered in Markdown viewers that support Mermaid
- Converted using online tools like [Mermaid Live Editor](https://mermaid.live)

### Graphviz Diagram
If you have Graphviz installed:
```bash
dot -Tpng workflow_diagram.dot -o output/workflow_diagram_graphviz.png
```

Or for other formats:
```bash
dot -Tsvg workflow_diagram.dot -o output/workflow_diagram_graphviz.svg
dot -Tpdf workflow_diagram.dot -o output/workflow_diagram_graphviz.pdf
```

## Workflow Overview

### Inputs
1. **config.yaml**: Configuration file with hyperparameters, tickers, quarters, etc.
2. **Stock Data**: Historical price data for sector ETFs (XLK, XLY, ITA, XLE, XLV, XLF)
3. **Political News**: Collected from GDELT API
4. **Bloomberg Headlines**: Text files with Bloomberg headlines
5. **FOMC PDFs**: Federal Reserve meeting minutes and statements
6. **Investor Research**: Quarterly sector research reports

### Processing Pipeline
1. **DataLoader**: Loads and formats all data sources for a given quarter
2. **LLMAnalyzer**: Uses Finance-LLM to analyze data and generate investment views
3. **ViewsConverter**: Converts LLM views into Black-Litterman format (P, Q, Ω matrices)
4. **PortfolioOptimizer**: Runs Black-Litterman optimization to generate optimal portfolio weights
5. **Backtester**: Simulates portfolio performance quarter by quarter
6. **PerformanceMetrics**: Calculates risk/return metrics (Sharpe ratio, max drawdown, etc.)

### Outputs
1. **Portfolio Allocations** (JSON): Optimal weights for each quarter
2. **LLM Views** (JSON): Generated views from the LLM analysis
3. **Performance Charts** (PNG): Visualizations of returns, allocations, risk/return scatter
4. **Backtest Results** (JSON): Detailed backtest simulation results
5. **Full Report** (TXT, JSON): Comprehensive analysis report
6. **Metrics Summary**: Performance metrics and statistics

### Optional: Hyperparameter Optimization
The `HyperparameterOptimizer` can be run separately to find optimal hyperparameters:
- Grid search, random search, or ML-based optimization
- Updates `config.yaml` with best parameters
- Feeds back into the main pipeline

## Quarterly Loop

The entire pipeline runs **for each quarter** specified in `config.yaml`:
- Q1_2024 → Q2_2024 → Q3_2024 → Q4_2024 → Q1_2025 → Q2_2025 → Q3_2025

Each quarter is processed independently, with portfolio rebalancing at the start of each quarter.

## Usage in Documentation

### In Markdown
```markdown
![Workflow Diagram](output/workflow_diagram.png)
```

### In Presentations
The PNG files can be directly inserted into PowerPoint, Keynote, or other presentation software.

### In LaTeX
```latex
\includegraphics[width=\textwidth]{output/workflow_diagram.png}
```

## Customization

To customize the diagrams, edit `workflow_diagram.py`:
- Adjust colors by modifying the color variables
- Add/remove components by modifying the box lists
- Change layout by adjusting coordinates
- Modify styling by changing `FancyBboxPatch` parameters

## Dependencies

For the Python script:
- `matplotlib`
- `numpy`

For Graphviz:
- Install Graphviz: `brew install graphviz` (macOS) or `apt-get install graphviz` (Linux)

For Mermaid:
- No installation needed (renders in GitHub and many Markdown viewers)

