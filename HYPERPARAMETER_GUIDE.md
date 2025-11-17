# Hyperparameter Optimization Guide

## Temperature: Is 0.7 Good?

**Temperature 0.7 is reasonable but can be optimized** depending on your goals:

### Temperature Guidelines

- **0.0-0.3**: Very deterministic, consistent outputs. Good for:
  - Reproducible results
  - When you want the model to follow instructions precisely
  - Risk: May be too conservative, less creative analysis

- **0.5-0.7**: Balanced creativity and consistency (current setting). Good for:
  - General-purpose analysis
  - Balanced between exploration and exploitation
  - Most common default

- **0.8-1.2**: More creative and diverse outputs. Good for:
  - Exploring different perspectives
  - When you want varied analysis
  - Risk: May produce inconsistent or extreme views

- **>1.2**: Very creative, potentially erratic. Usually not recommended for financial analysis.

### Recommendation for Your Use Case

For **portfolio optimization**, I recommend:
- **0.3-0.5**: More deterministic, consistent views
- **0.5-0.7**: Balanced (current setting)
- Test both and see which produces better Sharpe ratios

## Hyperparameters to Optimize

### 1. LLM Hyperparameters (if using API models)

- **temperature**: 0.3-0.9 (sampling randomness)
- **top_p**: 0.7-0.95 (nucleus sampling)

### 2. Black-Litterman Hyperparameters

- **risk_aversion**: 1.5-4.0
  - Lower = more aggressive (higher expected returns, more risk)
  - Higher = more conservative (lower risk, lower returns)
  - Typical: 2.5-3.0 for equity portfolios

- **tau_for_covariance**: 0.01-0.1
  - Uncertainty in prior (market equilibrium)
  - Lower = trust market more
  - Higher = trust views more
  - Typical: 0.025-0.05

- **tau_omega**: 0.025-0.1
  - Uncertainty in views
  - Lower = more confident in views
  - Higher = less confident in views
  - Typical: 0.05

- **relative_confidence**: 0.5-1.5
  - Scales view confidence
  - Lower = less weight on views
  - Higher = more weight on views
  - Typical: 1.0

### 3. Portfolio Constraints

- **allow_shorts**: True/False
  - True = can short assets (more flexibility)
  - False = long-only (more conservative)

- **max_weight**: 1.0-2.0
  - Maximum position size
  - 1.0 = no leverage
  - 2.0 = 200% max position

### 4. Data Settings

- **lookback_days**: 126-315
  - Historical data window for covariance
  - Shorter = more responsive to recent changes
  - Longer = more stable, less responsive
  - Typical: 252 (1 year)

## How to Run Hyperparameter Optimization

### Quick Start: Random Search (Finance-LLM Local Model)

```bash
# Test 20 random configurations with local finance-llm model
python hyperparameter_optimization.py --method random --n-trials 20 --llm-type finance-llm

# Test with specific quarters
python hyperparameter_optimization.py --method random --n-trials 30 --llm-type finance-llm --quarters Q1_2024 Q2_2024 Q3_2024
```

### Grid Search (More Exhaustive)

```bash
# Test up to 50 combinations with finance-llm
python hyperparameter_optimization.py --method grid --max-combinations 50 --llm-type finance-llm
```

### With Simulated Model (Faster Testing)

```bash
# Use simulated model for faster testing (no LLM inference)
# Note: This only tests Black-Litterman hyperparameters, not LLM temperature
python hyperparameter_optimization.py --method random --n-trials 20 --llm-type simulated
```

### Machine Learning-Based Optimization (Recommended)

The system now supports several ML-based hyperparameter tuning methods that are more efficient than random or grid search:

#### 1. Bayesian Optimization (Gaussian Process)

Uses Gaussian Process to model the objective function and intelligently select hyperparameters:

```bash
# Bayesian optimization with 30 trials (recommended for efficiency)
python hyperparameter_optimization.py --method bayesian --n-trials 30 --llm-type finance-llm --objective sharpe_ratio

# Optimize for annualized return instead
python hyperparameter_optimization.py --method bayesian --n-trials 30 --llm-type finance-llm --objective annualized_return
```

**Advantages:**
- More efficient than random search (finds better configs with fewer trials)
- Models uncertainty in predictions
- Good for expensive evaluations (like LLM inference)

**Requirements:** `pip install scikit-optimize`

#### 2. Tree-structured Parzen Estimator (TPE) - Optuna

Uses TPE algorithm which is particularly effective for hyperparameter optimization:

```bash
# TPE optimization (default Optuna sampler)
python hyperparameter_optimization.py --method optuna --n-trials 30 --llm-type finance-llm

# Or explicitly specify TPE
python hyperparameter_optimization.py --method optuna-tpe --n-trials 30 --llm-type finance-llm

# CMA-ES sampler (alternative)
python hyperparameter_optimization.py --method optuna-cmaes --n-trials 30 --llm-type finance-llm
```

**Advantages:**
- State-of-the-art performance for hyperparameter optimization
- Handles mixed parameter types well (continuous, discrete, categorical)
- Multiple sampler options (TPE, CMA-ES, Random)

**Requirements:** `pip install optuna`

#### 3. Random Forest-Based Optimization

Uses Random Forest surrogate model to guide hyperparameter search:

```bash
# Random Forest optimization
python hyperparameter_optimization.py --method random-forest --n-trials 30 --llm-type finance-llm
```

**Advantages:**
- Good for high-dimensional spaces
- Handles non-linear relationships well
- Fast predictions once trained

**Requirements:** `pip install scikit-learn` (usually already installed)

### Comparison of Methods

| Method | Efficiency | Best For | Trials Needed |
|--------|-----------|----------|---------------|
| Random Search | Low | Quick exploration | 50-100+ |
| Grid Search | Very Low | Small spaces | 20-50 |
| Bayesian (GP) | High | Expensive evaluations | 20-40 |
| Optuna (TPE) | Very High | General purpose | 20-40 |
| Random Forest | Medium-High | High-dimensional | 30-50 |

**Recommendation:** Start with **Optuna (TPE)** for best results, or **Bayesian Optimization** if you prefer scikit-optimize.

## Understanding Results

The optimizer evaluates configurations using:
1. **Sharpe Ratio** (primary metric) - risk-adjusted returns
2. **Annualized Return** (secondary metric)
3. **Sortino Ratio** - downside risk-adjusted returns
4. **Max Drawdown** - worst peak-to-trough decline
5. **Outperformance** - vs equal-weight benchmark
6. **Information Ratio** - active return per unit of tracking error

### Best Practices

1. **Start with ML-Based Methods**: Use Bayesian Optimization or Optuna (TPE) for better efficiency
2. **Use Validation Quarters**: Don't optimize on test data
3. **Focus on Sharpe Ratio**: Better risk-adjusted performance (default objective)
4. **Check Consistency**: Best config should work across multiple quarters
5. **Consider Transaction Costs**: Very high turnover may not be realistic
6. **Cache Results**: ML methods automatically cache evaluations to avoid redundant work
7. **Start Small**: Begin with 20-30 trials, then increase if needed

### Example Workflow (Finance-LLM)

**Recommended ML-Based Workflow:**

```bash
# Step 1: ML-based optimization with Optuna TPE (30 trials)
python hyperparameter_optimization.py --method optuna --n-trials 30 --llm-type finance-llm --objective sharpe_ratio

# Step 2: Review results in output/hyperparameter_optimization/
# Look at analysis_*.json for best configurations

# Step 3: Fine-tune around best config (optional - run more trials)
python hyperparameter_optimization.py --method optuna --n-trials 20 --llm-type finance-llm --quarters Q1_2024 Q2_2024 Q3_2024

# Step 4: Validate on hold-out quarters with best config
python main.py --llm-type finance-llm --quarters Q1_2025 Q2_2025 Q3_2025
```

**Alternative: Quick Random Search Workflow:**

```bash
# Step 1: Quick exploration with random search (20 trials)
python hyperparameter_optimization.py --method random --n-trials 20 --llm-type finance-llm

# Step 2-4: Same as above
```

**Note**: The finance-llm model requires:
- `transformers` and `torch` installed
- GPU recommended (but will work on CPU, slower)
- Model downloads automatically on first use (~7GB)

**ML Optimization Libraries:**
- For Bayesian Optimization: `pip install scikit-optimize`
- For Optuna (TPE): `pip install optuna`
- For Random Forest: `scikit-learn` (usually already installed)
- All libraries are optional - the script will use available methods

## Expected Results

Based on typical Black-Litterman optimization:

- **Risk Aversion**: 2.0-3.0 usually optimal
- **Tau Values**: 0.025-0.075 range
- **Temperature**: 0.3-0.7 for financial analysis
- **Lookback**: 252 days (1 year) is standard

## Tips

1. **Temperature**: Lower (0.3-0.5) often better for financial analysis
2. **Risk Aversion**: Start at 2.5, adjust based on risk tolerance
3. **Tau Values**: Smaller values = trust market more, larger = trust views more
4. **Allow Shorts**: Usually improves Sharpe ratio but increases complexity
5. **Lookback**: Longer windows = more stable, shorter = more responsive

## Output Files

Results are saved to `output/hyperparameter_optimization/`:
- `results_*.json`: All trial results
- `analysis_*.json`: Summary with best configurations

