#!/usr/bin/env python3
"""
Hyperparameter Optimization
Grid search, random search, and ML-based optimization for optimal hyperparameters
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from itertools import product
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import DataLoader
from data.formatter import DataFormatter
from llm.analyzer import LLMAnalyzer
from optimization.portfolio_optimizer import PortfolioOptimizer
from backtesting.backtester import Backtester
from backtesting.metrics import PerformanceMetrics

# ML optimization libraries (optional imports)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HyperparameterOptimizer:
    """Optimize hyperparameters using grid search, random search, or ML-based methods"""
    
    def __init__(self, quarters: List[str] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            quarters: List of quarters to use for optimization
        """
        self.quarters = quarters or ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
        self.loader = DataLoader()
        self.formatter = DataFormatter()
        self.backtester = Backtester(initial_capital=100000)
        self._evaluation_cache = {}  # Cache for repeated evaluations
        
    def define_search_space(self) -> Dict:
        """
        Define hyperparameter search space
        
        Returns:
            Dictionary with parameter ranges
        """
        return {
            # LLM hyperparameters (for finance-llm local model)
            'llm_temperature': [0.3, 0.5, 0.7, 0.9],  # Lower = more deterministic
            'llm_top_p': [0.7, 0.9, 0.95],  # Nucleus sampling
            
            # Black-Litterman hyperparameters
            'risk_aversion': [1.5, 2.0, 2.5, 3.0, 3.5],  # Higher = more conservative
            'tau_for_covariance': [0.01, 0.025, 0.05, 0.075, 0.1],  # Prior uncertainty
            'tau_omega': [0.025, 0.05, 0.075, 0.1],  # Views uncertainty
            'relative_confidence': [0.5, 0.75, 1.0, 1.25, 1.5],  # Confidence scaling
            
            # Portfolio constraints
            'allow_shorts': [True, False],
            'max_weight': [1.0, 1.5, 2.0],  # Max position size
            
            # Data settings
            'lookback_days': [126, 189, 252, 315]  # 6 months to 15 months
        }
    
    def define_continuous_search_space(self) -> Dict:
        """
        Define continuous search space for ML-based optimization
        
        Returns:
            Dictionary with continuous parameter ranges (min, max)
        """
        return {
            'llm_temperature': (0.3, 0.9),
            'llm_top_p': (0.7, 0.95),
            'risk_aversion': (1.5, 3.5),
            'tau_for_covariance': (0.01, 0.1),
            'tau_omega': (0.025, 0.1),
            'relative_confidence': (0.5, 1.5),
            'max_weight': (1.0, 2.0),
            'lookback_days': (126, 315)
        }
    
    def generate_random_config(self, search_space: Dict) -> Dict:
        """
        Generate random hyperparameter configuration
        
        Args:
            search_space: Parameter search space
            
        Returns:
            Random configuration dictionary
        """
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        return config
    
    def evaluate_config(self, config: Dict, llm_type: str = 'simulated', 
                        use_cache: bool = True) -> Dict:
        """
        Evaluate a hyperparameter configuration
        
        Args:
            config: Hyperparameter configuration
            llm_type: LLM model type to use
            use_cache: Whether to use cached results for identical configs
            
        Returns:
            Dictionary with performance metrics
        """
        # Check cache
        if use_cache:
            config_key = json.dumps(config, sort_keys=True)
            if config_key in self._evaluation_cache:
                return self._evaluation_cache[config_key]
        
        try:
            # Initialize components with config
            analyzer = LLMAnalyzer(
                model_type=llm_type,
                temperature=config.get('llm_temperature', 0.7),
                top_p=config.get('llm_top_p', 0.9)
            )
            optimizer = PortfolioOptimizer(
                allow_shorts=config.get('allow_shorts', True),
                max_weight=config.get('max_weight', 2.0)
            )
            
            # Set Black-Litterman hyperparameters
            optimizer.set_hyperparameters(
                risk_aversion=config.get('risk_aversion', 2.5),
                tau_for_covariance=config.get('tau_for_covariance', 0.025),
                tau_omega=config.get('tau_omega', 0.05),
                relative_confidence=config.get('relative_confidence', 1.0)
            )
            
            # Process quarters
            portfolio_results = []
            for quarter in self.quarters:
                try:
                    quarter_data = self.loader.load_quarterly_data(quarter)
                    llm_views = analyzer.generate_views(quarter_data)
                    
                    portfolio = optimizer.optimize_quarter(
                        quarter_data, 
                        llm_views,
                        lookback_days=config.get('lookback_days', 252)
                    )
                    portfolio_results.append(portfolio)
                except Exception as e:
                    print(f"  Error processing {quarter}: {e}")
                    continue
            
            if len(portfolio_results) == 0:
                return {'error': 'No successful portfolios'}
            
            # Run backtest
            stock_data = self.loader.load_stock_data()
            quarter_dates_map = self.loader.quarter_dates
            
            backtest_results = self.backtester.run_backtest(
                portfolio_results,
                stock_data,
                quarter_dates_map
            )
            
            # Calculate metrics
            equal_weights = {ticker: 1/6 for ticker in optimizer.tickers}
            benchmark_results = self.backtester.run_benchmark(
                equal_weights,
                stock_data,
                quarter_dates_map,
                [p['quarter'] for p in portfolio_results]
            )
            
            metrics = PerformanceMetrics.calculate_all_metrics(
                backtest_results,
                benchmark_results
            )
            
            # Return key metrics
            result = {
                'config': config,
                'annualized_return': metrics['annualized_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics['max_drawdown'],
                'outperformance': metrics.get('outperformance', 0),
                'information_ratio': metrics.get('information_ratio', 0),
                'total_return': metrics['total_return'],
                'volatility': metrics['volatility'],
                'win_rate': metrics['win_rate']
            }
            
            # Cache result
            if use_cache:
                config_key = json.dumps(config, sort_keys=True)
                self._evaluation_cache[config_key] = result
            
            return result
            
        except Exception as e:
            error_result = {'error': str(e), 'config': config}
            if use_cache:
                config_key = json.dumps(config, sort_keys=True)
                self._evaluation_cache[config_key] = error_result
            return error_result
    
    def grid_search(self, search_space: Dict, llm_type: str = 'simulated',
                   max_combinations: int = 50) -> List[Dict]:
        """
        Perform grid search over hyperparameter space
        
        Args:
            search_space: Parameter search space
            llm_type: LLM model type
            max_combinations: Maximum number of combinations to test
            
        Returns:
            List of evaluation results
        """
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[p] for p in param_names]
        
        all_combinations = list(product(*param_values))
        
        # Limit if too many
        if len(all_combinations) > max_combinations:
            print(f"Too many combinations ({len(all_combinations)}). "
                  f"Randomly sampling {max_combinations}...")
            all_combinations = random.sample(all_combinations, max_combinations)
        
        results = []
        print(f"\n{'='*80}")
        print(f"GRID SEARCH: Testing {len(all_combinations)} configurations")
        print(f"{'='*80}\n")
        
        for i, combination in enumerate(all_combinations, 1):
            config = dict(zip(param_names, combination))
            print(f"[{i}/{len(all_combinations)}] Testing: {config}")
            
            result = self.evaluate_config(config, llm_type)
            if 'error' not in result:
                print(f"  ✓ Return: {result['annualized_return']:.2%}, "
                      f"Sharpe: {result['sharpe_ratio']:.4f}")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        return results
    
    def random_search(self, search_space: Dict, n_trials: int = 20,
                     llm_type: str = 'simulated') -> List[Dict]:
        """
        Perform random search over hyperparameter space
        
        Args:
            search_space: Parameter search space
            n_trials: Number of random configurations to test
            llm_type: LLM model type
            
        Returns:
            List of evaluation results
        """
        results = []
        print(f"\n{'='*80}")
        print(f"RANDOM SEARCH: Testing {n_trials} random configurations")
        print(f"{'='*80}\n")
        
        for i in range(n_trials):
            config = self.generate_random_config(search_space)
            print(f"[{i+1}/{n_trials}] Testing: {config}")
            
            result = self.evaluate_config(config, llm_type)
            if 'error' not in result:
                print(f"  ✓ Return: {result['annualized_return']:.2%}, "
                      f"Sharpe: {result['sharpe_ratio']:.4f}")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        return results
    
    def bayesian_optimization(self, search_space: Dict, n_trials: int = 30,
                             llm_type: str = 'simulated', 
                             objective: str = 'sharpe_ratio') -> List[Dict]:
        """
        Perform Bayesian Optimization using Gaussian Process (scikit-optimize)
        
        Uses Gaussian Process to model the objective function and intelligently
        select the next hyperparameters to evaluate.
        
        Args:
            search_space: Parameter search space (discrete values)
            n_trials: Number of optimization trials
            llm_type: LLM model type
            objective: Objective metric to optimize ('sharpe_ratio' or 'annualized_return')
            
        Returns:
            List of evaluation results
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian Optimization. "
                "Install with: pip install scikit-optimize"
            )
        
        print(f"\n{'='*80}")
        print(f"BAYESIAN OPTIMIZATION (Gaussian Process): {n_trials} trials")
        print(f"Objective: Maximize {objective}")
        print(f"{'='*80}\n")
        
        # Convert discrete search space to continuous for skopt
        continuous_space = self.define_continuous_search_space()
        
        # Define skopt dimensions
        dimensions = []
        param_names = []
        
        for param in ['llm_temperature', 'llm_top_p', 'risk_aversion', 
                     'tau_for_covariance', 'tau_omega', 'relative_confidence',
                     'max_weight', 'lookback_days']:
            if param in continuous_space:
                min_val, max_val = continuous_space[param]
                dimensions.append(Real(min_val, max_val, name=param))
                param_names.append(param)
        
        # Handle categorical parameters
        allow_shorts_values = [True, False]
        dimensions.append(Categorical(allow_shorts_values, name='allow_shorts'))
        param_names.append('allow_shorts')
        
        # Objective function
        @use_named_args(dimensions=dimensions)
        def objective_function(**params):
            # Convert continuous values back to discrete if needed
            config = {}
            for param in param_names:
                if param == 'allow_shorts':
                    config[param] = params[param]
                elif param in search_space:
                    # Find closest discrete value
                    value = params[param]
                    closest = min(search_space[param], key=lambda x: abs(x - value))
                    config[param] = closest
                else:
                    config[param] = params[param]
            
            # Evaluate configuration
            result = self.evaluate_config(config, llm_type, use_cache=True)
            
            if 'error' in result:
                return 1e6  # Large penalty for errors
            
            # Return negative objective (since we're minimizing)
            score = -result[objective]  # Negative because gp_minimize minimizes
            print(f"  Trial: {config} -> {objective}: {result[objective]:.4f}")
            return score
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            n_calls=n_trials,
            random_state=42,
            acq_func='EI',  # Expected Improvement
            n_initial_points=5  # Random initialization
        )
        
        # Convert results back to our format
        results = []
        for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals)):
            config = {}
            for j, param in enumerate(param_names):
                if param == 'allow_shorts':
                    config[param] = x[j]
                elif param in search_space:
                    # Find closest discrete value
                    value = x[j]
                    closest = min(search_space[param], key=lambda x_val: abs(x_val - value))
                    config[param] = closest
                else:
                    config[param] = x[j]
            
            # Get full evaluation result
            eval_result = self.evaluate_config(config, llm_type, use_cache=True)
            results.append(eval_result)
        
        return results
    
    def optuna_optimization(self, search_space: Dict, n_trials: int = 30,
                           llm_type: str = 'simulated',
                           objective: str = 'sharpe_ratio',
                           sampler: str = 'tpe') -> List[Dict]:
        """
        Perform optimization using Optuna with TPE (Tree-structured Parzen Estimator)
        or other samplers
        
        Args:
            search_space: Parameter search space
            n_trials: Number of optimization trials
            llm_type: LLM model type
            objective: Objective metric to optimize
            sampler: Sampler type ('tpe', 'random', 'cmaes')
            
        Returns:
            List of evaluation results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for TPE optimization. "
                "Install with: pip install optuna"
            )
        
        print(f"\n{'='*80}")
        print(f"OPTUNA OPTIMIZATION ({sampler.upper()}): {n_trials} trials")
        print(f"Objective: Maximize {objective}")
        print(f"{'='*80}\n")
        
        # Select sampler
        if sampler == 'tpe':
            study_sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler == 'random':
            study_sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler == 'cmaes':
            study_sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            study_sampler = optuna.samplers.TPESampler(seed=42)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=study_sampler
        )
        
        # Define objective function
        def optuna_objective(trial):
            # Suggest hyperparameters
            config = {}
            
            # Continuous parameters
            config['llm_temperature'] = trial.suggest_float('llm_temperature', 0.3, 0.9)
            config['llm_top_p'] = trial.suggest_float('llm_top_p', 0.7, 0.95)
            config['risk_aversion'] = trial.suggest_float('risk_aversion', 1.5, 3.5)
            config['tau_for_covariance'] = trial.suggest_float('tau_for_covariance', 0.01, 0.1)
            config['tau_omega'] = trial.suggest_float('tau_omega', 0.025, 0.1)
            config['relative_confidence'] = trial.suggest_float('relative_confidence', 0.5, 1.5)
            config['max_weight'] = trial.suggest_float('max_weight', 1.0, 2.0)
            config['lookback_days'] = trial.suggest_int('lookback_days', 126, 315)
            
            # Categorical parameters
            config['allow_shorts'] = trial.suggest_categorical('allow_shorts', [True, False])
            
            # Round to nearest discrete values if needed
            for param in ['llm_temperature', 'llm_top_p', 'risk_aversion',
                         'tau_for_covariance', 'tau_omega', 'relative_confidence',
                         'max_weight']:
                if param in search_space:
                    closest = min(search_space[param], key=lambda x: abs(x - config[param]))
                    config[param] = closest
            
            # Evaluate configuration
            result = self.evaluate_config(config, llm_type, use_cache=True)
            
            if 'error' in result:
                return -1e6  # Large penalty for errors
            
            score = result[objective]
            print(f"  Trial {trial.number}: {objective}: {score:.4f}")
            return score
        
        # Run optimization
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
        
        # Collect all trial results
        results = []
        for trial in study.trials:
            config = {}
            for param, value in trial.params.items():
                config[param] = value
            
            # Get full evaluation result
            eval_result = self.evaluate_config(config, llm_type, use_cache=True)
            results.append(eval_result)
        
        print(f"\n✓ Best {objective}: {study.best_value:.4f}")
        print(f"✓ Best parameters: {study.best_params}")
        
        return results
    
    def random_forest_optimization(self, search_space: Dict, n_trials: int = 30,
                                  llm_type: str = 'simulated',
                                  objective: str = 'sharpe_ratio',
                                  n_initial: int = 10) -> List[Dict]:
        """
        Perform optimization using Random Forest surrogate model
        
        Uses Random Forest to model the objective function and guide search
        
        Args:
            search_space: Parameter search space
            n_trials: Total number of trials
            llm_type: LLM model type
            objective: Objective metric to optimize
            n_initial: Number of random initial trials
            
        Returns:
            List of evaluation results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for Random Forest optimization. "
                "Install with: pip install scikit-learn"
            )
        
        print(f"\n{'='*80}")
        print(f"RANDOM FOREST OPTIMIZATION: {n_trials} trials ({n_initial} initial random)")
        print(f"Objective: Maximize {objective}")
        print(f"{'='*80}\n")
        
        results = []
        continuous_space = self.define_continuous_search_space()
        
        # Phase 1: Random initialization
        print(f"Phase 1: Random initialization ({n_initial} trials)")
        for i in range(n_initial):
            config = self.generate_random_config(search_space)
            result = self.evaluate_config(config, llm_type, use_cache=True)
            results.append(result)
            if 'error' not in result:
                print(f"  [{i+1}/{n_initial}] {objective}: {result[objective]:.4f}")
        
        # Phase 2: RF-guided search
        print(f"\nPhase 2: Random Forest-guided search ({n_trials - n_initial} trials)")
        for i in range(n_initial, n_trials):
            # Train Random Forest on current results
            valid_results = [r for r in results if 'error' not in r]
            if len(valid_results) < 5:
                # Not enough data, use random
                config = self.generate_random_config(search_space)
            else:
                # Prepare training data
                X_train = []
                y_train = []
                
                for result in valid_results:
                    config = result['config']
                    features = []
                    # Encode config as features
                    for param in ['llm_temperature', 'llm_top_p', 'risk_aversion',
                                 'tau_for_covariance', 'tau_omega', 'relative_confidence',
                                 'max_weight', 'lookback_days']:
                        if param in config:
                            features.append(float(config[param]))
                        else:
                            features.append(0.0)
                    # Encode allow_shorts as 0/1
                    features.append(1.0 if config.get('allow_shorts', True) else 0.0)
                    
                    X_train.append(features)
                    y_train.append(result[objective])
                
                # Train Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                # Generate candidate configs and predict
                candidates = []
                candidate_configs = []
                for _ in range(50):  # Generate 50 candidates
                    config = {}
                    for param in continuous_space:
                        min_val, max_val = continuous_space[param]
                        config[param] = random.uniform(min_val, max_val)
                    config['allow_shorts'] = random.choice([True, False])
                    
                    # Round to discrete values
                    for param in search_space:
                        if param in config and param in search_space:
                            closest = min(search_space[param], 
                                         key=lambda x: abs(x - config[param]))
                            config[param] = closest
                    
                    # Encode as features
                    features = []
                    for param in ['llm_temperature', 'llm_top_p', 'risk_aversion',
                                 'tau_for_covariance', 'tau_omega', 'relative_confidence',
                                 'max_weight', 'lookback_days']:
                        features.append(float(config.get(param, 0.0)))
                    features.append(1.0 if config.get('allow_shorts', True) else 0.0)
                    
                    candidates.append(features)
                    candidate_configs.append(config)
                
                # Predict and select best candidate
                predictions = rf.predict(candidates)
                best_idx = np.argmax(predictions)
                config = candidate_configs[best_idx]
            
            # Evaluate selected config
            result = self.evaluate_config(config, llm_type, use_cache=True)
            results.append(result)
            if 'error' not in result:
                print(f"  [{i+1}/{n_trials}] {objective}: {result[objective]:.4f}")
        
        return results
    
    def walk_forward_optimization(self, search_space: Dict, 
                                 train_quarters: List[str],
                                 test_quarters: List[str],
                                 optimization_method: str = 'optuna',
                                 n_trials_per_fold: int = 20,
                                 llm_type: str = 'simulated',
                                 objective: str = 'sharpe_ratio') -> Dict:
        """
        Walk-forward optimization: Forward-looking hyperparameter tuning
        
        This method optimizes hyperparameters on training quarters and evaluates
        on future (test) quarters, making it truly forward-looking.
        
        Args:
            search_space: Parameter search space
            train_quarters: Quarters to use for optimization (training)
            test_quarters: Quarters to use for validation (testing - future data)
            optimization_method: Method to use ('optuna', 'bayesian', 'random-forest', 'random')
            n_trials_per_fold: Number of trials per optimization fold
            llm_type: LLM model type
            objective: Objective metric to optimize
            
        Returns:
            Dictionary with walk-forward results
        """
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD OPTIMIZATION (Forward-Looking)")
        print(f"{'='*80}")
        print(f"Training Quarters: {', '.join(train_quarters)}")
        print(f"Test Quarters: {', '.join(test_quarters)}")
        print(f"Method: {optimization_method}")
        print(f"Trials per fold: {n_trials_per_fold}")
        print(f"{'='*80}\n")
        
        # Step 1: Optimize on training quarters
        print("="*80)
        print("PHASE 1: OPTIMIZING ON TRAINING DATA")
        print("="*80)
        
        train_optimizer = HyperparameterOptimizer(quarters=train_quarters)
        
        if optimization_method == 'optuna':
            train_results = train_optimizer.optuna_optimization(
                search_space,
                n_trials=n_trials_per_fold,
                llm_type=llm_type,
                objective=objective,
                sampler='tpe'
            )
        elif optimization_method == 'bayesian':
            train_results = train_optimizer.bayesian_optimization(
                search_space,
                n_trials=n_trials_per_fold,
                llm_type=llm_type,
                objective=objective
            )
        elif optimization_method == 'random-forest':
            train_results = train_optimizer.random_forest_optimization(
                search_space,
                n_trials=n_trials_per_fold,
                llm_type=llm_type,
                objective=objective
            )
        else:  # random
            train_results = train_optimizer.random_search(
                search_space,
                n_trials=n_trials_per_fold,
                llm_type=llm_type
            )
        
        train_analysis = train_optimizer.analyze_results(train_results)
        best_config = train_analysis['best_config']
        
        print(f"\n✓ Best configuration from training:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"✓ Training {objective}: {train_analysis['best_metrics'][objective]:.4f}")
        
        # Step 2: Evaluate on test quarters (forward-looking)
        print(f"\n{'='*80}")
        print("PHASE 2: EVALUATING ON TEST DATA (FORWARD-LOOKING)")
        print(f"{'='*80}")
        
        test_optimizer = HyperparameterOptimizer(quarters=test_quarters)
        test_result = test_optimizer.evaluate_config(best_config, llm_type, use_cache=False)
        
        if 'error' in test_result:
            print(f"✗ Error evaluating on test data: {test_result['error']}")
            return {
                'best_config': best_config,
                'train_metrics': train_analysis['best_metrics'],
                'test_result': test_result,
                'forward_looking_score': -1e6
            }
        
        print(f"\n✓ Test Results (Forward-Looking):")
        print(f"  {objective}: {test_result[objective]:.4f}")
        print(f"  Annualized Return: {test_result['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {test_result['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {test_result['max_drawdown']:.2%}")
        
        # Calculate forward-looking score (test performance)
        forward_looking_score = test_result[objective]
        
        # Step 3: Compare train vs test (check for overfitting)
        train_score = train_analysis['best_metrics'][objective]
        test_score = test_result[objective]
        overfitting_ratio = train_score / test_score if test_score > 0 else float('inf')
        
        print(f"\n{'='*80}")
        print("FORWARD-LOOKING ANALYSIS")
        print(f"{'='*80}")
        print(f"Training {objective}: {train_score:.4f}")
        print(f"Test {objective}: {test_score:.4f}")
        print(f"Overfitting Ratio: {overfitting_ratio:.2f}")
        
        if overfitting_ratio > 1.5:
            print("⚠ Warning: Significant overfitting detected (train >> test)")
        elif overfitting_ratio < 0.8:
            print("✓ Good generalization (test >= train)")
        else:
            print("✓ Reasonable generalization")
        
        return {
            'best_config': best_config,
            'train_metrics': train_analysis['best_metrics'],
            'test_metrics': {
                'annualized_return': test_result['annualized_return'],
                'sharpe_ratio': test_result['sharpe_ratio'],
                'sortino_ratio': test_result['sortino_ratio'],
                'max_drawdown': test_result['max_drawdown'],
                'outperformance': test_result['outperformance']
            },
            'forward_looking_score': forward_looking_score,
            'overfitting_ratio': overfitting_ratio,
            'train_quarters': train_quarters,
            'test_quarters': test_quarters
        }
    
    def analyze_results(self, results: List[Dict], top_n: int = 10) -> Dict:
        """
        Analyze optimization results and find best configurations
        
        Args:
            results: List of evaluation results
            top_n: Number of top configurations to return
            
        Returns:
            Analysis dictionary
        """
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        
        if len(valid_results) == 0:
            return {'error': 'No valid results'}
        
        # Sort by Sharpe ratio (primary) and return (secondary)
        sorted_results = sorted(
            valid_results,
            key=lambda x: (x['sharpe_ratio'], x['annualized_return']),
            reverse=True
        )
        
        top_configs = sorted_results[:top_n]
        
        # Calculate statistics
        returns = [r['annualized_return'] for r in valid_results]
        sharpes = [r['sharpe_ratio'] for r in valid_results]
        
        analysis = {
            'total_trials': len(results),
            'valid_trials': len(valid_results),
            'best_config': top_configs[0]['config'],
            'best_metrics': {
                'annualized_return': top_configs[0]['annualized_return'],
                'sharpe_ratio': top_configs[0]['sharpe_ratio'],
                'sortino_ratio': top_configs[0]['sortino_ratio'],
                'max_drawdown': top_configs[0]['max_drawdown'],
                'outperformance': top_configs[0]['outperformance']
            },
            'top_configs': top_configs,
            'statistics': {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'min_sharpe': np.min(sharpes),
                'max_sharpe': np.max(sharpes)
            }
        }
        
        return analysis
    
    def save_results(self, results: List[Dict], analysis: Dict, 
                    output_dir: str = 'output/hyperparameter_optimization'):
        """
        Save optimization results
        
        Args:
            results: All evaluation results
            analysis: Analysis dictionary
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all results
        results_path = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, f'analysis_{timestamp}.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Saved results to {results_path}")
        print(f"✓ Saved analysis to {analysis_path}")
        
        # Generate visualizations
        try:
            from hyperparameter_visualization import HyperparameterVisualizer
            print(f"\n{'='*80}")
            print("GENERATING VISUALIZATIONS")
            print(f"{'='*80}")
            visualizer = HyperparameterVisualizer(results_dir=output_dir)
            visualizer.create_summary_report({'results': results, 'analysis': analysis})
        except Exception as e:
            print(f"\nWarning: Could not generate visualizations: {e}")
            print("You can generate them later with: python hyperparameter_visualization.py")
        
        # Print summary
        print(f"\n{'='*80}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nBest Configuration:")
        for param, value in analysis['best_config'].items():
            print(f"  {param}: {value}")
        print(f"\nBest Metrics:")
        for metric, value in analysis['best_metrics'].items():
            if isinstance(value, float):
                if 'ratio' in metric.lower() or 'drawdown' in metric.lower():
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")
        print(f"\nStatistics across all trials:")
        stats = analysis['statistics']
        print(f"  Mean Return: {stats['mean_return']:.2%} ± {stats['std_return']:.2%}")
        print(f"  Mean Sharpe: {stats['mean_sharpe']:.4f} ± {stats['std_sharpe']:.4f}")
        print(f"  Return Range: [{stats['min_return']:.2%}, {stats['max_return']:.2%}]")
        print(f"  Sharpe Range: [{stats['min_sharpe']:.4f}, {stats['max_sharpe']:.4f}]")


def main():
    """Run hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--method', type=str, default='random',
                       choices=['grid', 'random', 'bayesian', 'optuna', 'optuna-tpe', 
                               'optuna-random', 'optuna-cmaes', 'random-forest'],
                       help='Search method: grid, random, bayesian (GP), optuna (TPE), '
                            'optuna-random, optuna-cmaes, or random-forest')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of trials for random/ML-based search')
    parser.add_argument('--max-combinations', type=int, default=50,
                       help='Max combinations for grid search')
    parser.add_argument('--objective', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'annualized_return'],
                       help='Objective metric to optimize')
    parser.add_argument('--llm-type', type=str, default='finance-llm',
                       choices=['finance-llm', 'simulated'],
                       help='LLM model type: finance-llm (local AdaptLLM model, default) or simulated (fallback only)')
    parser.add_argument('--quarters', type=str, nargs='+',
                       default=['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'],
                       help='Quarters to use for optimization')
    parser.add_argument('--walk-forward', action='store_true',
                       help='Use walk-forward optimization (forward-looking)')
    parser.add_argument('--train-quarters', type=str, nargs='+',
                       default=['Q1_2024', 'Q2_2024', 'Q3_2024'],
                       help='Training quarters for walk-forward optimization')
    parser.add_argument('--test-quarters', type=str, nargs='+',
                       default=['Q4_2024'],
                       help='Test quarters for walk-forward optimization (future data)')
    
    args = parser.parse_args()
    
    optimizer = HyperparameterOptimizer(quarters=args.quarters)
    search_space = optimizer.define_search_space()
    
    # Walk-forward optimization (forward-looking)
    if args.walk_forward:
        results_dict = optimizer.walk_forward_optimization(
            search_space,
            train_quarters=args.train_quarters,
            test_quarters=args.test_quarters,
            optimization_method=args.method if args.method != 'grid' else 'optuna',
            n_trials_per_fold=args.n_trials,
            llm_type=args.llm_type,
            objective=args.objective
        )
        
        # Save walk-forward results
        os.makedirs('output/hyperparameter_optimization', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join('output/hyperparameter_optimization', 
                                   f'walk_forward_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Saved walk-forward results to {results_path}")
        return
    
    # Standard optimization
    if args.method == 'grid':
        results = optimizer.grid_search(
            search_space,
            llm_type=args.llm_type,
            max_combinations=args.max_combinations
        )
    elif args.method == 'random':
        results = optimizer.random_search(
            search_space,
            n_trials=args.n_trials,
            llm_type=args.llm_type
        )
    elif args.method == 'bayesian':
        results = optimizer.bayesian_optimization(
            search_space,
            n_trials=args.n_trials,
            llm_type=args.llm_type,
            objective=args.objective
        )
    elif args.method.startswith('optuna'):
        sampler = 'tpe'
        if args.method == 'optuna-random':
            sampler = 'random'
        elif args.method == 'optuna-cmaes':
            sampler = 'cmaes'
        
        results = optimizer.optuna_optimization(
            search_space,
            n_trials=args.n_trials,
            llm_type=args.llm_type,
            objective=args.objective,
            sampler=sampler
        )
    elif args.method == 'random-forest':
        results = optimizer.random_forest_optimization(
            search_space,
            n_trials=args.n_trials,
            llm_type=args.llm_type,
            objective=args.objective
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    analysis = optimizer.analyze_results(results)
    optimizer.save_results(results, analysis)


if __name__ == "__main__":
    main()

