"""
Configuration Loader
Loads configuration from config.yaml and provides defaults
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """Get the project root directory (parent of this file's directory)"""
    return Path(__file__).resolve().parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml
    
    Args:
        config_path: Optional path to config file. If None, uses config.yaml in project root.
        
    Returns:
        Configuration dictionary with defaults applied
    """
    project_root = get_project_root()
    
    if config_path is None:
        config_path = project_root / 'config.yaml'
    else:
        config_path = Path(config_path)
    
    # Default configuration
    default_config = {
        'llm': {
            'model_type': 'finance-llm'
        },
        'portfolio': {
            'tickers': ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF'],
            'initial_capital': 100000
        },
        'black_litterman': {
            'risk_aversion': 2.5,
            'tau_for_covariance': 0.025,
            'tau_omega': 0.05,
            'relative_confidence': 1.0,
            'allow_shorts': True,
            'min_weight': -1.0,
            'max_weight': 2.0
        },
        'data': {
            'lookback_days': 189,  # 9 months - optimal for quarterly rebalancing
            'quarters': ['Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024', 'Q1_2025', 'Q2_2025', 'Q3_2025']
        },
        'output': {
            'charts': True,
            'reports': True,
            'json': True,
            'directories': {
                'portfolios': 'output/portfolios',
                'views': 'output/views',
                'charts': 'output/charts',
                'reports': 'output/reports',
                'llm_responses': 'output/llm_responses'
            }
        },
        'performance': {
            'risk_free_rate': 0.02,
            'var_confidence': 0.95
        },
        'reporting': {
            'include_methodology': True,
            'include_quarterly_breakdown': True,
            'include_allocation_history': True
        }
    }
    
    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Deep merge defaults with file config
            config = _deep_merge(default_config, file_config)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}. Using defaults.")
            config = default_config
    else:
        print(f"Warning: config.yaml not found at {config_path}. Using defaults.")
        config = default_config
    
    # Add project root to config
    config['project_root'] = str(project_root)
    
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# Global config instance (lazy-loaded)
_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Get the global configuration (singleton pattern)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Reload configuration (useful for testing)"""
    global _config
    _config = load_config(config_path)
    return _config

