#!/usr/bin/env python3
"""
Update config.yaml with best hyperparameters from optimization results
"""

import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def find_latest_analysis_file(results_dir: str = 'output/hyperparameter_optimization') -> Optional[str]:
    """Find the most recent analysis file"""
    if not os.path.exists(results_dir):
        return None
    
    analysis_files = [
        f for f in os.listdir(results_dir) 
        if f.startswith('analysis_') and f.endswith('.json')
    ]
    
    if not analysis_files:
        return None
    
    # Sort by timestamp in filename
    analysis_files.sort(reverse=True)
    return os.path.join(results_dir, analysis_files[0])


def load_best_config(analysis_file: Optional[str] = None) -> Dict:
    """
    Load best configuration from optimization results
    
    Args:
        analysis_file: Path to analysis JSON file. If None, uses latest.
        
    Returns:
        Dictionary with best configuration
    """
    if analysis_file is None:
        analysis_file = find_latest_analysis_file()
    
    if analysis_file is None or not os.path.exists(analysis_file):
        raise FileNotFoundError(
            "No optimization analysis file found. "
            "Run hyperparameter optimization first."
        )
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    if 'error' in analysis:
        raise ValueError(f"Analysis file contains error: {analysis['error']}")
    
    return analysis['best_config']


def update_config_yaml(best_config: Dict, config_path: str = 'config.yaml', 
                       backup: bool = True) -> None:
    """
    Update config.yaml with best hyperparameters
    
    Args:
        best_config: Best configuration dictionary from optimization
        config_path: Path to config.yaml
        backup: Whether to create a backup before updating
    """
    config_path = Path(config_path)
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Create backup
        if backup:
            backup_path = config_path.with_suffix('.yaml.backup')
            with open(backup_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"✓ Created backup: {backup_path}")
    else:
        config = {}
        print(f"⚠ Config file not found at {config_path}, creating new one")
    
    # Map optimization config to config.yaml structure
    # Update Black-Litterman hyperparameters
    if 'black_litterman' not in config:
        config['black_litterman'] = {}
    
    if 'risk_aversion' in best_config:
        config['black_litterman']['risk_aversion'] = round(best_config['risk_aversion'], 3)
    
    if 'tau_for_covariance' in best_config:
        config['black_litterman']['tau_for_covariance'] = round(best_config['tau_for_covariance'], 4)
    
    if 'tau_omega' in best_config:
        config['black_litterman']['tau_omega'] = round(best_config['tau_omega'], 4)
    
    if 'relative_confidence' in best_config:
        config['black_litterman']['relative_confidence'] = round(best_config['relative_confidence'], 3)
    
    if 'allow_shorts' in best_config:
        config['black_litterman']['allow_shorts'] = best_config['allow_shorts']
    
    if 'max_weight' in best_config:
        config['black_litterman']['max_weight'] = round(best_config['max_weight'], 2)
    
    # Update data settings
    if 'data' not in config:
        config['data'] = {}
    
    if 'lookback_days' in best_config:
        config['data']['lookback_days'] = int(best_config['lookback_days'])
    
    # Update LLM settings
    if 'llm' not in config:
        config['llm'] = {}
    
    if 'llm_temperature' in best_config:
        config['llm']['temperature'] = round(best_config['llm_temperature'], 3)
    
    if 'llm_top_p' in best_config:
        config['llm']['top_p'] = round(best_config['llm_top_p'], 3)
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                 allow_unicode=True)
    
    print(f"✓ Updated {config_path} with best hyperparameters")


def print_best_config(best_config: Dict, best_metrics: Dict) -> None:
    """Print best configuration in a readable format"""
    print("\n" + "="*80)
    print("BEST CONFIGURATION FROM OPTIMIZATION")
    print("="*80)
    
    print("\nBlack-Litterman Hyperparameters:")
    print(f"  risk_aversion:         {best_config.get('risk_aversion', 'N/A'):>8.3f}")
    print(f"  tau_for_covariance:    {best_config.get('tau_for_covariance', 'N/A'):>8.4f}")
    print(f"  tau_omega:             {best_config.get('tau_omega', 'N/A'):>8.4f}")
    print(f"  relative_confidence:   {best_config.get('relative_confidence', 'N/A'):>8.3f}")
    print(f"  allow_shorts:          {best_config.get('allow_shorts', 'N/A')}")
    print(f"  max_weight:            {best_config.get('max_weight', 'N/A'):>8.2f}")
    
    print("\nData Settings:")
    print(f"  lookback_days:         {best_config.get('lookback_days', 'N/A'):>8.0f}")
    
    print("\nLLM Settings:")
    print(f"  temperature:           {best_config.get('llm_temperature', 'N/A'):>8.3f}")
    print(f"  top_p:                 {best_config.get('llm_top_p', 'N/A'):>8.3f}")
    
    print("\nPerformance Metrics:")
    for metric, value in best_metrics.items():
        if isinstance(value, float):
            if 'ratio' in metric.lower() or 'drawdown' in metric.lower():
                print(f"  {metric:25s}: {value:8.4f}")
            else:
                print(f"  {metric:25s}: {value:8.2%}")
        else:
            print(f"  {metric:25s}: {value}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update config.yaml with best hyperparameters from optimization'
    )
    parser.add_argument('--analysis-file', type=str, default=None,
                       help='Path to analysis JSON file (default: latest)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml file')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of config.yaml')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without actually updating')
    parser.add_argument('--show-only', action='store_true',
                       help='Only show best config, do not update')
    
    args = parser.parse_args()
    
    try:
        # Load best configuration
        analysis_file = args.analysis_file or find_latest_analysis_file()
        
        if analysis_file is None:
            print("Error: No optimization analysis file found.")
            print("Run hyperparameter optimization first:")
            print("  python hyperparameter_optimization.py --method optuna --n-trials 100")
            return 1
        
        print(f"Loading best configuration from: {analysis_file}")
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        best_config = analysis['best_config']
        best_metrics = analysis.get('best_metrics', {})
        
        # Print best configuration
        print_best_config(best_config, best_metrics)
        
        if args.show_only:
            return 0
        
        if args.dry_run:
            print("\n" + "="*80)
            print("DRY RUN - Would update config.yaml with above values")
            print("="*80)
            print("\nTo actually update, run without --dry-run flag")
            return 0
        
        # Update config.yaml
        print("\n" + "="*80)
        print("UPDATING CONFIG.YAML")
        print("="*80)
        
        update_config_yaml(
            best_config,
            config_path=args.config,
            backup=not args.no_backup
        )
        
        print("\n✓ Config updated successfully!")
        print("\nNext steps:")
        print("  1. Review the updated config.yaml")
        print("  2. Run main pipeline: python main.py")
        print("  3. The pipeline will now use the optimized hyperparameters")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

