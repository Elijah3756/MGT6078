"""
LLM Analyzer
Main interface for LLM-based portfolio analysis
"""

import os
import json
from typing import Dict, Optional
from llm.prompts import SYSTEM_PROMPT, create_user_prompt
from llm.views_parser import ViewsParser


class LLMAnalyzer:
    """Analyze financial data and generate investment views using local AdaptLLM/Finance-LLM model"""
    
    def __init__(self, model_type="finance-llm", temperature=0.7, top_p=0.9):
        """
        Initialize LLM Analyzer
        
        Args:
            model_type: 'finance-llm' (default) or 'simulated' (fallback only)
            temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
            top_p: Nucleus sampling parameter (0.0-1.0)
        """
        # Only support finance-llm (local model) or simulated (fallback)
        if model_type not in ["finance-llm", "simulated"]:
            print(f"Warning: {model_type} not supported. Using finance-llm.")
            model_type = "finance-llm"
        
        self.model_type = model_type
        self.temperature = temperature
        self.top_p = top_p
        self.parser = ViewsParser()
        self.finance_model = None
        self.finance_tokenizer = None
        self._model_load_attempted = False
        self._model_load_failed = False
        
        # Check if dependencies are available (but don't load model yet)
        if model_type == "finance-llm":
            self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available without loading the model"""
        try:
            import transformers
            import torch
            self._dependencies_available = True
        except ImportError:
            print("Warning: transformers or torch not available. Finance-LLM model cannot be used.")
            print("Falling back to simulated mode")
            self._dependencies_available = False
            self.model_type = "simulated"
            self._model_load_failed = True
    
    def _load_finance_llm(self):
        """Lazy load the AdaptLLM/finance-LLM model (only when actually needed)"""
        # Don't attempt if already tried and failed
        if self._model_load_failed:
            return False
        
        # Don't attempt if dependencies not available
        if not self._dependencies_available:
            return False
        
        # Don't reload if already loaded
        if self.finance_model is not None and self.finance_tokenizer is not None:
            return True
        
        # Mark that we're attempting to load
        if self._model_load_attempted:
            return False  # Already attempted, don't retry
        
        self._model_load_attempted = True
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print("Loading AdaptLLM/finance-LLM model (lazy loading)...")
            print("  This may take a few minutes on first use (~7GB download)...")
            
            self.finance_tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM")
            self.finance_model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.finance_model = self.finance_model.cuda()
                print("  Model loaded on GPU")
            else:
                print("  Model loaded on CPU (slower)")
            
            print("✓ Finance-LLM model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"Error: Required dependencies not available: {e}")
            print("Install with: pip install transformers torch")
            print("Falling back to simulated mode")
            self._model_load_failed = True
            self.model_type = "simulated"
            return False
            
        except Exception as e:
            print(f"Error loading finance-LLM model: {e}")
            print("This may be due to:")
            print("  - Insufficient memory/VRAM")
            print("  - Network issues downloading the model")
            print("  - Missing dependencies")
            print("Falling back to simulated mode")
            self._model_load_failed = True
            self.model_type = "simulated"
            return False
    
    def generate_views_finance_llm(self, quarter_data: Dict) -> Optional[Dict]:
        """
        Generate views using AdaptLLM/finance-LLM model
        
        Args:
            quarter_data: Dictionary with quarterly data
            
        Returns:
            Views dictionary or None if failed
        """
        # Lazy load the model only when actually needed
        if self.finance_model is None or self.finance_tokenizer is None:
            if not self._load_finance_llm():
                return None  # Model loading failed, fallback will be used
        
        try:
            import torch
            
            # Create a condensed prompt for the local model (it has smaller context window)
            user_prompt = create_user_prompt(quarter_data)
            
            # Truncate if too long (finance-LLM has limited context)
            max_input_chars = 2000
            if len(user_prompt) > max_input_chars:
                user_prompt = user_prompt[:max_input_chars] + "\n\n[Content truncated for processing]"
            
            # Combine system and user prompts
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nOutput your investment views in JSON format:"
            
            # Tokenize and generate
            inputs = self.finance_tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finance_model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=self.top_p,
                    pad_token_id=self.finance_tokenizer.eos_token_id
                )
            
            response_text = self.finance_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response_text = response_text[len(full_prompt):]
            
            # Save raw response for debugging
            os.makedirs('output/llm_responses', exist_ok=True)
            with open(f'output/llm_responses/{quarter_data["quarter"]}_finance_llm_response.txt', 'w') as f:
                f.write(response_text)
            
            # Parse response
            views = self.parser.parse_and_validate(response_text, quarter_data['quarter'])
            views['method'] = 'finance-llm'
            
            return views
            
        except Exception as e:
            print(f"Error generating views with finance-LLM: {e}")
            return None
    
    def generate_views_simulated(self, quarter_data: Dict) -> Dict:
        """
        Generate simulated views based on historical returns and simple heuristics
        This is used as a fallback when finance-llm model fails to load
        
        Args:
            quarter_data: Dictionary with quarterly data
            
        Returns:
            Views dictionary
        """
        print("Using simulated views generation (fallback mode)")
        
        quarter = quarter_data['quarter']
        views = []
        
        # Generate views based on historical returns with some noise
        import numpy as np
        np.random.seed(hash(quarter) % 2**32)
        
        for ticker in ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']:
            ticker_info = quarter_data['tickers'].get(ticker, {})
            hist_return = ticker_info.get('historical_return', 0.0)
            
            # Use momentum: if last quarter was positive, expect positive (but regress to mean)
            expected_return = hist_return * 0.4 + np.random.normal(0.02, 0.03)
            expected_return = max(-0.15, min(0.15, expected_return))
            
            # Confidence varies with data availability
            headlines_count = len(ticker_info.get('bloomberg_headlines', []))
            confidence = min(70, 40 + headlines_count)
            
            # Generate reasoning
            if hist_return > 0.05:
                reasoning = f"Strong momentum from previous quarter (+{hist_return:.1%}). Expecting continued but moderated growth."
            elif hist_return < -0.05:
                reasoning = f"Recovery expected after previous decline ({hist_return:.1%}). Mean reversion likely."
            else:
                reasoning = f"Stable performance expected based on historical patterns and current market conditions."
            
            views.append({
                'ticker': ticker,
                'expected_return': round(expected_return, 4),
                'confidence': int(confidence),
                'reasoning': reasoning
            })
        
        return {
            'quarter': quarter,
            'views': views,
            'method': 'simulated'
        }
    
    
    def generate_views(self, quarter_data: Dict) -> Dict:
        """
        Generate investment views for a quarter using local AdaptLLM/Finance-LLM model
        
        Args:
            quarter_data: Dictionary with all quarterly data
            
        Returns:
            Views dictionary with expected returns and confidence levels
        """
        print(f"\nGenerating investment views for {quarter_data['quarter']}...")
        print(f"Method: {self.model_type}")
        
        views = None
        
        # Try finance-llm first (local model)
        if self.model_type == "finance-llm":
            views = self.generate_views_finance_llm(quarter_data)
        
        # Fallback to simulated if finance-llm failed or simulated was requested
        if views is None or self.model_type == "simulated":
            if self.model_type == "finance-llm" and views is None:
                print("Finance-LLM model failed, falling back to simulated mode")
            views = self.generate_views_simulated(quarter_data)
        
        # Save views
        self.save_views(views)
        
        print(f"✓ Generated views for {len(views['views'])} tickers")
        
        return views
    
    def save_views(self, views: Dict):
        """
        Save generated views to file
        
        Args:
            views: Views dictionary
        """
        os.makedirs('output/views', exist_ok=True)
        
        output_path = f'output/views/{views["quarter"]}_views.json'
        with open(output_path, 'w') as f:
            json.dump(views, f, indent=2)
        
        print(f"  Saved views to {output_path}")
    
    def print_views_summary(self, views: Dict):
        """
        Print a summary of generated views
        
        Args:
            views: Views dictionary
        """
        print(f"\n{'='*80}")
        print(f"INVESTMENT VIEWS SUMMARY - {views['quarter']}")
        print(f"{'='*80}")
        print(f"Method: {views.get('method', 'unknown')}\n")
        
        for view in views['views']:
            print(f"{view['ticker']:4s}: {view['expected_return']:+7.2%} "
                  f"(confidence: {view['confidence']:2d}%) - {view['reasoning'][:60]}...")


def main():
    """Test LLM analyzer"""
    from data.loader import DataLoader
    
    loader = DataLoader()
    analyzer = LLMAnalyzer(model_type="finance-llm")
    
    # Load Q1 2024 data
    data = loader.load_quarterly_data('Q1_2024')
    
    # Generate views
    views = analyzer.generate_views(data)
    
    # Print summary
    analyzer.print_views_summary(views)


if __name__ == "__main__":
    main()

