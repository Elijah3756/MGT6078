"""
LLM Model Loader (Legacy)
This file is kept for backward compatibility but model loading is now handled
in llm/analyzer.py. Importing this module no longer triggers automatic model loading.
"""

# Model loading is now lazy and handled in LLMAnalyzer class
# This file exists for backward compatibility only

if __name__ == "__main__":
    # Only load model if run directly (for testing)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Loading AdaptLLM/finance-LLM model...")
    tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM")
    model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM")
    print("Model loaded successfully")
