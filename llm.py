# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM")
model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM")
