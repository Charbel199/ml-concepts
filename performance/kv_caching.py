import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input text
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 1. Run without KV cache
start = time.time()
output_no_cache = model.generate(input_ids, max_new_tokens=5, use_cache=False)
end = time.time()
print("Without KV caching: {:.4f} seconds".format(end - start))
print("Output without KV cache:", tokenizer.decode(output_no_cache[0], skip_special_tokens=True))

# 2. Run with KV cache
# Generate initial output and store past_key_values for caching
start = time.time()
outputs = model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values  # Cache generated during forward pass
# Generate new tokens with KV cache
new_input_ids = tokenizer(" The quick brown fox ", return_tensors="pt").input_ids
output_with_cache = model(new_input_ids, past_key_values=past_key_values, use_cache=True)
end = time.time()
print("With KV caching: {:.4f} seconds".format(end - start))
print("Output with KV cache:", tokenizer.decode(output_with_cache.logits.argmax(-1)[0]))

# Display the generated tokens for each case
print("\nGenerated Tokens without KV cache:", tokenizer.decode(output_no_cache[0], skip_special_tokens=True).split())
print("Generated Tokens with KV cache:", tokenizer.decode(output_with_cache.logits.argmax(-1)[0]).split())
