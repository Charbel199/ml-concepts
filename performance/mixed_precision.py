import torch
from torch import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input text and move to GPU
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Measure time without optimization
start = time.time()
output = model.generate(input_ids, max_new_tokens=50)
end = time.time()

print("Without mixed precision: {:.4f} seconds".format(end - start))
print(tokenizer.decode(output[0], skip_special_tokens=True))


# Measure time with mixed precision
start = time.time()
with autocast("cuda"):
    output = model.generate(input_ids, max_new_tokens=50)
end = time.time()
print("With mixed precision: {:.4f} seconds".format(end - start))

# Decode and print the result
print(tokenizer.decode(output[0], skip_special_tokens=True))
