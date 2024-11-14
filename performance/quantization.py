import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input text
input_text = "Hello, how are you? What's your name?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 1. Run inference in full precision
start = time.time()
output = model.generate(input_ids, max_new_tokens=50)
end = time.time()
print("Full precision: {:.4f} seconds".format(end - start))
print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
# 2. Apply dynamic quantization to the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Run inference with quantized model
start = time.time()
output_quantized = quantized_model.generate(input_ids, max_new_tokens=50)
end = time.time()
print("Quantized model: {:.4f} seconds".format(end - start))

# Decode and print the result to check quality
print("Output with quantization:", tokenizer.decode(output_quantized[0], skip_special_tokens=True))
