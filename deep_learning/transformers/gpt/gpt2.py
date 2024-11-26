from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

# Load pretrained model and tokenizer
model_name = "gpt2"  # You can replace this with a larger model if desired
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset (Shakespeare text as an example)
shakespeare_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = Dataset.from_dict({"text": [shakespeare_text] * 100})  # Repeat for a larger dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert tokenized dataset to PyTorch format
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./shakespeare-finetuned-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./shakespeare-finetuned-gpt2")
tokenizer.save_pretrained("./shakespeare-finetuned-gpt2")

# Generate text with the fine-tuned model
model.eval()
input_text = "To be, or not to be"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

if torch.cuda.is_available():
    model.to("cuda")
    input_ids = input_ids.to("cuda")

output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

print("Generated Text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
