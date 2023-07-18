# This code was based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." youtube video
import requests

BLOCK_SIZE = 8
BATCH_SZE = 4
TRAIN_SIZE = 0.9
from typing import List
import torch
torch.manual_seed(8)

def load_txt(txt_path: str):
    with open(txt_path,'r',encoding='utf-8') as f:
        text = f.read()
    return text

def get_text_info(text:str):
    # Returns all characters and vocab size
    all_chars = sorted(list(set(text)))
    vocab_size = len(all_chars)
    return all_chars, vocab_size

def create_encode(all_chars: List):
    str_to_i = {char:i for i,char in enumerate(all_chars)}
    return lambda str: [str_to_i[char] for char in str]

def create_decode(all_chars: List):
    i_to_char = {i:char for i,char in enumerate(all_chars)}
    return lambda  list_i: [i_to_char[i] for i in list_i]

def create_dataset(data: torch.Tensor, train_size = 0.9):
    n = int(train_size * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data



text = load_txt(f"shakespear.txt")
all_chars, vocab_size = get_text_info(text)
encode = create_encode(all_chars)
data = torch.tensor(encode(text), dtype=torch.long)

train_data, val_data = create_dataset(data, train_size=TRAIN_SIZE)

def generate_batch(data: torch.Tensor, block_size = 8, batch_size = 4):
    # Genereate batch_size number of indexes
    random_index = torch.randint(len(data) - block_size, (batch_size,))    
    
    # Generate context and targets
    # Example:
    # x: [1 2 3 4 5 6]
    # y: [2 3 4 5 6 7]
    x = torch.stack([data[i:i+block_size] for i in random_index])
    y = torch.stack([data[i+1:i+1+block_size] for i in random_index])
    return x,y



x_train,y_train = generate_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SZE)
x_val,y_val = generate_batch(data=val_data, block_size=BLOCK_SIZE, batch_size=BATCH_SZE)

print(x_train.shape)
print(x_train)

print(y_train)