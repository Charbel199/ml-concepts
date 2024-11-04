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


txt_file_path = "shakespear.txt"
print(f"Loading txt file: {txt_file_path}")
text = load_txt(txt_file_path)
all_chars, vocab_size = get_text_info(text)
print(f"All chars are {all_chars}, vocab size is: {vocab_size} ")
encode = create_encode(all_chars)
decode = create_decode(all_chars)
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

'''
Very simple bigram model which looks at a letter and predicts the next one.

B -> Batch size
T -> Context
C -> Channels = Vocab size

T -> Context for bigram model isnt very useful as the Embedding layer only looks at a single token at a time

'''
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
           # print(f"Current idx: {idx}")
            # get the predictions
            logits, loss = self(idx)
          #  print(f"Logits shape: {logits.shape}")
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
          #  print(f"Logits after reshape: {logits.shape}")
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
           # print(f"probs: {probs.shape}")
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
           # print(f"Idx next is: {idx_next}")
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(x_train, y_train)
print(logits.shape)
print(loss)

res = decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())

print(''.join(res))



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb,yb = generate_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SZE)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

res = decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())

print(''.join(res))
