# This code was based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." youtube video

BLOCK_SIZE = 256
BATCH_SZE = 64
TRAIN_SIZE = 0.9
from typing import List
import torch
torch.manual_seed(8)
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda'


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
    x, y = x.to(device), y.to(device)
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
C -> Channels = Vocab size in this case

T -> Context for bigram model isnt very useful as the Embedding layer only looks at a single token at a time

'''


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embed, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, is_decoder = True):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias = False)
        self.q = nn.Linear(n_embed, head_size, bias = False)
        self.v = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.is_decoder = is_decoder

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        B, T, C = x.shape
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        if self.is_decoder:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class FeedForward(nn.Module):
    # This is independent of the context, each token/embdedding is processed independently
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block: MultiHeadAttention -> FeedForward, communication -> computation
    def __init__(self, n_heads, n_embed, block_size):
        super().__init__()
        head_size = n_embed // n_heads
        self.multi_head_attention = MultiHeadAttention(n_heads, n_embed, head_size, block_size)
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # Every n_embed tokens get layer normalized
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x+self.multi_head_attention(self.ln1(x))
        x = x+self.feed_forward(self.ln2(x))
        return x

class LayerNorm1d: # Layernorm makes sure that rows have a mean of 0 and std of 1, batchnorm does the same but for the column

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size,n_head, n_embed, n_layer=8):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # STEP 1
        #self.attention_head = Head(n_embed, head_size=32, block_size=block_size)

        # STEP 2
        # self.multi_head_attention = MultiHeadAttention(n_heads=4, n_embed=n_embed, head_size=n_embed//4, block_size=block_size)
        # self.feed_forward = FeedForward(n_embed)
        
        # STEP 3
        self.blocks = nn.Sequential(*[Block(n_heads=n_head, n_embed=n_embed, block_size=block_size) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        position_embed = self.position_embedding_table(torch.arange(T, device = device))
        x = token_embed + position_embed

        #x = self.attention_head(x)
        #x = self.multi_head_attention(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
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

model = BigramLanguageModel(vocab_size,block_size=BLOCK_SIZE, n_embed=384,n_head=6, n_layer=6)
model.to(device)



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


eval_iters = 200
eval_interval = 3000
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = generate_batch(data=val_data, block_size=BLOCK_SIZE, batch_size=BATCH_SZE)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


max_iters = 5000
for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = generate_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SZE)


    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"Done with iter: {iter}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
res = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(''.join(res))