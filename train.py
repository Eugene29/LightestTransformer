# -*- coding: utf-8 -*-
"""Transformer from Scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_D1IqYhq1zG9fw6O2FzDFdylsotq6-j0
"""

# !pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import requests
import tarfile, lzma, re

## download little shakespear
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# response = requests.get(url)
# if response.status_code == 200:
#     with open("input.txt", "wb") as f:
#         f.write(response.content)
#     print("File downloaded successfully.")
# else:
#     print("Failed to download the file.")
## or
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## hyperparameters
# load_path = "shakespear"

# data loading
def get_batch(split):
  data = train if split=='train' else val
  ix = torch.randint(len(data) - block_size, (batch_size,)) # ix=? batch_size number of indices where we will collect block_size length of sequence each.
  # comma is added after batch_size to represent a tuple and not a scalar.
  x = torch.stack([data[i:i+block_size] for i in ix]) # extracting and then stacking the block_size length of sequences
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(dev), y.to(dev)
  return x, y

# function to estimate our train, val loss
# We created a separate estimate_loss function to evaluate how our model is doing. The regular loss computation at the end of each cycle of steps are
# too stochastic and not reliable because the loss is computed on the last batch. In estimate_loss(), we can also compute the val_loss.
@torch.no_grad() # useful for calculation that we will not take backward() or compute backpropagation on
def estimate_loss():
  out = {}
  # model.eval() or model.train() is needed incase we have batch norm or drop out. Good practice to keep it regardless.
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      _, loss = m(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# Self attention class
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.h = head_size
    self.q = nn.Linear(emb_dim, head_size, bias=False)
    self.k = nn.Linear(emb_dim, head_size, bias=False)
    self.v = nn.Linear(emb_dim, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # we need register_buffer for tensors that are not parameters?
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    B, T, C = x.shape
    wei = self.k(x) @ self.q(x).transpose(-2, -1)  * self.h ** (-1/2) # (B, T, H) @ (B, H, T) => (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # so that it works when generating smaller sentences.
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei) # prohibit some nodes from communicating to build more robust model.
    att_out = wei @ self.v(x)
    return att_out
  
class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(emb_dim, 4 * emb_dim),
      nn.ReLU(),
      nn.Linear(4 * emb_dim, emb_dim),
      nn.Dropout(dropout) # dropout on the residual connection and not the main passage
    )
    
  def forward(self, x):
    return self.net(x)

class MultiHeadAttention(nn.Module):
  """multiple heads of self-attnetion in parallel"""

  def __init__(self, num_heads):
    super().__init__()
    head_size = emb_dim // num_heads
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # nn.ModuleList is used when using having modules inside a list. It just has a couple
    self.proj = nn.Linear(emb_dim, emb_dim) # Linear Layer = Linear Transformation = Projection # This layer helps combine the features learned from mha.
    self.dropout = nn.Dropout(dropout)
    # special features

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out
  
class Block1(nn.Module):
  """Layer block including sa, ffnn"""

  def __init__(self, n_head):
    super().__init__()
    self.sa = MultiHeadAttention(n_head)
    self.ffnn = FeedForward()
    self.ln1 = nn.LayerNorm(emb_dim) # here the shape of the sequence is emb_dim because sequence length is preserved throughout all layers(mha concats)
    self.ln2 = nn.LayerNorm(emb_dim)
  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # This is called pre-norm where we layer_norm before our transformation. 
    x = x + self.ffnn(self.ln2(x))
    return x 

# bigram model
class BinaryGramLanguageModel(nn.Module):
  def __init__(self, load_path):
    super().__init__()
    data_bank = dict({0: "shakespear", 2: 'quotes'})
    if load_path == "quotes.pth":
      batch_size = 64 # num of sequences to compute in parallel per batch
      block_size = 256 # max_len of each sequence
      emb_dim = 408
      lr = 3e-4
      max_iters = 8500
      eval_interval = 500 # when to estimate loss
      eval_iters = 200 # iteration to estimate loss
      n_head = 6 # num of heads in mha
      n_layer = 6 # num of blocks(mha + ffnn)
      dropout = 0.2 # dropout rate
      choose_data = data_bank['quotes']
    else:
      batch_size = 64 # num of sequences to compute in parallel per batch
      block_size = 256 # max_len of each sequence
      emb_dim = 384
      lr = 3e-4
      max_iters = 4001
      eval_interval = 500 # when to estimate loss
      eval_iters = 200 # iteration to estimate loss
      n_head = 6 # num of heads in mha
      n_layer = 6 # num of blocks(mha + ffnn)
      dropout = 0.2 # dropout rate
      choose_data = data_bank['shakespear']

    #----------------
    ## extract and read for tar.xz files
    def extract_and_read(file_path):
      with tarfile.open(file_path, "r") as tar:
        first_fname = tar.getmembers()[0]
        print(f'unzipping tar file: {first_fname}')
        extracted_f = tar.extractfile(first_fname)
        text = extracted_f.read()#.decode('utf-8')
      return text
    if choose_data == data_bank[0]:
      with open("lilshakespear.txt", "r") as f: # Change input name...
        text = f.read()
    elif choose_data == data_bank[1]:
      fname = 'openwebtext.tar.xz'
      text = extract_and_read(fname)
      def decompress_xz_file(file_path):
        with lzma.open(file_path, 'rt', encoding='utf-8') as f:  # 'rt' mode opens it in text mode
            return f.read()

      ## workon this one day
      def filter_content(text):
        # Split the text by patterns that look like "0000068-1ad7382d9e2e97d93b002a64642ab65d.txt0000644..."
        patterns = [
              r'\s+0000644\s+0+\s+0+\s+000000\d+\s+0+\s+\d+\s+\d\s+',
              r'\s+ustar\s+',
              r'\s+0+\s+0+\s+'
          ]
        patterns2 = r'.+\.txt'
        # chunks = re.sub(r'.+\.txt', text)
        for pattern in patterns:
          text = re.sub(pattern, "", text)
        text = re.sub(patterns2, "", text)
        # chunks = re.split(r'.+\.txt', text)
        # Filter out short or undesired chunks
        # return ''.join([chunk for chunk in chunks if len(chunk) > 50])  # 50 is just an example, adjust as needed
        return text
      file_path = 'openwebtext/urlsf_subset00-1_data.xz'
      text = decompress_xz_file(file_path)
      file_path2 = 'openwebtext/urlsf_subset00-2_data.xz'
      text2 = decompress_xz_file(file_path2)
      text += text2
      filtered = filter_content(text)
      text = filtered.replace("\x00", " ")
      print(len(text))
      print(text[:100000])
      # test_str = '                                                        0000644 0000000 0000000 00000013365 00000000000 015042  0                                                                                                    ustar                                                                   0000000 0000000                                                                                                                                                                        '
      # filter_content(test_str)
    elif choose_data == data_bank[2]:
      with open("train.txt", 'r', encoding='utf-8') as f:
        text = f.read()
      with open("valid.txt", 'r', encoding='utf-8') as f:
        text += f.read()
      with open("test.txt", 'r', encoding='utf-8') as f:
        text += f.read()

    dev = "cuda" if torch.cuda.is_available() else "cpu" # gpu or cpu here
    print(dev)
    # dev = xm.xla_device() # This is for TPU
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i, ch in enumerate(chars)} # strings to integers. Used for character level Tokenizer
    itos = {i:ch for ch, i in stoi.items()} # inverse dict for stoi
    encode = lambda text: [stoi[ch] for ch in text]
    decode = lambda encoded: "".join([itos[i] for i in encoded])
    data = torch.tensor(encode(text))
    print(f'length of data trained on: {len(data)}')
    ## train/val split 90%
    n = int(0.9 * len(text))
    train = data[:n]
    val = data[n:]

    self.token_embedding_table = nn.Embedding(vocab_size, emb_dim) # one_hot to word2vec (B, T, emb)
    self.position_embedding_table = nn.Embedding(block_size, emb_dim) # encoding positions (B, T, P, emb)
    self.blocks = nn.Sequential(*[Block1(n_head) for _ in range(n_layer)])
    self.lnf = nn.LayerNorm(emb_dim)
    self.lm_head = nn.Linear(emb_dim, vocab_size) # emb to vocab_size (B, T, P, Voc)

  def forward(self, idx, targets=None):  # forward is the default function called when you call this class.
    '''
    idx: token id of inputs (B, T)
    targets: token id of outputs (B, T)
    returns
    logits: supposed to be likelihood scoe but is predictions??
    loss: cross entropy loss
    '''
    B, T = idx.shape

    word_emb = self.token_embedding_table(idx) # (B, T) => (B, T, w_Emb) where idx = tokenized_id\
    pos_emb = self.position_embedding_table(torch.arange(T, device=dev)) # (T) => (T, p_Emb)
    x = word_emb + pos_emb # B, T, Emb 
    x = self.blocks(x)
    x = self.lnf(x)
    logits = self.lm_head(x) # (B, T, Vocab)

    # The intuition for above would be like having a list of logit score distributions per target token.
    if targets==None:
      loss = 0
    else:
      ## below calculates the loss
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # has C dimension due to embedding. We combine the B and T because pytorch function wants it flattened out.
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  def generate(self, idx, max_new_tokens):
    '''
    idx: tokenized ids in batches (B, T)
    max_new_tokens: max new tokens you want to generate (int)
    returns
    concatnated tokens with generated extentions. (B, T+max_new_tokens)
    '''
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      # idx_cond = idx
      logits, loss = self(idx_cond) # calls the forward function as default.
      logits = logits[:, -1, :] # only preversing the probability score distribution of the last time step.
      probs = F.softmax(logits, dim=-1) # since logits = (B, C)
      generated = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, generated), dim=1) # dim=1 because we want (B, T+1)
    return idx

if __name__ == "__main__":
  model = BinaryGramLanguageModel('quotes.pth')
  m = model.to(dev) # moving the model to the device. Moving the model also moves the weights assigned to the device to expediate the compute.
  optimizer = torch.optim.AdamW(m.parameters(), lr=lr) # typical lr is 0.001 but we can get away with higher for models that are smaller.
  ## training
  for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
      losses = estimate_loss()
      print(f"step {iter}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
    xb, yb = get_batch('train')  # sample a batch randomly from train
    # evaluate loss
    logits, loss = m(xb, yb) # model m was initialized in the previous code
    optimizer.zero_grad(set_to_none=True) # clearing previous gradient calculations. Using 'set_to_none=True' helps garbage collecter in between each step
    # clear out the gradients. Quite trivial, but might provide subtle help for a larger model.
    loss.backward() # calculating the backprob to get the gradients
    optimizer.step() # updating the parameters with gradients

  # generating again after some training. We also want to generate tensors inside the device.
  print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=dev), max_new_tokens=300)[0].tolist()))
  # Logs:
  ## word embedding only
  # step 4400: train_loss 2.5221, val_loss 2.5416

  ## Word_emb + Linear
  # step 4400: train_loss 2.4572, val_loss 2.4861

  ## Word_emb + Pos_emb + Linear # positioning embedding made bigram worse!
  # step 4600: train_loss 2.4722, val_loss 2.4933

  ## Word emb + Pos_emb + Self_att + Linear
  # step 4600: train_loss 2.4872, val_loss 2.4873

  ## Word emb + Pos_emb + Multihead_att + Linear
  # step 4600: train_loss 2.1787, val_loss 2.1919

  ## scaled up
  # around 1.5?

  ## small scaled QK^T
  # step 5000: train_loss 2.3458, val_loss 2.3581
  ## small scaled KQ^T
  # step 5000: train_loss 2.3352, val_loss 2.3456
  ## conclusion: The roles of Q and K change but the overall performance stays the same.

  ## save model
  # torch.save(m.state_dict(), "model.pth")