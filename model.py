import torch
from train import BinaryGramLanguageModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
load_path = "model.pth" ## for shakespear
# load_path = "quotes.pth" ## for random philosphic quote
model = BinaryGramLanguageModel(load_path, dev)
model.load_state_dict(torch.load(load_path))
model.to(dev)
model.eval()
## Generate - this may take a while depending on your computer
# you can change max_new_tokens to change genereated text length
print(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=dev), max_new_tokens=600))