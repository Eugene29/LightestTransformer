import torch
from train import BinaryGramLanguageModel, decode
dev = "cuda" if torch.cuda.is_available() else "cpu"
dev
# load_path = "model.pth" ## for shakespear
load_path = "quotes.pth" ## for random philosphic quote
model = BinaryGramLanguageModel()
model.load_state_dict(torch.load(load_path))
model.to(dev)
model.eval()
## Generate
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=dev), max_new_tokens=300)[0].tolist()))