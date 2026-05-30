import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import torchvision.models as models
import json
from torchvision.models import resnet50, ResNet50_Weights

class LatexTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.token2idx = {}
        self.idx2token = {}

    def tokenize_formula(self, formula):
        return formula.strip().split()

    def build_vocab(self, formulas):
        tokens_set = set()
        for f in formulas:
            tokens_set.update(self.tokenize_formula(f))

        self.vocab.extend(sorted(list(tokens_set)))
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, formula, max_len=50):
        tokens = self.tokenize_formula(formula)
        tokens = [self.sos_token] + tokens + [self.eos_token]

        indices = [self.token2idx.get(t, self.token2idx[self.unk_token]) for t in tokens]
        if len(indices) < max_len:
            indices += [self.token2idx[self.pad_token]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        tokens = [self.idx2token[int(idx)] for idx in indices]
        return " ".join([t for t in tokens if t not in [self.pad_token, self.sos_token, self.eos_token]])
    
class LaTeXOCR(nn.Module, PyTorchModelHubMixin):
  def __init__(self, vocab_size: int = 113, dim: int = 2048, hidden_dim: int = 512) -> None:
      super().__init__()
      self.feature_extractor = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2])
      self.embedding = nn.Embedding(vocab_size, dim)

      self.init_h = nn.Linear(dim, hidden_dim)
      self.init_c = nn.Linear(dim, hidden_dim)

      self.lstm = nn.LSTM(dim, hidden_dim, batch_first = True)
      self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, image, label):
    features = self.feature_extractor(image)
    img_embeddings = features.mean(dim=[2, 3])
    h0 = self.init_h(img_embeddings).unsqueeze(0)
    c0 = self.init_c(img_embeddings).unsqueeze(0)

    tgt_embeddings = self.embedding(label[:, :-1])

    output, _ = self.lstm(tgt_embeddings, (h0, c0))
    logits = self.fc(output)

    return logits


model = LaTeXOCR.from_pretrained("fklska/LaTeX_OCR").to("cpu")
vocab_file_path = hf_hub_download(repo_id="fklska/LaTeX_OCR", filename="token2idx.json")

with open(vocab_file_path, "r", encoding="utf-8") as f:
    saved_token2idx = json.load(f)

tokenizer = LatexTokenizer()
tokenizer.token2idx = saved_token2idx
tokenizer.idx2token = {idx: token for token, idx in saved_token2idx.items()}
tokenizer.vocab = list(saved_token2idx.keys())