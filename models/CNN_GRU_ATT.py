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
from transformers import SwinModel


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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size = 512, key_dim = 2048):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(key_dim, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class AttGRU(nn.Module, PyTorchModelHubMixin):
  def __init__(self, vocab_size: int = 113, dim: int = 2048, hidden_dim: int = 512, dropout_p: int = 0.1) -> None:
      super().__init__()
      self.feature_extractor = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2])

      self.embedding = nn.Embedding(vocab_size, hidden_dim)
      self.attention = BahdanauAttention(hidden_dim, dim)
      self.gru = nn.GRU(dim + hidden_dim, hidden_dim, batch_first=True)
      self.out = nn.Linear(hidden_dim, vocab_size)
      self.dropout = nn.Dropout(dropout_p)

  def forward(self, image, label):
      features = self.feature_extractor(image)
      keys = features.flatten(2).permute(0, 2, 1)
      hidden = torch.zeros(1, image.size(0), 512, device=image.device)
      tgt_embeddings = self.dropout(self.embedding(label[:, :-1]))

      outputs = []
      for t in range(tgt_embeddings.size(1)):
          current_emb = tgt_embeddings[:, t : t + 1, :]
          query = hidden.permute(1, 0, 2)
          context, _ = self.attention(query, keys)
          gru_input = torch.cat([current_emb, context], dim=-1)
          output, hidden = self.gru(gru_input, hidden)
          outputs.append(output)

      outputs = torch.cat(outputs, dim=1)
      logits = self.out(self.dropout(outputs))
      return logits
  
model = AttGRU.from_pretrained("fklska/LaTeX_OCR-AttGRU").to("cpu")
vocab_file_path = hf_hub_download(repo_id="fklska/LaTeX_OCR", filename="token2idx.json")

with open(vocab_file_path, "r", encoding="utf-8") as f:
    saved_token2idx = json.load(f)

tokenizer = LatexTokenizer()
tokenizer.token2idx = saved_token2idx
tokenizer.idx2token = {idx: token for token, idx in saved_token2idx.items()}
tokenizer.vocab = list(saved_token2idx.keys())