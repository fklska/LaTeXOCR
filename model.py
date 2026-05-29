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
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_mapping = nn.Linear(encoder_dim, attention_dim)
        self.decoder_mapping = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_hidden):
        enc_proj = self.encoder_mapping(encoder_outputs)
        
        dec_proj = self.decoder_mapping(decoder_hidden).unsqueeze(1)
        
        scores = self.full_attention(torch.tanh(enc_proj + dec_proj))
        
        scores = scores.squeeze(2)
        alphas = F.softmax(scores, dim=-1)
        
        context_vector = torch.sum(encoder_outputs * alphas.unsqueeze(2), dim=1)
        
        return context_vector, alphas
    

class AttentionResNetLaTeXModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, emb_dim=256, decoder_dim=512, attention_dim=256):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_dim = 512
        
        self.attention = BahdanauAttention(
            encoder_dim=self.encoder_dim, 
            decoder_dim=decoder_dim, 
            attention_dim=attention_dim
        )
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.lstm_cell = nn.LSTMCell(input_size=emb_dim + self.encoder_dim, hidden_size=decoder_dim)
        
        self.fc_out = nn.Linear(decoder_dim, vocab_size)
        
        self.init_h = nn.Linear(self.encoder_dim, decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, decoder_dim)
        
        self.decoder_dim = decoder_dim

    def _init_hidden_state(self, encoder_outputs):
        mean_encoder_outputs = encoder_outputs.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_outputs))
        c = torch.tanh(self.init_c(mean_encoder_outputs))
        return h, c

    def forward(self, images, formulas):
        batch_size = images.size(0)
        seq_len = formulas.size(1)
        
        encoder_outputs = self.backbone(images).flatten(2).permute(0, 2, 1)
        
        h, c = self._init_hidden_state(encoder_outputs)
        
        predictions = torch.zeros(batch_size, seq_len - 1, self.fc_out.out_features).to(images.device)
        
        embeddings = self.embedding(formulas)
        
        for t in range(seq_len - 1):
            context, _ = self.attention(encoder_outputs, h)
            
            current_emb = embeddings[:, t, :]
            
            lstm_input = torch.cat([current_emb, context], dim=1)
            
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            preds = self.fc_out(h)
            predictions[:, t, :] = preds
            
        return predictions
    

model = AttentionResNetLaTeXModel.from_pretrained("fklska/LaTeX_OCR")
vocab_file_path = hf_hub_download(repo_id="fklska/LaTeX_OCR", filename="token2idx.json")

with open(vocab_file_path, "r", encoding="utf-8") as f:
    saved_token2idx = json.load(f)

tokenizer = LatexTokenizer()
tokenizer.token2idx = saved_token2idx
tokenizer.idx2token = {idx: token for token, idx in saved_token2idx.items()}
tokenizer.vocab = list(saved_token2idx.keys())