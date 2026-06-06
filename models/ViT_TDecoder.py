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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        return outputs.last_hidden_state

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.W = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1)

    def forward(self, h, enc_out):
        src_len = enc_out.shape[1]
        h = h.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W(torch.cat((h, enc_out), dim=2)))
        attn = F.softmax(self.v(energy).squeeze(2), dim=1)
        return attn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        batch, seq_len = tgt.shape
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch, -1)
        x = self.embed(tgt) + self.pos_embed(positions)
        x = self.transformer(tgt=x, memory=memory,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc(x)

class Im2LatexModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=3, dropout=0.2, max_len=100, enc_hid=1024):
        super().__init__()
        self.encoder = Encoder()
        self.enc_to_dec = nn.Linear(enc_hid, d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_decoder_layers,
                                          dim_feedforward=d_model*4, dropout=dropout, max_len=max_len)
        self.max_len = max_len

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, images, tgt):
        memory = self.encoder(images)
        memory = self.enc_to_dec(memory)
        tgt_input = tgt[:, :-1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1)).to(images.device)
        return self.decoder(tgt_input, memory, tgt_mask=tgt_mask)

    def predict(self, image_tensor, tokenizer, max_len=100, beam_width=3):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(image_tensor)
            memory = self.enc_to_dec(memory)
            batch = memory.size(0)
            start_token = tokenizer.token2idx[tokenizer.sos_token]
            sequences = [(torch.full((1, 1), start_token, device=image_tensor.device), 0.0)]
            for _ in range(max_len):
                all_candidates = []
                for seq, score in sequences:
                    if seq[0, -1].item() == tokenizer.token2idx[tokenizer.eos_token]:
                        all_candidates.append((seq, score))
                        continue
                    tgt_mask = self._generate_square_subsequent_mask(seq.size(1)).to(image_tensor.device)
                    logits = self.decoder(seq, memory, tgt_mask=tgt_mask)
                    probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()
                    topk = torch.topk(probs, beam_width)
                    for i in range(beam_width):
                        cand = torch.cat([seq, topk.indices[i].view(1, 1)], dim=1)
                        all_candidates.append((cand, score + topk.values[i].item()))
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(seq[0, -1].item() == tokenizer.token2idx[tokenizer.eos_token] for seq, _ in sequences):
                    break
            best_seq = sequences[0][0].squeeze(0)
            return [tokenizer.decode(best_seq)]


model = Im2LatexModel.from_pretrained("fklska/ViT_TDecoderOCR").to("cpu")
vocab_file_path = hf_hub_download(repo_id="fklska/LaTeX_OCR", filename="token2idx.json")

with open(vocab_file_path, "r", encoding="utf-8") as f:
    saved_token2idx = json.load(f)

tokenizer = LatexTokenizer()
tokenizer.token2idx = saved_token2idx
tokenizer.idx2token = {idx: token for token, idx in saved_token2idx.items()}
tokenizer.vocab = list(saved_token2idx.keys())