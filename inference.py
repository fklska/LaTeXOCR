from model import model, tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

device = "cpu"

def predict_from_image(pil_image, max_len=50):
    transforms = T.Compose([
        T.Resize((128, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transforms(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs = model.backbone(image_tensor).flatten(2).permute(0, 2, 1)
        h, c = model._init_hidden_state(encoder_outputs)
        
    generated_tokens = [tokenizer.token2idx["<sos>"]]
    eos_idx = tokenizer.token2idx["<eos>"]
    
    for _ in range(max_len):
        last_token = generated_tokens[-1]
        token_tensor = torch.tensor([[last_token]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            current_emb = model.embedding(token_tensor).squeeze(1)
            context, _ = model.attention(encoder_outputs, h)
            lstm_input = torch.cat([current_emb, context], dim=1)
            h, c = model.lstm_cell(lstm_input, (h, c))
            logits = model.fc_out(h)
            next_token = logits.argmax(dim=-1).item()
            
        generated_tokens.append(next_token)
        if next_token == eos_idx:
            break
            
    return tokenizer.decode(generated_tokens)