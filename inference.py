from models.ViT_TDecoder import model, tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def inference_cnn_lstm(image, model = model, tokenizer=tokenizer, max_len=100, device="cpu"):

    model.eval()
    model.to(device)

    transforms = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.feature_extractor(image_tensor)
        img_embeddings = features.mean(dim=[2, 3])
        h = model.init_h(img_embeddings).unsqueeze(0)
        c = model.init_c(img_embeddings).unsqueeze(0)

        current_token = torch.tensor([[tokenizer.token2idx[tokenizer.sos_token]]], device=device)
        generated_tokens = []

        for _ in range(max_len):
            tgt_embedding = model.embedding(current_token)
            output, (h, c) = model.lstm(tgt_embedding, (h, c))
            logits = model.fc(output)

            next_token = logits.argmax(dim=-1)
            token_id = next_token.item()

            if token_id == tokenizer.token2idx[tokenizer.eos_token]:
                break

            generated_tokens.append(token_id)
            current_token = next_token

    return tokenizer.decode(generated_tokens)


def inference_attention_GRU(
    image, model=model, tokenizer=tokenizer, max_len=100, device="cpu"
):
    model.eval()
    model.to(device)

    image = image.convert("RGB")

    transforms = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.feature_extractor(image_tensor)
        keys = features.flatten(2).permute(0, 2, 1)

        hidden = torch.zeros(1, 1, 512, device=device)

        current_token = torch.tensor(
            [[tokenizer.token2idx[tokenizer.sos_token]]], device=device
        )
        generated_tokens = []

        for _ in range(max_len):
            current_emb = model.embedding(current_token)
            query = hidden.permute(1, 0, 2)

            context, _ = model.attention(query, keys)
            gru_input = torch.cat([current_emb, context], dim=-1)

            output, hidden = model.gru(gru_input, hidden)
            logits = model.out(output)

            next_token = logits.argmax(dim=-1)
            token_id = next_token.item()

            if token_id == tokenizer.token2idx[tokenizer.eos_token]:
                break

            generated_tokens.append(token_id)
            current_token = next_token

    return tokenizer.decode(generated_tokens)


