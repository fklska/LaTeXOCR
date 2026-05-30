from model import model, tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def inference(image, model = model, tokenizer=tokenizer, max_len=100, device="cpu"):
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