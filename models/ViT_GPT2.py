from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch


device = "cpu"

model = VisionEncoderDecoderModel.from_pretrained("fklska/vit_gpt2_latex").to(device)
processor = ViTImageProcessor.from_pretrained("fklska/vit_gpt2_latex")
tokenizer = AutoTokenizer.from_pretrained("fklska/vit_gpt2_latex")

# 1. Добавляем новый pad-токен (если его нет или он совпадает с eos)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Расширяем эмбеддинги (новый токен получает случайные веса, но мы скопируем с eos)
    model.decoder.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        # Копируем веса, чтобы не ломать модель
        model.decoder.transformer.wte.weight[pad_id] = model.decoder.transformer.wte.weight[eos_id].clone()
        model.decoder.lm_head.weight[pad_id] = model.decoder.lm_head.weight[eos_id].clone()

# 2. Принудительно обновляем конфиг декодера (GPT2Config) – это ключевой шаг
model.config.decoder.pad_token_id = tokenizer.pad_token_id
model.config.decoder.eos_token_id = tokenizer.eos_token_id
model.config.decoder.bos_token_id = tokenizer.bos_token_id   # GPT-2: 50256
model.config.pad_token_id = tokenizer.pad_token_id           # общий конфиг
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

# 3. Устанавливаем decoder_start_token_id (обычно bos_token_id)
model.config.decoder_start_token_id = tokenizer.bos_token_id

def predict_latex(image: Image.Image, max_length: int = 100, num_beams: int = 4) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,    # можно и не передавать, если конфиг обновлён
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()