import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

device = "cpu"

repo_name = "fklska/trocr_latex"

model = VisionEncoderDecoderModel.from_pretrained(repo_name).to(device)
processor = TrOCRProcessor.from_pretrained(repo_name)

def predict_latex(image: Image.Image, max_length: int = 64, num_beams: int = 4) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()