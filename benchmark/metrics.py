from evaluate import load
import pandas as pd
import tqdm

data = pd.read_csv("benchmark/results/SWIN_TDecoder.csv")

wer = load("wer")
cer = load("cer")
bleu = load("bleu")

predictions = data["Pred"].to_list()
references = data["True"].to_list()

wer_score = wer.compute(predictions=predictions, references=references)
cer_score = cer.compute(predictions=predictions, references=references)
bleu_score = bleu.compute(predictions=predictions, references=references)

print(wer_score, cer_score, bleu_score["bleu"])