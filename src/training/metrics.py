import numpy as np
import evaluate
from transformers import Wav2Vec2Processor

wer_metric = evaluate.load("wer")

def compute_metrics(pred, processor: Wav2Vec2Processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = processor.batch_decode(pred_ids)

    label_ids_cleaned = []
    for label_seq in pred.label_ids:
        label_ids_cleaned.append([
            token_id for token_id in label_seq
            if token_id != -100 and token_id != processor.tokenizer.pad_token_id
        ])

    label_str = processor.batch_decode(label_ids_cleaned, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
