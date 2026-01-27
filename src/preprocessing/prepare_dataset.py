from transformers import Wav2Vec2Processor
from datasets import Audio

def prepare_dataset(batch, processor: Wav2Vec2Processor):
    audio = batch["audio"]

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids

    return batch
