import os
import re
import json
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import load_dataset
from src.preprocessing.text_cleaning import *
from src.preprocessing.prepare_dataset import prepare_dataset
from src.training.data_collator import DataCollatorCTCWithPadding
from src.training.metrics import compute_metrics
from src.model.load_model import load_model
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor
import evaluate
from jiwer import wer

dataset = load_dataset("atlasia/DODa-audio-dataset", split="train[:30%]")

dataset = dataset.map(remove_special_characters)
dataset = dataset.map(remove_digits)
dataset = dataset.map(normalize)
dataset = dataset.map(prepare_dataset)

processor = Wav2Vec2Processor.from_pretrained("./vocab")

model = load_model(processor)

training_args = TrainingArguments(
    output_dir="./models/wav2vec2_darija",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    eval_strategy="steps",
    logging_steps=50,
    save_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorCTCWithPadding(processor),
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./models/wav2vec2_darija")
processor.save_pretrained("./models/wav2vec2_darija")
