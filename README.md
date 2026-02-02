Moroccan Darija Speech Recognition

This project builds an automatic speech recognition system for Moroccan Darija.

The goal is to convert spoken Darija audio into written text using deep learning.

Data

The system is trained on a dataset of Moroccan Darija speech recordings.

Each audio sample is paired with its corresponding written transcription.

The dataset contains speech in Arabic script.

Text Processing

The transcription text is cleaned before training.

Unnecessary symbols and digits are removed.

Arabic characters are normalized to ensure consistent input.

Missing or empty transcriptions are handled safely.

Vocabulary

A character-level vocabulary is created from the cleaned text.

This allows the model to learn how speech sounds map to written characters.

Special symbols are added for:

Unknown characters

Word separation

Padding

Model

The system is based on a modern speech recognition architecture.

It learns directly from raw audio signals.

It is trained to align speech with text automatically.

Training

The model is trained using supervised learning.

Audio samples and their transcriptions are used as training pairs.

The training process includes regular evaluation of predictions.

Model checkpoints are saved during training.

Evaluation

Performance is measured using Word Error Rate (WER).

This metric shows how many word-level errors appear in predicted text.

Lower WER indicates better transcription quality.

Inference

After training, the system can transcribe new audio recordings.

An audio file is given as input to the model.

The output is a written transcription in Moroccan Darija.
