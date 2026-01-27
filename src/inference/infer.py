import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_PATH = "models/wav2vec2_darija"

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).cuda()

def transcribe(audio_path):
    speech, sr = torchaudio.load(audio_path)

    inputs = processor(
        speech[0],
        sampling_rate=sr,
        return_tensors="pt"
    ).input_values.cuda()

    with torch.no_grad():
        logits = model(inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.decode(pred_ids[0])

if __name__ == "__main__":
    print(transcribe("data/samples/0.wav"))
