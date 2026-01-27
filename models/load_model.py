from transformers import Wav2Vec2ForCTC

def load_model(processor):
    model = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean"
    )

    model.freeze_feature_encoder()
    return model
