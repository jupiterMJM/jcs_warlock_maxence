import torch

from datasets import load_dataset
from transformers import pipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pipeline
pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-small-cv11-french", device=device)

# NB: set forced_decoder_ids for generation utils
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")

# Load data
ds_mcv_test = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="test", streaming=True)
test_segment = next(iter(ds_mcv_test))
waveform = test_segment["audio"]

# Run
generated_sentences = pipe(waveform, max_new_tokens=225)["text"]  # greedy
# generated_sentences = pipe(waveform, max_new_tokens=225, generate_kwargs={"num_beams": 5})["text"]  # beam search

# Normalise predicted sentences if necessary
