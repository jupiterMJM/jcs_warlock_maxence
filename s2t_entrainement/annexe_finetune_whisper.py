"""
fichier annexe du programme d'entrainement principal finetune_whisper.py
:auteur: Maxence BARRE
:date: 18/10/2024
:projet: JCS_warlock
:commentaire: fichier annexe, A NE PAS LANCER
:WARNING: ne pas oublier de se connecter au compte hugging-face pour download le dataset
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import torch

def prepare_dataset(batch, processor):
    """
    Now we can write a function that takes a single training sample and passes it through the processor to prepare it for our model.
    source: https://huggingface.co/blog/audio-datasets
    """
    audio = batch["audio"]

    batch = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["sentence"],
    )

    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    """
    the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model
    les input_features (aka les audios en entrée sont déjà paddé/tronqué à 30s et sont déjà sous forme de log-mel) => il faut les transformer en pytorch tenseur
    par contre, les labels ne sont pas paddé => il faut les padder (on prend la longueur max et on pad avec des -100 pour que les 'ajouts' ne soient pas pris en compte 
    dans le calcul de l'erreur)
    """

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")   # ici ce n'est pas le pad qui ns interesse, c'est le return_tensor => on transforme en tensor

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

def compute_metrics(pred):
    """
    calcule l'erreur entre ce qui a été prédit et ce qu'il fallait trouver
    se base sur l'erreur WER qui est communément utilisé dans les pb de RN audio
    renvoie 2 mesure, un WER "normal" et un WER du texte normalisé
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    # se ref à https://huggingface.co/learn/audio-course/chapter5/evaluation
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}