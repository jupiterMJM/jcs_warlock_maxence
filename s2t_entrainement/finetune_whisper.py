"""
Finetune du modèle whisper OpenAI pour de meilleures performances en français
:auteur: Maxence BARRE
:date: 18/10/2024
:projet: JCS_warlock
:commentaire: fichier à utiliser pour entrainement sur machine distante
:WARNING: ne pas oublier de se connecter au compte hugging-face pour download le dataset
"""


# importation de TOUS les modules utilisés
print("[INFO] Importation des modules")
# datasets est un module de hugging face et facilite le téléchargement et le pré-traitement des données issues de banques de données reconnues (dont MCVD)
from datasets import load_dataset, DatasetDict  # pour télécharger et mettre en forme le dataset  
from datasets import Audio                      # pour effectuer des opérations sur les fichiers audio (comme une modif de la fréquence d'enregistrement, sampling)

# transformers est un module de hugging face et fournit des centaines de modèles pré-entrainé. Celui qui nous interesse est le whisperprocessor (issu du papier: https://cdn.openai.com/papers/whisper.pdf)
# de ce que j'ai compris, le whisperProcessor n'est pas le modèle en soit, il s'agit plutot du module qui met en forme les données pour etre admissible par le modèle (mais où il est téléchargé ????)
# This processor pre-processes the audio to input features and tokenises the target text to labels.
# gère le padding/troncature, le log-mel, ainsi que la tokenization
from transformers import WhisperProcessor
# là, on importe vraiment le modèle whisper-small déjà pré-entrainé (on importe plutot la fonction qui importera le modèle pré-entrainé)
from transformers import WhisperForConditionalGeneration
# le module ci-dessous va permettre  de "normaliser" le txt, cad d'enlever les ponctuations et les accents
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import Seq2SeqTrainingArguments       # nrmlt, ca facilite l'entrainement du modèle (on l'utilise pr  faire passer les arg de l'entrainement)
from transformers import Seq2SeqTrainer                 # "fusionne"  les arg d'entrainement avec le modèle

# importation de pytorch, le module d'IA
import torch

# le module utilisé pour avoir la métrique d'évaluation (l'erreur)
import evaluate

# autres modules de python "annexes"
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial

# importation des modules perso (surtout des fonctions)
import annexe_finetune_whisper as annexe
print("[SUCCESS] Importation des modules reussie")

# téléchargement du Dataset
print("[INFO] Téléchargement du dataset")
common_voice = DatasetDict()

# téléchargement des données d'entrainement
# le fr spécifie que l'on prend QUE des données en francais
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_6_0", "fr", split="train+validation", trust_remote_code=True
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_6_0", "fr", split="test", trust_remote_code=True
)
print("[SUCCESS] Téléchargement du dataset réussi!")



# pré-traitement du dataset
common_voice = common_voice.select_columns(["audio", "sentence"])   # la BDD rassemble d'autres caractéristiques comme l'age, le genre, l'accent etc. cela ne nous interesse pas.
# on décide d'utiliser le modèle préentrainé whisper-small 3e sur les 6 proposés (en terme de taille). Le modèle contient 244M de paramètres
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="french", task="transcribe"
)
# le transcribe indique explicitement que l'on veut en fr-oral to fr-texte
# le modèle a probablement été entrainé de facon multi-linguistique => grande polyvalence (aussi disponible en english uniquement)

# The load_dataset function prepares audio samples with the sampling rate that they were published with. 
# This is not always the sampling rate expected by our model. In this case, we need to resample the audio to the correct sampling rate.
sampling_rate = processor.feature_extractor.sampling_rate           # le RN est créé pour une fréq d'enregistrement bien défini
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))    # indique qu'il faut resampler les audios (mais ne les modifient pas, c'est "on the fly")
common_voice.cleanup_cache_files()      # a quoi ca sert? jsp, pr clean des potentiels fichiers en cache???? ca n'a pas l'air très important en tous cas


common_voice = common_voice.map(
    annexe.prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2
)
# https://huggingface.co/docs/datasets/process
# l'arg num_proc permet d'appliquer la fonction prepare_dataset en multiprocessing (ici en utilisant 2 coeur de proco/gpu)
# ATTENTION, visiblement ca peut merder entre linux et windows, cf https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/12?page=2


# mise des données sous forme de batch (pour l'entrainement)
data_collator = annexe.DataCollatorSpeechSeq2SeqWithPadding(processor=processor)   # et on initialise le data collator que l'on vient de définir
# la métrique Word Error Rate WER est utilisée (c'est celle qui est utilisée communément pour les problèmes de reconnaissance vocale)
metric = evaluate.load("wer")

# et là, on télécharge le modèle RN pré-entrainé
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")     # on dl la version small (à changer potentiellement, dans tous les cas, c'est ce qui est communément utilisé)


# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="french", task="transcribe", use_cache=True
)   # ici, on indique au modèle qu'il parle francais, et qu'il doit faire du speech-to-text + qu'il a le droit d'utiliser la mémoire cache
# est equivalent à model.generation_config.task = "transcribe" (pr la partie transcribe)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-fr",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    use_cpu=False
)

# on fusionne les arg d'entrainement avec le modèle (comme ca, on pourra faire un model.train après)
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=annexe.compute_metrics,
    tokenizer=processor,
)


trainer.train()     # et on lance l'entrainement!!!