# paper-2025-anonymous-submission

This repository contains a dataset and code to reproduce the results of experiments. You can use the table of contents for easy navigation.

## Content

- [All Datasets](/Data/processed_data/)
  - [KoWIT-24 dataset](/Data/processed_data/dataset.json)
  - [KoWIT-24 Dev dataset](Data/processed_data/dev_dataset.json)
  - [RIA Novosti dataset](Data/processed_data/ria_dataset.json)
- [Prompts]()
  - KoWIT-24
    - [Wordplay detection](Data/processed_data/dataset_wordplay_detection_propmts.json)
    - [Wordplay detection (extended by instruction)](Data/processed_data/dataset_wordplay_detection_propmts_extended.json)
    - [Wordplay interpretation](Data/processed_data/dataset_wordplay_interpretation_propmts.json)
    - [Wordplay interpretation (extended by instruction)](Data/processed_data/dataset_wordplay_interpretation_propmts_extended.json)
  - RIA Novosti
    - [Wordplay detection](Data/processed_data/ria_detection.json)
    - [Wordplay detection (extended by instruction)](Data/processed_data/ria_detection_extended.json)
    - [Wordplay interpretation](Data/processed_data/ria_interpretation.json)
    - [Wordplay interpretation (extended by instruction)](Data/processed_data/ria_interpretation_extended.json)

- [LLM Usage](Notebooks)
  - [Wordplay detection](Notebooks/wordplay_detection)
    - [GigaChat Lite](Notebooks/wordplay_detection/GigaChat-Lite)
    - [GigaChat Max](Notebooks/wordplay_detection/GigaChat-Max)
    - [YandexGPT4](Notebooks/wordplay_detection/YandexGPT-4)
    - [Mistral](Notebooks/wordplay_detection/Mistral-Nemo)
    - [GPT-4o](Notebooks/wordplay_detection/GPT-4o)
  - [Wordplay detection (extended by instruction)](Notebooks/wordplay_detection_extended_prompt)
    - [GigaChat Lite](Notebooks/wordplay_detection_extended_prompt/GigaChat-Lite)
    - [GigaChat Max](Notebooks/wordplay_detection_extended_prompt/GigaChat-Max)
    - [YandexGPT4](Notebooks/wordplay_detection_extended_prompt/YandexGPT-4)
    - [Mistral](Notebooks/wordplay_detection_extended_prompt/Mistral-Nemo)
    - [GPT-4o](Notebooks/wordplay_detection_extended_prompt/GPT-4o)
  - [Wordplay interpretation](Notebooks/wordplay_interpretation)
    - [GigaChat Lite](Notebooks/wordplay_interpretation/GigaChat-Lite)
    - [GigaChat Max](Notebooks/wordplay_interpretation/GigaChat-Max)
    - [YandexGPT4](Notebooks/wordplay_interpretation/YandexGPT-4)
    - [Mistral](Notebooks/wordplay_interpretation/Mistral-Nemo)
    - [GPT-4o](Notebooks/wordplay_interpretation/GPT-4o)
  - [Wordplay interpretation (extended by instruction)](Notebooks/wordplay_interpretation_extended_prompt)
    - [GigaChat Lite](Notebooks/wordplay_interpretation_extended_prompt/GigaChat-Lite)
    - [GigaChat Max](Notebooks/wordplay_interpretation_extended_prompt/GigaChat-Max)
    - [YandexGPT4](Notebooks/wordplay_interpretation_extended_prompt/YandexGPT-4)
    - [Mistral](Notebooks/wordplay_interpretation_extended_prompt/Mistral-Nemo)
    - [GPT-4o](Notebooks/wordplay_interpretation_extended_prompt/GPT-4o)

- [Automatic interpretation](Notebooks/automatic_evaluation_interpretations/evaluation_extended_prompt.ipynb)
- [Raw LLM Predictions](Data/predictions)
