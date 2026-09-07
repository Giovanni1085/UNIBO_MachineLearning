# Machine Learning for the Arts&Humanities

This repository is part of the course **Machine Learning for the Arts&Humanities** at the University of Bologna, master degree in [Digital Humanities and Digital Knowledge](https://corsi.unibo.it/2cycle/DigitalHumanitiesKnowledge).

*Please note that this repository will be updated continuosly in the future, as new editions of this course are proposed.*

See the [course program](https://www.unibo.it/it/studiare/insegnamenti-competenze-trasversali-moocs/insegnamenti/insegnamento/2026/542135) for the week-by-week schedule.

## Contents

0. [Coding agents](0_coding_agents.ipynb) — week 1
1. [Linear regression](1_linear_regression.ipynb) — week 2
2. [Linear classification](2_linear_classification.ipynb) — week 2
3. [PyTorch](3_pytorch.ipynb) — week 3 (theory/practice + linear regression and MNIST lab)
4. [Transformers](4_transformers.ipynb) — week 4 (attention and text classification, then a CNN recap and CLIP)
5. [Generative Models and RAG](5_generative_models.ipynb) — week 5 (local vs frontier models, RAG)
6. [Retrieval Augmented Generation app](rag_app) — week 6

## Datasets and exercises

We will use several **datasets**, available in the [Data folder](data/). 

These datasets include:
* [Applied Data Analysis](https://github.com/mromanello/ADA-DHOxSS/tree/master/data), various datasets.
* [Computer Vision for the Humanities: An Introduction to Deep Learning for Image Classification](https://programminghistorian.org/en/lessons/computer-vision-deep-learning-pt1) from the Programming Historian series.
* [British Library 19th century books](https://github.com/mromanello/ADA-DHOxSS/tree/master/data#british-library-19th-century-books).
* [TopRes19th dataset](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-topres19th.md#topres19th-dataset) — used as optional homework (Named Entity Recognition), not in the active notebooks.

## Coding agents

Starting from week 2, exercises are meant to be tackled with the help of coding agents such as [Claude Code](https://claude.com/product/claude-code) and [OpenAI Codex](https://openai.com/index/introducing-codex/). See [notebook 0](0_coding_agents.ipynb) for a quickstart, configuration, and usage guide.

## Setting-up your working environment

Please see the `requirements` file for a list of dependencies. [PyTorch can be installed following these instructions](https://pytorch.org/get-started/locally/). For notebook 5 and the RAG app, you will also need:
* [Ollama](https://ollama.com/) installed locally, with `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`.
* API keys for `MISTRAL_API_KEY` (notebook 5) and `OPENAI_API_KEY` (RAG app), set as environment variables or in a local `.env` file (never commit this file — it is already covered by `.gitignore`). Copy `.env.template` to `.env` and fill in your keys.

Lastly, to setup your working environment, refer to [this guide](https://github.com/Giovanni1085/UNIBO_Programmazione_LM/blob/main/setup.md).

## Book

These materials are in part based on the book *[Dive into Deep Learning](https://d2l.ai/)*.

## Acknowledgements

Some materials are re-purposed from:
* [UvA Deep Learning course](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html).
* [PyTorch tutorials](https://pytorch.org/tutorials/).
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
* [CLIP: Connecting text and images](https://openai.com/index/clip/) (OpenAI).
* [Ollama](https://ollama.com/) for local model serving.
