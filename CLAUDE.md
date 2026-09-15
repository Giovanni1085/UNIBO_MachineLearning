# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Teaching material for the **Machine Learning for the Arts & Humanities** course (University of Bologna, Digital Humanities and Digital Knowledge master). The repository is a sequence of Jupyter notebooks plus one standalone demo app, not a packaged library. Content is edited and re-run across course editions rather than deployed.

## Structure

- `0_coding_agents.ipynb` → `5_generative_models.ipynb`: ordered course notebooks (coding agents → linear regression → linear classification → PyTorch → transformers → generative models/RAG). Each is self-contained, mixing markdown explanations, math, and runnable code cells; later notebooks assume familiarity with concepts from earlier ones but do not import from each other. See `PROGRAM.md` for the week-by-week schedule these notebooks map onto.
- `4_transformers.ipynb` covers both text (attention, encoder/decoder, a BERT-based classification exercise on `data/bl_books`) and images (a brief CNN recap, then CLIP zero-shot classification/search on `data/newspaper_images`) — it replaces the older, separate `4_machine_vision.ipynb` and absorbs the transformer theory that used to live in `5_language_processing.ipynb`. NER on `data/topRes19th_v2` was cut from the active notebooks (available in git history) to fit the new schedule.
- `5_generative_models.ipynb` covers using local (Ollama) vs frontier (Anthropic/OpenAI) models via prompting, and a from-scratch RAG implementation that leads into `rag_app/`.
- `data/`: datasets used by the notebooks, one subfolder per dataset (e.g. `apprenticeship_venice`, `bl_books`, `musk_tweets`, `newspaper_images`, `topRes19th_v2`). Notebooks reference these paths directly (relative to repo root), so keep dataset paths stable when editing notebooks.
- `figures/`: images embedded in notebook markdown cells via relative paths (e.g. `figures/data-science-model.png`).
- `rag_app/`: a separate, runnable RAG (Retrieval-Augmented Generation) chat application, independent of the numbered notebooks — see below.
- `requirements.txt`: flat dependency list for the whole repo (notebooks + rag_app). PyTorch is intentionally excluded from pinned install instructions — installed separately per https://pytorch.org/get-started/locally/.

## Working with the notebooks

- There is no build/lint/test tooling in this repo — validate changes by executing notebook cells (or `jupyter nbconvert --to notebook --execute <file>.ipynb`) rather than looking for a CI config.
- Preserve the existing structure of numbered sections/markdown explanations when editing — these notebooks double as lecture material, so prose and pedagogical ordering matter as much as code correctness.
- Keep any new datasets under `data/<dataset_name>/` and reference them with paths relative to the repo root, consistent with existing notebooks.

## `rag_app/` architecture

A minimal RAG chat app built with **Chainlit** (UI/session layer) and **LlamaIndex** (retrieval/indexing), used as a live teaching example and course exercise starting point.

- `app.py`: the entire application.
  - `auth_callback`: simple username/password gate read from `CHAINLIT_USERNAME` / `CHAINLIT_PASSWORD` env vars.
  - `@cl.on_chat_start`: on each new session, loads all files under `rag_app/data/` via `SimpleDirectoryReader`, builds a `VectorStoreIndex`, and wraps it in a context-mode chat engine with a hardcoded system prompt (`prompt_text`) describing a fixed 6-paper knowledge base.
  - `@cl.on_message`: streams the LLM response token-by-token and appends a "Sources" message listing the PDF files retrieved for that answer (rendered inline via `cl.Pdf`).
  - LLM/embedding backend is swappable: OpenAI (`gpt-4-turbo` + `text-embedding-3-small`) is active by default; Ollama (`llama3.1` + `nomic-embed-text`) lines are present but commented out for local-LLM use.
- `rag_app/data/*.pdf`: the fixed knowledge base (6 papers on ML/AI in arts, humanities, cultural heritage) — the system prompt in `app.py` explicitly enumerates and cites these.
- Requires `OPENAI_API_KEY` (and `CHAINLIT_USERNAME`/`CHAINLIT_PASSWORD`) as environment variables; run with `chainlit run app.py` from `rag_app/`.
- The course exercise (see `rag_app/README.md`) is to extend this app: swap in different documents, change the vector DB, add UI features, or compare LLMs — so when asked to modify `rag_app`, prefer additive/parametrized changes over rewriting the existing flow.
