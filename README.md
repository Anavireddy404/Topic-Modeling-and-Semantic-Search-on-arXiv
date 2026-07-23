# Arxiv BERT Topic Modelling

### Project Overview

This project applies BERTopic-based topic modelling to paper abstracts from the Cornell University arXiv dataset. It builds a full pipeline — from raw metadata download through embedding, clustering, and topic labelling — to automatically discover and visualize the major research topics present across arXiv's paper abstracts.

### Dataset

The project uses the [Cornell University arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle, downloaded directly via the Kaggle API. The full snapshot contains metadata (id, abstract, categories, update date) for roughly **2.8 million papers** spanning **176 unique arXiv subject categories**.

The dataset is downloaded, unzipped, and streamed line-by-line into memory-safe Parquet chunks (`chunk_*.parquet`) before being merged into a single DataFrame for processing.

### Pipeline

1. **Data ingestion** — Download the arXiv metadata snapshot from Kaggle and parse it into a DataFrame (`id`, `abstract`, `categories`, `year`), chunked to Parquet to avoid memory issues.
2. **Abstract cleaning** — Lowercase text, expand/replace LaTeX symbols (`\alpha` → `alpha`, etc.), strip LaTeX equations, URLs, arXiv IDs, and emails, and normalize whitespace.
3. **Embedding generation** — Encode cleaned abstracts with the Hugging Face `SentenceTransformer` model `BAAI/bge-small-en`.
4. **Dimensionality reduction** — Reduce embeddings with `UMAP` (2D for visualization, 10D for clustering input).
5. **Clustering** — Group similar abstracts using `HDBSCAN` (`min_cluster_size=100`, `min_samples=10`).
6. **Topic representation** — Label each topic cluster using a combination of:
   - `KeyBERTInspired`
   - `MaximalMarginalRelevance`
   - `GPT-4o-mini` (via the OpenAI API) for human-readable topic labels
7. **Model assembly** — Combine all sub-models into a single `BERTopic` pipeline and fit it on the full document set.
8. **Saving** — Persist both the embedding model and the trained BERTopic model (`safetensors` format) for reuse.
9. **Visualization** — Generate interactive topic maps (Plotly scatter plots, bar charts) and a labelled document map via `datamapplot`.
10. **Evaluation** — Report topic count, average documents per topic, and a topic diversity score (unique top words across topics).

### Repository Structure

```
Arxiv-BERT-Topic-Modelling/
└── Arxiv_BERT_Topic_Modelling_Full_code_100,000samples.ipynb
```

### Requirements

The notebook relies on the following Python packages:

```
pandas
numpy
matplotlib
plotly
tqdm
scikit-learn
sentence-transformers
umap-learn
hdbscan
bertopic
datamapplot
openai
```

Install them with:

```bash
pip install pandas numpy matplotlib plotly tqdm scikit-learn sentence-transformers umap-learn hdbscan bertopic datamapplot openai
```

The notebook was originally written for Google Colab (it uses `google.colab.files` and the Kaggle CLI download pattern); running locally just means skipping the Colab-specific upload cell.

### Configuration

This pipeline calls the OpenAI API (`gpt-4o-mini`) to generate human-readable topic labels, and the Kaggle API to download the dataset. Before running, set both as environment variables rather than hardcoding them in the notebook:

```bash
export OPENAI_API_KEY="your-key-here"
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-kaggle-key"
```

Then in the notebook, load the key with `os.environ["OPENAI_API_KEY"]` instead of pasting it directly into the code.

> **Note:** the current notebook has a live OpenAI API key hardcoded in the "GPT 4o mini Initialization" cell. Since this repo is public, that key is exposed — revoke/rotate it at [platform.openai.com](https://platform.openai.com/api-keys) and remove it from the notebook before your next commit.

### How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/arxiv-bert-topic-modelling.git
   cd arxiv-bert-topic-modelling
   ```

2. Install the dependencies listed above.

3. Set your Kaggle and OpenAI credentials as environment variables (see **Configuration**).

4. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook "Arxiv_BERT_Topic_Modelling_Full_code_100,000samples.ipynb"
   ```

5. Run all cells from top to bottom. The dataset download and full embedding/clustering pipeline can take a while given the dataset size (~2.8M abstracts) — consider subsampling `documents` for a faster local test run.

### Results

- **Papers processed:** ~2.8 million abstracts
- **Unique arXiv categories:** 176
- **Embedding model:** `BAAI/bge-small-en`


