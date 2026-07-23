arXiv Topic Modeling with BERTopic - README
Overview
This project implements topic modeling on arXiv research paper abstracts using BERTopic. The code processes 100,000 arXiv papers, creates embeddings using BERTopic, and applies clustering techniques like UMAP and HDBScan to identify related papers with interactive visualizations.

Requirements

Python 3.7+
Required packages:
bertopic
datamapplot
sentence-transformers
umap-learn
hdbscan
plotly
openai
scikit-learn
pandas
numpy
matplotlib
tqdm


How to Run

Install required packages:
pip install -r requirements

IMPORTANT: Replace the OpenAI API key in the script:

Locate this section in the code:
pythonclient = openai.OpenAI(api_key="<PersonalOpenAIKey>")

Replace <PersonalOpenAIKey> with your own OpenAI API key:
pythonclient = openai.OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxx")



Run the main script:
python Arxiv_BERT_Topic_Modelling_Full_code_100,000samples.ipynb

For Google Colab:

Upload the script to Colab and run all cells in order



Project Files

Arxiv_BERT_Topic_Modelling_Full_code_100,000samples.ipynb: Complete implementation code
Arxiv_Abstracts_Data_Visualizations.ipynb: Visualization outputs and charts
Arxiv_Abstracts_Datacleaning.ipynb: Data cleaning and preprocessing scripts
arxiv_sampled_dataset.json: Sampled dataset for model training
requirements.txt: List of required packages

Data Processing

The script automatically downloads the arXiv dataset from Kaggle
Randomly samples 100,000 papers from the complete dataset
Cleans abstracts by removing LaTeX symbols, equations, URLs, etc.
Creates document embeddings using the SentenceTransformer model

Model Components

Uses UMAP for dimensionality reduction (n_neighbors=30, n_components=10)
Applies HDBSCAN for clustering (min_cluster_size=100)
Implements multiple topic representation models:

KeyBERT and MMR for topic word extraction
GPT-4o mini for generating descriptive topic labels



Output Files

Saved BERTopic model: my_topic_model_2.7
Saved embedding model: my_embedding_model
Topic visualization plot: plot_datamapplot.png

Resource Requirements

Processing 100,000 documents requires significant computational resources
For memory issues and api rate limit, consider reducing the sample size in the code
