<b>Topic Relatedness Analysis with Graph Neural Networks</b>

<b>Overview</b>

This project leverages a Graph Neural Network (GNN) to analyze how related social media topics are based on co-occurrence in tweets.
It builds a graph of topics as nodes, with edges weighted by how often topics appear together, and outputs a numerical relatedness score between topics.

<b>Tech Stack</b>

Python – Data handling and preprocessing

PyTorch Geometric – Graph Neural Network implementation

Pandas / NumPy – Large-scale data processing

NetworkX – Graph building and analysis

Jupyter Notebook – Experimentation and visualization

<b>Key Features</b>

Twitter Data Ingestion: Fetches tweets for multiple topics and extracts co-occurrence information

Graph Construction: Creates a weighted graph where nodes are topics and edges represent shared mentions

GNN Model: Trains a Graph Neural Network to compute topic relatedness scores

Evaluation: Outputs similarity metrics and supports visual inspection of related topics
