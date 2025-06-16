#Correr servidor: uvicorn main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
import random
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
from typing import List
from torch_geometric.utils import from_networkx
import torch.nn as nn
import torch.nn.functional as F

class SEALGNN(torch.nn.Module):
    def __init__(self, num_labels, hidden_dim=64, dropout_rate=0.5):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_labels, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.label_emb(data.x)
        x = self.conv1(x, data.edge_index, data.edge_weight).relu()
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, data.edge_weight).relu()
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        return self.mlp(x).squeeze()

app = FastAPI()

selected_topics = []
#load the topics folder
with open("topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute topic embeddings (mean pooling)
topic_embeddings = {
    topic_id: model.encode(words, convert_to_tensor=True).mean(dim=0)
    for topic_id, words in topics.items()
}
#load model 

checkpoint = torch.load("seal_model.pt")

# Rebuild the architecture exactly the same
trained_model = SEALGNN(
    num_labels=checkpoint['num_labels'],
    hidden_dim=checkpoint['hidden_dim']
)

trained_model.load_state_dict(checkpoint['state_dict'])
trained_model.eval()

#compute graph
G = nx.Graph()

with open("graph_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for topic1, topic2, weight in reader:
        G.add_edge(int(topic1), int(topic2), weight=float(weight))

#stuff for the seal model to work

def drnl_node_labeling(subgraph, u, v):
    dist_u = nx.single_source_shortest_path_length(subgraph, u)
    dist_v = nx.single_source_shortest_path_length(subgraph, v)
    
    labels = {}
    for node in subgraph.nodes():
        d1 = dist_u.get(node, 1e6)
        d2 = dist_v.get(node, 1e6)
        if d1 + d2 == 0:
            label = 0
        else:
            label = 1 + min(d1, d2) + ((d1 + d2) * (d1 + d2 + 1)) // 2
        labels[node] = label

    nx.set_node_attributes(subgraph, labels, "x")  # PYG will treat 'x' as node feature
    return subgraph

def extract_enclosing_subgraph(graph, u, v, h=3, top_k=5):
    def bfs_topk(node, h, k):
        visited = set()
        queue = [(node, 0)]
        collected = set([node])

        while queue:
            current, depth = queue.pop(0)
            if depth >= h:
                continue

            # Get neighbors sorted by descending weight
            neighbors = sorted(
                graph[current].items(),
                key=lambda x: x[1].get('weight', 1.0),
                reverse=True
            )
            # Take top-k neighbors only
            for neighbor, _ in neighbors[:k]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    collected.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return collected

    # Get truncated neighborhoods from u and v
    nodes_u = bfs_topk(u, h, top_k)
    nodes_v = bfs_topk(v, h, top_k)

    # Combine nodes and extract subgraph
    nodes = nodes_u.union(nodes_v)
    subgraph = graph.subgraph(nodes).copy()

    # Apply DRNL labeling
    return drnl_node_labeling(subgraph, u, v)

def convert_to_pyg(subgraph):
    for u, v in subgraph.edges():
        if 'weight' not in subgraph[u][v]:
            subgraph[u][v]['weight'] = 1.0
    
    # Remove edge_attrs from function call
    data = from_networkx(subgraph, group_node_attrs=['x'])

    # Add edge weights manually
    edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    data.x = data.x.long().squeeze()  # DRNL labels
    return data

def predict_link_score(model, graph, u, v, h=3, device='cpu'):
    subgraph_nx = extract_enclosing_subgraph(graph, u, v, h)
    data = convert_to_pyg(subgraph_nx)
    data.edge_weight = data.weight  # pass edge weights to GCNConv
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)  # single graph
    
    model.eval()
    with torch.no_grad():
        score = model(data).item()
    return score

# Function to assign a word to the best matching topic
def assign_word_to_topic(word, topic_embeddings):
    word_embedding = model.encode(word, convert_to_tensor=True)
    
    best_topic = None
    highest_score = -1

    for topic_id, topic_emb in topic_embeddings.items():
        score = util.cos_sim(word_embedding, topic_emb).item()
        if score > highest_score:
            best_topic = topic_id
            highest_score = score

    return best_topic, highest_score

def get_relation(u,v):
    score = predict_link_score(model=trained_model,graph=G,u=u,v=v,h=3)
    prob = torch.sigmoid(torch.tensor(score)).item()
    print(f"Predicted score between {u} and {v}: {prob:.4f}")
    return prob

class Topic(BaseModel):
    id: int
    word: str

@app.get("/topics")
def get_topics():
    return topics
@app.get("/recommended_topics")
def get_recommended_topics():
    random_keys = random.sample(list(topics.keys()), 3)
    # Select one random word from each list
    random_words = [random.choice(topics[key][:2]) for key in random_keys]

    return random_words

@app.get("/topics/search")
def select_topic(word: str):
    topic_id, similarity = assign_word_to_topic(word, topic_embeddings)
    if similarity > 0.5 and topic_id != -1:
        new_topic = Topic(id=topic_id, word=word)
        
        return {
            "message": "Topic found!",
            "selected_topics": new_topic.dict()
        }
    else:
        return {
            "message": "Topic not found",
            "detail": "Couldn't assign word to any specific topic"
        }

class TopicsInput(BaseModel):
    selected_topics: List[Topic]

@app.post("/generate_relation")
def generate_relation(data: TopicsInput):
    selected_topics = data.selected_topics

    if not selected_topics:
        return {"message": "No selected topics"}

    main_topic = selected_topics[0]
    graph = [[main_topic.word, 1.0]]

    for topic in selected_topics[1:]:
        relation = get_relation(main_topic.id, topic.id)
        graph.append([topic.word, relation])

    return {"graph": graph}
