import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) für den Graph Transformer.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Transformationsmatrix für Eingabefeatures
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        # Aufmerksamkeitsparameter
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        # Aktivierungsfunktion und Dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, h, adj):
        # h: Knotenfeatures [N, in_features]
        # adj: Adjazenzmatrix [N, N]
        
        # Feature-Transformation
        Wh = torch.mm(h, self.W)  # [N, out_features]
        
        # Berechne Aufmerksamkeitskoeffizienten
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]
        
        # Maskiere nicht verbundene Knoten
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        
        # Normalisiere Aufmerksamkeitskoeffizienten mit Softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Anwenden der Aufmerksamkeit auf Features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        
        # Wiederhole die Features für jeden Knoten
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Kombiniere Features für alle Knotenpaare
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Reshape zu [N, N, 2*out_features]
        return all_combinations.view(N, N, 2 * self.out_features)

class GraphTransformer(nn.Module):
    """
    Graph Transformer für die Verarbeitung des disjunktiven Graphen.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Eingabe-Embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-Head Attention Layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            heads = nn.ModuleList()
            for _ in range(num_heads):
                heads.append(GraphAttentionLayer(hidden_dim, hidden_dim // num_heads, dropout=dropout))
            self.attention_layers.append(heads)
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        # x: Knotenfeatures [N, input_dim]
        # adj: Adjazenzmatrix [N, N]
        
        # Eingabe-Embedding
        h = self.embedding(x)
        
        # Multi-Head Attention Layers
        for i, heads in enumerate(self.attention_layers):
            h_cat = []
            for head in heads:
                h_head = head(h, adj)
                h_cat.append(h_head)
            
            # Konkateniere die Heads
            h_multi = torch.cat(h_cat, dim=1)
            
            # Residual Connection und Layer Normalization
            h = h + self.dropout(h_multi)
            h = self.layer_norms[i](h)
        
        # Output Layer
        output = self.output_layer(h)
        
        return output

def extract_graph_features(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrahiert Features und Adjazenzmatrix aus dem disjunktiven Graphen.
    
    Args:
        G: NetworkX DiGraph-Objekt
        
    Returns:
        Tuple aus (Knotenfeatures, Adjazenzmatrix)
    """
    # Erstelle eine Mapping von Knotennamen zu Indizes
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Extrahiere Knotenfeatures
    features = []
    for node in nodes:
        attrs = G.nodes[node]
        
        # Standardfeatures für alle Knoten
        node_type = 1.0 if attrs.get('type') == 'operation' else 0.0
        processing_time = float(attrs.get('time', 0))
        priority = float(attrs.get('priority', 1))
        
        # Berechne zusätzliche Features
        in_degree = float(G.in_degree(node))
        out_degree = float(G.out_degree(node))
        
        # Berechne die Anzahl der disjunktiven Kanten
        disj_edges = sum(1 for _, _, attr in G.out_edges(node, data=True) 
                         if attr.get('type') == 'disjunctive')
        
        # Kombiniere alle Features
        node_features = [
            node_type,
            processing_time,
            priority,
            in_degree,
            out_degree,
            float(disj_edges)
        ]
        
        features.append(node_features)
    
    # Erstelle Adjazenzmatrix
    n = len(nodes)
    adj = torch.zeros((n, n))
    
    # Fülle die Adjazenzmatrix
    for u, v, attr in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        
        # Gewichte für verschiedene Kantentypen
        if attr.get('type') == 'conjunctive':
            adj[i, j] = 1.0
        elif attr.get('type') == 'disjunctive':
            adj[i, j] = 0.5
    
    return torch.FloatTensor(features), adj

def create_graph_transformer(G: nx.DiGraph) -> GraphTransformer:
    """
    Erstellt einen Graph Transformer für den gegebenen disjunktiven Graphen.
    
    Args:
        G: NetworkX DiGraph-Objekt
        
    Returns:
        GraphTransformer-Modell
    """
    # Extrahiere Features und Adjazenzmatrix
    features, _ = extract_graph_features(G)
    
    # Bestimme die Eingabedimension basierend auf den extrahierten Features
    input_dim = features.shape[1]
    
    # Erstelle den Graph Transformer
    model = GraphTransformer(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    return model