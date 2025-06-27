# Plotly graph functions
import pandas as pd
import plotly.graph_objects as go

def generate_sankey(df):
    nodes = []
    for path in df['career_path']:
        nodes.extend(path.split(" -> "))
    nodes = list(set(nodes))
    node_map = {k: i for i, k in enumerate(nodes)}

    sources, targets = [], []
    for path in df['career_path']:
        steps = path.split(" -> ")
        for i in range(len(steps)-1):
            sources.append(node_map[steps[i]])
            targets.append(node_map[steps[i+1]])

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=list(node_map.keys()), pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=[1]*len(sources))
    )])
    return fig

# utils/graph_utils.py

import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(graph_data):
    G = nx.Graph()
    
    # Assuming graph_data is a dict with 'nodes' and 'edges'
    G.add_nodes_from(graph_data['nodes'])
    G.add_edges_from(graph_data['edges'])

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15)
    plt.title("Knowledge Graph")
    plt.show()

