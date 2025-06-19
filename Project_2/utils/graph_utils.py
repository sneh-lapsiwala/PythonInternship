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
