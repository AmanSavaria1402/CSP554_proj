# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
# need to install networkx
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "networkx"])
sys.path.insert(0, "/home/hadoop/.local/lib/python3.11/site-packages")
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import boto3
import numpy as np


# spark session
spark = SparkSession.builder.appName("hypergraph-analysis").getOrCreate()

# loading the graph
boto3.client("s3").download_file("csp554-storage", "graph/hypergraph.pkl", "/tmp/hypergraph.pkl")
with open("/tmp/hypergraph.pkl", "rb") as f:
    G = pickle.load(f)
author_ids = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
paper_ids = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}

# getting metrics from the graph
author_degs = [G.degree(n) for n in author_ids]
paper_degs = [G.degree(n) for n in paper_ids]
# we have the average degree from the dataset profiling, but its good to have a mean degree here too, jsut in case
mean_author_degs = np.mean(author_degs)
mean_paper_degs = np.mean(paper_degs)

print(f"Mean author degree: {mean_author_degs}")
print(f"Mean paper degree: {mean_paper_degs}")

# connected components
comps = list(nx.connected_components(G))
lcc = max(comps, key=len)
print(f"Connected components: {len(comps)}") # 17204
print(f"Largest CC (% of nodes): {len(lcc)/G.number_of_nodes():.1%}") # 50.2%

# plotting the graph for top 30 authors (most connected)
top_authors = sorted(author_ids, key=lambda n: G.degree(n), reverse=True)[:30]
vis_papers = {p for a in top_authors for p in list(G.neighbors(a))[:4]}
sub = G.subgraph(set(top_authors) | vis_papers).copy()

sorted_a = sorted([n for n in sub if sub.nodes[n]["bipartite"]==0], key=lambda n: -sub.degree(n))
sorted_p = sorted([n for n in sub if sub.nodes[n]["bipartite"]==1], key=lambda n: -sub.degree(n))

pos = {n: (0.0, i / max(len(sorted_a)-1,1)) for i, n in enumerate(sorted_a)}
pos.update({n: (1.0, i / max(len(sorted_p)-1,1)) for i, n in enumerate(sorted_p)})

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#cccccc", width=0.6, alpha=0.6)
nx.draw_networkx_nodes(sub, pos, nodelist=sorted_a, ax=ax,
    node_color="#4C72B0", node_size=[80+25*sub.degree(n) for n in sorted_a], alpha=0.9)
nx.draw_networkx_nodes(sub, pos, nodelist=sorted_p, ax=ax,
    node_color="#DD8452", node_size=60, node_shape="s", alpha=0.7)

labels = {n: " ".join([p[0]+"." for p in sub.nodes[n]["name"].split()[:-1]]
                       + [sub.nodes[n]["name"].split()[-1]])
          for n in sorted_a}
nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=7,
                        horizontalalignment="right")

ax.text(0.02, 1.01, "AUTHORS", transform=ax.transAxes, color="#4C72B0", fontweight="bold")
ax.text(0.95, 1.01, "PAPERS",  transform=ax.transAxes, color="#DD8452", fontweight="bold")
ax.set_title("Bipartite Incidence Subgraph (top-30 authors)", fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("/tmp/bipartite_subgraph.png", dpi=150)

boto3.client("s3").upload_file("/tmp/bipartite_subgraph.png", "csp554-storage", "plots/bipartite_subgraph.png")
print("Bipartite subgraph saved to S3")