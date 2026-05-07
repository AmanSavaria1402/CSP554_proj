# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
# need to install networkx
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "networkx"])
sys.path.insert(0, "/home/hadoop/.local/lib/python3.11/site-packages")
subprocess.run([sys.executable, "-m", "pip", "install", "scipy"])
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

# making a projected graph
proj = nx.projected_graph(G, author_ids)
print(f"Hypergraph authorship edges : {G.number_of_edges():,}") # 216,463
print(f"Projection co-author edges : {proj.number_of_edges():,}") # 470,622
print(f"Edge inflation factor : {proj.number_of_edges()/G.number_of_edges():.2f}x") # 2.17x

# comparing the degree distribution
hyper_author_degs = [G.degree(n) for n in author_ids]
proj_degs = [proj.degree(n) for n in proj.nodes()]

print(f"Hypergraph avg author degree : {np.mean(hyper_author_degs):.2f}") # 1.35
print(f"Projection avg degree : {np.mean(proj_degs):.2f}") # 5.89
print(f"Hypergraph max author degree : {max(hyper_author_degs)}") # 31
print(f"Projection max degree : {max(proj_degs)}") # 264


# plotting degree distributions side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

c1 = Counter(hyper_author_degs)
ax1.bar(c1.keys(), c1.values(), color="#4C72B0", alpha=0.85)
ax1.axvline(np.mean(hyper_author_degs), color="red", linestyle="--",
            label=f"mean={np.mean(hyper_author_degs):.1f}")
ax1.set_yscale("log")
ax1.set_xlabel("Papers per author")
ax1.set_ylabel("Count")
ax1.set_title("Hypergraph: Author Degree Distribution")
ax1.legend()

c2 = Counter(proj_degs)
ax2.bar(c2.keys(), c2.values(), color="#DD8452", alpha=0.85)
ax2.axvline(np.mean(proj_degs), color="red", linestyle="--",
            label=f"mean={np.mean(proj_degs):.1f}")
ax2.set_yscale("log")
ax2.set_xlabel("Co-authors per author")
ax2.set_ylabel("Count")
ax2.set_title("Projection: Author Degree Distribution")
ax2.legend()

plt.tight_layout()
plt.savefig("/tmp/degree_comparison.png", dpi=150)
boto3.client("s3").upload_file("/tmp/degree_comparison.png", "csp554-storage", "plots/degree_comparison.png")
print("Degree comparison plot saved to S3")

# clustering coefficient of projection
print(f"Avg clustering coefficient : {nx.average_clustering(proj):.4f}") # 0.8255

# plotting the projection
# since the projection will have a lot of nodes and edges, best to take a subset of top 500 nodes or something like that.
fig, ax = plt.subplots(figsize=(12, 12))
top_authors_proj = sorted(proj.nodes(), key=lambda n: proj.degree(n), reverse=True)[:500]
sub_proj = proj.subgraph(top_authors_proj).copy()
pos = nx.kamada_kawai_layout(sub_proj)

nx.draw_networkx(
    sub_proj,
    pos=pos,
    ax=ax,
    node_size=30,
    node_color="steelblue",
    edge_color="gray",
    alpha=0.6,
    with_labels=False,
    width=0.4
)

ax.set_title("Pairwise Co-authorship Projection")
ax.axis("off")
plt.tight_layout()
plt.savefig("/tmp/projected_graph.png", dpi=150)
plt.close(fig)

boto3.client("s3").upload_file("/tmp/projected_graph.png", "csp554-storage", "plots/projected_graph.png")
print("Projected graph plot saved to S3")