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
import boto3

# spark session
spark = SparkSession.builder.appName("hypergraph-analysis").getOrCreate()

# reading the csvs
authors_df = spark.read.csv("s3://csp554-storage/neptune-csv/author_nodes/", header=True)
papers_df = spark.read.csv("s3://csp554-storage/neptune-csv/paper_nodes/", header=True)
edges_df = spark.read.csv("s3://csp554-storage/neptune-csv/edges/", header=True)

# creating networkx graph
authors = authors_df.collect()
papers = papers_df.collect()
edges = edges_df.collect()

G = nx.Graph()
for r in authors:
    G.add_node(r["~id"], bipartite=0, name=r["name:String"])
for r in papers:
    G.add_node(r["~id"], bipartite=1, title=r["title:String"])
for r in edges:
    G.add_edge(r["~from"], r["~to"])

print(f"Nodes : {G.number_of_nodes():,}") # 209,776
print(f"Edges : {G.number_of_edges():,}") # 216,463

# saving graph as pickle so it can be reused
import pickle
# Save locally first
with open("/tmp/hypergraph.pkl", "wb") as f:
    pickle.dump(G, f)
# Upload to S3
boto3.client("s3").upload_file("/tmp/hypergraph.pkl", "csp554-storage", "graph/hypergraph.pkl")
print("Graph saved to S3")