from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import boto3

spark = SparkSession.builder.appName("hypergraph-profiling").getOrCreate()

# reading the csvs
authors_df = spark.read.csv("s3://csp554-storage/neptune-csv/author_nodes/", header=True)
papers_df  = spark.read.csv("s3://csp554-storage/neptune-csv/paper_nodes/", header=True)
edges_df   = spark.read.csv("s3://csp554-storage/neptune-csv/edges/", header=True)

# node and edge counts in the graph
num_authors = authors_df.count()
num_papers  = papers_df.count()
num_edges   = edges_df.count()

print(f"Authors : {num_authors:,}") # 159766
print(f"Papers : {num_papers:,}") # 50000
print(f"Authorship edges : {num_edges:,}") # 216463

# authors per paper (this will be the size of the hyperedge)
authors_per_paper = (
    edges_df
    .groupBy("~to")
    .agg(F.count("~from").alias("author_count"))
)

avg_hyperedge_size = authors_per_paper.agg(F.avg("author_count")).collect()[0][0]
print(f"Avg hyperedge size : {avg_hyperedge_size:.2f}") # prints 4.33

# plotting: hyperedge size distribution
# Collect counts for plotting
paper_deg_counts = (
    authors_per_paper
    .groupBy("author_count")
    .agg(F.count("*").alias("freq"))
    .orderBy("author_count")
    .collect()
)

# plot of average hyperedge size
x = [r["author_count"] for r in paper_deg_counts]
y = [r["freq"]         for r in paper_deg_counts]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, y, color="#DD8452", alpha=0.85)
ax.axvline(avg_hyperedge_size, color="red", linestyle="--",
           label=f"mean={avg_hyperedge_size:.1f}")
ax.set_xlabel("Authors per paper (hyperedge size)")
ax.set_ylabel("Count")
ax.set_title("Figure 1: Hyperedge Size Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/fig1_hyperedge_size.png")
boto3.client("s3").upload_file("/tmp/fig1_hyperedge_size.png", "csp554-storage", "plots/fig1_hyperedge_size.png")
print("Figure 1 saved to S3")

# plot of author degree distribution
# papers per author
papers_per_author = (
    edges_df
    .groupBy("~from")
    .agg(F.count("~to").alias("paper_count"))
)

author_deg_counts = (
    papers_per_author
    .groupBy("paper_count")
    .agg(F.count("*").alias("freq"))
    .orderBy("paper_count")
    .collect()
)

avg_author_degree = papers_per_author.agg(F.avg("paper_count")).collect()[0][0]
print(f"Avg author degree : {avg_author_degree:.2f}") # prints 1.35

x = [r["paper_count"] for r in author_deg_counts]
y = [r["freq"]        for r in author_deg_counts]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, y, color="#4C72B0", alpha=0.85)
ax.axvline(avg_author_degree, color="red", linestyle="--",
           label=f"mean={avg_author_degree:.1f}")
ax.set_yscale("log")
ax.set_xlabel("Papers per author")
ax.set_ylabel("Count")
ax.set_title("Figure 2: Author Activity Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/fig2_author_activity.png")

boto3.client("s3").upload_file("/tmp/fig2_author_activity.png", "csp554-storage", "plots/fig2_author_activity.png")
print("Figure 2 saved to S3")
