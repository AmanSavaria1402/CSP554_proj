from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
import os


# INPUT AND OUTPUT PATHS
S3_INPUT = "s3://csp554-storage/DBLP-raw/dblp.xml"
S3_OUTPUT = "s3://csp554-storage/neptune-csv/"
YEAR_START = 2022
YEAR_END = 2025
MAX_PAPERS = 50000

# creating a spark session
spark_session = SparkSession.builder.appName('dblp-hypergraph').getOrCreate()

# spark-xml reads each top-level element as a row
# rowTag should match the record element in DBLP XML (article, inproceedings, etc.)
raw_dblp = spark_session.read.format("xml").option("rowTag", "inproceedings").load(S3_INPUT)

# filtering to get the subset we want
df_filtered = raw_dblp.filter(F.col("year").between(YEAR_START, YEAR_END)).filter(F.col("author").isNotNull())
df = df_filtered.withColumn(
    "authors",
    F.when(F.col("author._VALUE").isNotNull(),
           F.array(F.col("author._VALUE")))
           .otherwise(F.col("author").cast(ArrayType(StringType())))
                            )
df = df.filter(F.size("authors")>=2).dropDuplicates("_key").limit(MAX_PAPERS)

df.cache()
print(f"Filtered paper counts: {df.count()}")


# CREATING IDS AND EDGES
df = (df
        .withColumn("paper_id", F.concat(F.lit("p_"), F.md5(F.col("_key"))))
        .withColumn("title", F.col("title").cast(StringType()))
        .withColumn("year", F.col("year").cast(StringType()))
)

edges_raw = (df
                .select("paper_id", "_key", "title", "year", "authors")
                .withColumn("author_name", F.explode("authors"))
                .withColumn("author_name", F.trim(F.col("author_name")))
                .withColumn("author_id",   F.concat(F.lit("a_"), F.md5("author_name")))
)


# BUILDING NEPTUNE BULK LOADER CSVS AND SAVING THEM TO S3
author_nodes = (edges_raw
    .select(F.col("author_id").alias("~id"),
            F.lit("author").alias("~label"),
            F.col("author_name").alias("name:String"))
    .distinct())

paper_nodes = (df
    .select(F.col("paper_id").alias("~id"),
            F.lit("paper").alias("~label"),
            F.col("title").alias("title:String"),
            F.col("year").alias("year:String"),
            F.col("_key").alias("dblp_key:String"))
    .distinct())

edges_df = (edges_raw
    .select(
        F.concat(F.lit("e_"), F.md5(
            F.concat("author_id", F.lit("|"), "paper_id")
        )).alias("~id"),
        F.col("author_id").alias("~from"),
        F.col("paper_id").alias("~to"),
        F.lit("authored").alias("~label"))
    .distinct())


# Write as single CSV files (coalesce(1)) so Neptune bulk loader can read them easily
author_nodes.coalesce(1).write.csv(S3_OUTPUT + "author_nodes/", header=True, mode="overwrite")
paper_nodes.coalesce(1).write.csv(S3_OUTPUT + "paper_nodes/", header=True, mode="overwrite")
edges_df.coalesce(1).write.csv(S3_OUTPUT + "edges/", header=True, mode="overwrite")