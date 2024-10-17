from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("Final-Project").getOrCreate()

if __name__ == "__main__":
    
    sc = spark.sparkContext
    # Loading in data
    df = spark.read.csv(sys.argv[1], header=True, sep='\t')
    print("Original DataFrame:")
    df.show()

    # Data Preprocessing:
     
    ## Drop Rows with Missing Values
    cleaned_data = df.na.drop()

    ## One-Hot Encoding Categorical Variables
    categorical_columns = ['Education', 'Marital_Status']
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_columns]
    encoders = [OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[f"{col}_ohe"]) for col in categorical_columns]

    # Converting All Other String Columns to Float
    for column in df.columns:
        if column not in categorical_columns:
            df = df.withColumn(column, df[column].cast("float"))
    
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    df = model.transform(df)
    
    ## Assembling Features
    flattened_columns = []
    for col in categorical_columns:
        flattened_columns.append(f"{col}_ohe")
    numeric_columns = [col for col in df.columns if col not in categorical_columns + [f"{col}_index" for col in categorical_columns]]
    assembler = VectorAssembler(inputCols=flattened_columns, outputCol="features")
    featuresData = assembler.transform(df)
    featuresData.select("features").show(truncate=False)

    # Training/Testing Split for Data
    train, test = featuresData.randomSplit([0.8, 0.2], seed=51)
    
    ## PCA for Dimensionality Reduction
    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(train)
    trainPCA = pca_model.transform(train)
    testPCA = pca_model.transform(test)

    ## Elbow Method for Choosing Number of Clusters and K-Means Clustering with Inertia Calculation
    inertia_values = []
    k_values = range(2, 11)
    for k in k_values:
        kmeans = KMeans(k=k, seed=1)
        model = kmeans.fit(trainPCA)
        inertia_values.append(model.summary.trainingCost)

    # Plotting Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid()
    plt.savefig("elbowmethod.pdf", format='pdf')

    ## K-Means Clustering with k=7
    k = 7
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(trainPCA)  
    predictions = model.transform(testPCA)  

    # Evaluate the clustering using Silhouette Score
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions) 
    print(f"Silhouette Score for k={k}: {silhouette}")

    # Visualizing Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.bar([f'Cluster {i}' for i in range(k)], predictions.groupBy('prediction').count().orderBy('prediction').rdd.map(lambda row: row[1]).collect())
    plt.title("Silhouette Score Visualization for k=7")
    plt.xlabel("Clusters")
    plt.ylabel("Number of Points in Each Cluster")
    plt.grid()
    plt.savefig("silhouetteScore_k7.pdf", format='pdf')

    # Showing Summary Statistics of Clusters 0,1,3 (Focusing on Higher Silhouette Clusters)
    cluster_0_data = predictions.filter(predictions.prediction == 0)
    summary_stats = cluster_0_data.describe()
    summary0 = summary_stats.toPandas().set_index("summary").transpose().round(2)
    print(summary0)
    
    cluster_1_data = predictions.filter(predictions.prediction == 1)
    summary_stats = cluster_1_data.describe()
    summary1 = summary_stats.toPandas().set_index("summary").transpose().round(2)
    print(summary1)

    cluster_3_data = predictions.filter(predictions.prediction == 3)
    summary_stats = cluster_3_data.describe()
    summary3 = summary_stats.toPandas().set_index("summary").transpose().round(2)
    print(summary3)
    



spark.stop()