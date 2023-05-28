from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from annoy import AnnoyIndex

def main(spark, als_filepath, n_trees, search_k):
    # Load the ALS model
    als = ALSModel.load(als_filepath)
    
    # Extract item and user latent factors
    item_lfs = als.itemFactors
    user_lfs = als.userFactors

    # New index
    user_lfs = user_lfs.withColumn("index", row_number().over(Window.partitionBy().orderBy("id")))
    item_lfs = item_lfs.withColumn("index", row_number().over(Window.partitionBy().orderBy("id")))

    # Load validation data
    validation_data = spark.read.parquet("hdfs:/user/ps4379_nyu_edu/validation_small.parquet")
    validation_data = validation_data.withColumnRenamed('_c0', 'user_id').withColumnRenamed('_c1', 'recording_msid').withColumnRenamed('_c2', 'listens')

    # Create the Annoy index
    f = len(item_lfs.first().features)
    t = AnnoyIndex(f, 'dot')

    for item in item_lfs.collect():
        t.add_item(item['index'], item['features'])

    t.build(n_trees)

    # Generate recommendations and evaluate on validation data
    user_rec = {}
    for user in user_lfs.collect():
        user_rec[user['id']] = t.get_nns_by_vector(user['features'], 10)

    # Convert user_rec to DataFrame
    user_rec_df = spark.createDataFrame([(k, Vectors.dense(v)) for k, v in user_rec.items()], ["user_id", "recommendations"])

    # Calculate metrics
    binary_evaluator = BinaryClassificationEvaluator(labelCol="listens")
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="listens")

    predictions = user_rec_df.join(validation_data, on="user_id", how="inner")

    accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    # Query time measurement
    query_start = time.time()
    for user in user_rec_df.collect():
        t.get_nns_by_vector(user['recommendations'], search_k)
    query_end = time.time()
    query_time = query_end - query_start

    print(f"Query Time: {query_time:.4f} seconds")


if __name__ == "__main__":
    spark = SparkSession.builder.appName('FastSearch').getOrCreate()
    als_filepath = sys.argv[1]
    n_trees = int(sys.argv[2])
    search_k = int(sys.argv[3])
    main(spark, als_filepath=als_filepath, n_trees=n_trees, search_k=search_k)