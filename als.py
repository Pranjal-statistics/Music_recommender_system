from pyspark.ml.evaluation import RegressionEvaluator,RankingEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import explode
from pyspark.sql.functions import udf, col, rank
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
import sys 
import os
import getpass

from pyspark.sql import SparkSession,Window


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import pyspark.sql.functions as func

import pandas as pd
import time


# def main(spark, userID):

#   train_df = spark.read.parquet("hdfs:/user/ps4379_nyu_edu/train_small.parquet")
#   val_df = spark.read.parquet("hdfs:/user/ps4379_nyu_edu/validation_small.parquet")

#   train_df.createOrReplaceTempView('train_df')
#   val_df.createOrReplaceTempView('val_df')
  
#   train_df = train_df.withColumn('user_id', col('user_id').cast('integer')).withColumn('item_msid', col('item_msid').cast('integer')).withColumn('listens', col('listens').cast('float'))
#   val_df = val_df.withColumn('user_id', col('user_id').cast('integer')).withColumn('item_msid', col('item_msid').cast('integer')).withColumn('listens', col('listens').cast('float'))
    
#   val_users = val_df.select('user_id').distinct()
#   val_users.show(20)
  
#   hyper_param_reg = [0.01]#,0.01,0.1,1]
#   hyper_param_rank = [200]#,20,100,200,400]
#   for i in hyper_param_reg:
#       for j in hyper_param_rank:
#         als = ALS(maxIter=20, regParam= i, userCol="user_id", itemCol="item_msid", ratingCol="listens", coldStartStrategy="drop", rank = j)
#         model = als.fit(train_df)
#         predictions = model.recommendForUserSubset(val_users, 100)
#         predictions.createOrReplaceTempView("predictions")
#         predictions = predictions.withColumn("item_msid",col("recommendations.item_msid"))
#         predictions.createOrReplaceTempView("predictions")
#         #predictions.printSchema()
#         #predictions.show(10)

#         #comparision with ground truth
#         groundtruth = val_df.groupby('user_id').agg(F.collect_list('item_msid').alias('groundtruth'))
#         groundtruth.createOrReplaceTempView("groundtruth")
#         total = spark.sql("SELECT g.user_id, g.groundtruth AS groundtruth, p.item_msid AS predictions FROM groundtruth g INNER JOIN predictions p ON g.user_id = p.user_id")
#         total.createOrReplaceTempView("total")
      
#         pandasDF = total.toPandas()
      
#         eval_list = []
#         for index, row in pandasDF.iterrows():
#             eval_list.append((row['predictions'], row['groundtruth']))
           
#         sc =  SparkContext.getOrCreate()
     
#         #Evaluation on val and test
#         predictionAndLabels = sc.parallelize(eval_list)
#         metrics = RankingMetrics(predictionAndLabels)
            
#         print(metrics.precisionAt(100))
#         print(metrics.meanAveragePrecision)
#         print(metrics.ndcgAt(100))
#         import getpass

# from pyspark.sql import SparkSession,Window
# from pyspark.ml.evaluation import RegressionEvaluator,RankingEvaluator
# from pyspark.ml.recommendation import ALS
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# from pyspark.sql.functions import col,rank
# import pyspark.sql.functions as func

# import pandas as pd
# import time

def main(spark, userID):
    train_df = spark.read.parquet("hdfs:/user/ps4379_nyu_edu/train_small.parquet")
    val_df = spark.read.parquet("hdfs:/user/ps4379_nyu_edu/validation_small.parquet")
    test_df = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/1004-project-2023/interactions_test.parquet')
    
    train_df = train_df.where(col("listens") > 1)
    val_df = val_df.where(col("listens") > 1)
    
    train_df = train_df.withColumn('user_id', col('user_id').cast('integer')).withColumn('recording_msid', col('recording_msid').cast('integer')).withColumn('listens', col('listens').cast('float'))
    val_df = val_df.withColumn('user_id', col('user_id').cast('integer')).withColumn('recording_msid', col('recording_msid').cast('integer')).withColumn('listens', col('listens').cast('float'))
    
    train_df.createOrReplaceTempView('train_df')
    val_df.createOrReplaceTempView('val_df')
    print('train_df')
    train_df.printSchema()
    print('val_df')
    val_df.printSchema()
    print('test_df')
    test_df.printSchema()
    
    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(userCol="user_id", itemCol="recording_msid", ratingCol="listens", nonnegative = True, coldStartStrategy="drop")
   
    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder().addGrid(als.rank, [20, 40, 50]).addGrid(als.regParam, [.01, 0.05, 0.1]).addGrid(als.maxIter, [10, 15]).build()


    # Define evaluator as RMSE 
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="listens", predictionCol="prediction") 
    
    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5,parallelism=2)

    #Fit cross validator to the training dataset
    model = cv.fit(train_df)
    #fetch the best model
    best_model = model.bestModel
    predictions = best_model.transform(val_df)
    rmse = evaluator.evaluate(predictions)
    
    print('RMSE = ' + str(rmse))
    print('*******Best Model********')
    print('rank: ', best_model.rank)
    print('maxIter:', best_model._java_obj.parent().getMaxIter())
    print('RegParam:', best_model._java_obj.parent().getRegParam())   
              
if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('als').getOrCreate()
    # Get file path for the dataset to split
    userID = os.environ['USER']
    # Calling the split function
    main(spark, userID)
