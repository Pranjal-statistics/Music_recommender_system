import getpass
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import desc
from pyspark.sql.functions import count, countDistinct
import os
import sys
from pyspark.sql.functions import count, countDistinct, desc, first, monotonically_increasing_id, col , broadcast, percentile_approx, avg, max, min
import random

# Function to define alternating least squares (ALS) method
def preprocessing(spark, userID):
  
  #import dataframes
  interactions_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
  tracks_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
  interactions_train_small.createOrReplaceTempView('interactions_train_small')
  tracks_train_small.createOrReplaceTempView('tracks_train_small')
  print('interactions_train_small schema')
  interactions_train_small.printSchema()
  print('tracks_train_small schema')
  tracks_train_small.printSchema()
  
  tracks_train_small = tracks_train_small.drop('__index_level_0__', 'artist_name', 'track_name')
  print('tracks_train_small schema after dropping columns')
  tracks_train_small.printSchema()  
  
  #combine the interactions and tracks dataframes
  df = interactions_train_small.join(tracks_train_small, on = 'recording_msid')
  df.createOrReplaceTempView('df')
  print('df schema')
  #df.show()
  
  df = df.sort('user_id')
  df = df.repartition(1000, 'user_id')

  recording_mbid_df = df.where(col("recording_mbid").isNotNull())
  recording_msid_df = df.where(col("recording_mbid").isNull())
  
  recording_mbid_df.createOrReplaceTempView('recording_mbid_df')
  print('recording_mbid_df schema')
  recording_mbid_df.printSchema()
  print('recording_msid_df schema')
  recording_msid_df.createOrReplaceTempView('recording_msid_df')
  recording_msid_df.printSchema()
  
  # Determine the number of partitions you want to use
  num_partitions = 1000
  
  # Repartition the DataFrame using the desired number of partitions
  recording_mbid_df = recording_mbid_df.repartition(num_partitions, "recording_mbid")
  
  # join the original DataFrame with the new DataFrame on recording_mbid
  window_spec = Window.partitionBy("recording_mbid").orderBy("timestamp")
  df_with_first_msid = recording_mbid_df.withColumn("first_msid", first("recording_msid").over(window_spec))
 
  print('df_with_first_msid schema')
  df_with_first_msid.createOrReplaceTempView('df_with_first_msid')
  df_with_first_msid.printSchema()
  #df_with_first_msid.show()
  
  # df_one_to_one solves recording_msid & recording_mbid relation
  df_one_to_one = recording_mbid_df.join(df_with_first_msid.select("recording_mbid", "first_msid"), on="recording_mbid")
  df_one_to_one = df_one_to_one.drop('recording_mbid','recording_msid')
  df_one_to_one = df_one_to_one.dropDuplicates()
  df_one_to_one = df_one_to_one.withColumnRenamed("first_msid", "recording_msid")
  
  print('df_one_to_one schema')
  df_one_to_one.createOrReplaceTempView('df_one_to_one')
  df_one_to_one.printSchema()
  #print('printing df_one_to_one')
  #df_one_to_one.show(10)
  
  df_one_to_one = df_one_to_one.groupBy('user_id','recording_msid').agg(count('timestamp').alias('listens')).orderBy(desc('listens'))
  recording_msid_df = recording_msid_df.groupBy('user_id','recording_msid').agg(count('timestamp').alias('listens')).orderBy(desc('listens'))
  
  final_df = df_one_to_one.union(recording_msid_df)
  item_df = final_df.select('recording_msid').distinct()
  item_df = item_df.withColumn('item_msid', monotonically_increasing_id())
  final_df = final_df.join(item_df, on = 'recording_msid')
  final_df = final_df.drop('recording_msid')
  final_df = final_df.withColumnRenamed('item_msid','recording_msid')
  final_df.createOrReplaceTempView('final_df')
  print('printing final_df schema')
  final_df.printSchema()
  #print('printing final_df')
  #final_df.show(10)
    
  # group the data by the "listens" column and compute the median
  median_listens = final_df.selectExpr("percentile_approx(listens, 0.5)").collect()[0][0]
  # print the median
  print("Median listens: ", median_listens)
  
  # compute the average, maximum, and minimum of the "listens" column
  agg_df = final_df.agg(avg("listens"), max("listens"), min("listens"))

  # extract the results from the aggregation DataFrame
  avg_listens = agg_df.collect()[0][0]
  max_listens = agg_df.collect()[0][1]
  min_listens = agg_df.collect()[0][2]

  # print the results
  print("Average listens: ", avg_listens)
  print("Maximum listens: ", max_listens)
  print("Minimum listens: ", min_listens)
  
  total_rows = final_df.count()
  listens_1_count = final_df.filter(final_df.listens == 1).count()
  print("total rows: ", total_rows)
  print("total number of rows that have listens = 1: ", listens_1_count)
  
  

def test(spark, userID):
    """
    Returns a dataframe with the top 100 most popular songs based on raw counts per recording_msid.
    """
    
    train_small = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/train_small.parquet')
    
    print('Printing train_small schema')
    train_small.printSchema()
    
    train_small.createOrReplaceTempView('train_small')
    
    total_users_in_train_small = spark.sql('select count(distinct user_id) from train_small')
    total_users_in_train_small.show()

    total_msid_in_train_small_distinct = spark.sql('select count(distinct recording_msid) from train_small')
    total_msid_in_train_small_distinct.show()
    
    total_msid_in_train_small = spark.sql('select count(recording_msid) from train_small')
    total_msid_in_train_small.show()
    
def baseline2(spark, userID):
    """
        Baseline popularity model -- version2 
    """
    #read dataset
    train_small = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/train_small.parquet')
    #train = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/train.parquet')
    validation_small = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/validation_small.parquet')
    #test = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    #validation = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/validation.parquet')
    
    #print dataset schema
    print('Printing train_small schema')
    train_small.printSchema()
    print('Printing validation_small schema')
    validation_small.printSchema()
    #print('Printing test schema')
    #test.printSchema()
    
    train_small.createOrReplaceTempView('train_small')
    validation_small.createOrReplaceTempView('validation_small')
    
    
    #calculate total number of listens of a song by a user on the entire dataset
    total_listens = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM train_small GROUP BY user_id, recording_msid order by user_id, total_listens desc")
    total_listens.createOrReplaceTempView('total_listens')
    
    average_listens = spark.sql("select recording_msid, sum(total_listens)/count(recording_msid) as avg_listens_per_user from total_listens group by recording_msid order by avg_listens_per_user desc limit 100")
    average_listens.createOrReplaceTempView('average_listens')
    final_listens = (train_small.select('recording_msid').distinct()).crossJoin(train_small.select('user_id').distinct())
    final_listens = final_listens.join(average_listens,['recording_msid'], 'left').select('user_id', 'recording_msid', 'avg_listens_per_user').dropna()
    final_listens.show()
    
def validation(spark, userID):
    val_df = spark.read.parquet("hdfs:/user/vr2229_nyu_edu/validation_small.parquet")
    val_df.createOrReplaceTempView('val_df')
    val_df.printSchema()
    val_df.show(20)
    
    val_users = val_df.select('user_id').distinct()
    print('Number of users in val_users dataframe:', val_users.count())
    
    train_df = spark.read.parquet("hdfs:/user/vr2229_nyu_edu/train_small.parquet")
    train_df.printSchema()
    train_df.show(20)
    
    train_users = train_df.select('user_id').distinct()
    print('Number of users in train_users dataframe:', train_users.count())

def common(spark, userID):
    # Read the parquet files into PySpark DataFrames
    train_df = spark.read.parquet(f'hdfs:/user/vr2229_nyu_edu/train_small.parquet')
    print('train df')
    train_df.printSchema()
    val_df = spark.read.parquet(f'hdfs:/user/vr2229_nyu_edu/validation_small.parquet')
    print('val df')
    val_df.printSchema()
    test_df = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    print('test df')
    test_df.printSchema()

    # Count the number of distinct user IDs in each DataFrame
    train_users = train_df.select(countDistinct("user_id")).collect()[0][0]
    val_users = val_df.select(countDistinct("user_id")).collect()[0][0]
    test_users = test_df.select(countDistinct("user_id")).collect()[0][0]

    # Find the intersection of user IDs in both DataFrames
    common_users_val = train_df.select("user_id").intersect(val_df.select("user_id")).count()
    common_users_test = train_df.select("user_id").intersect(test_df.select("user_id")).count()

    # Print the results
    print(f"Number of user IDs in train_df: {train_users}")
    print(f"Number of user IDs in val_df: {val_users}")
    print(f"Number of user IDs in test_df: {test_users}")
    print(f"Number of user IDs in both train_df and val_df: {common_users_val}")
    print(f"Number of user IDs in both train_df and test_df: {common_users_test}")
    
    total_rows_train = train_df.count()
    total_rows_val = val_df.count()
    total_rows_test = test_df.count()
    listens_1_count_train = train_df.filter(train_df.listens == 1).count()
    listens_1_count_val = val_df.filter(val_df.listens == 1).count()

    print("total rows in train: ", total_rows_train)
    print("total rows in val: ", total_rows_val)
    print("total rows in test: ", total_rows_test)
    print("train total number of rows that have listens = 1: ", listens_1_count_train)
    print("val total number of rows that have listens = 1: ", listens_1_count_val)

    
if __name__ == "__main__": 
    # Create Spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()
    # Get file path for the dataset to split
    userID = os.environ['USER']
    common(spark, userID)
