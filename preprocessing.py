# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as F
import sys
import os
from pyspark.sql.functions import count, countDistinct, desc, first, monotonically_increasing_id, col , broadcast
from pyspark.sql.window import Window
import random


# Function to define alternating least squares (ALS) method
def preprocessing(spark, userID):
  
  #import dataframes
  interactions_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
  tracks_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
  interactions_train_small.createOrReplaceTempView('interactions_train_small')
  tracks_train_small.createOrReplaceTempView('tracks_train_small')
  #print('interactions_train_small schema')
  #interactions_train_small.printSchema()
  #print('tracks_train_small schema')
  #tracks_train_small.printSchema()
  
  tracks_train_small = tracks_train_small.drop('__index_level_0__', 'artist_name', 'track_name')
  #print('tracks_train_small schema after dropping columns')
  #tracks_train_small.printSchema()  
  
  #combine the interactions and tracks dataframes
  df = interactions_train_small.join(tracks_train_small, on = 'recording_msid')
  df.createOrReplaceTempView('df')
  #print('df schema')
  #df.show()
  
  df = df.sort('user_id')
  df = df.repartition(1000, 'user_id')

  recording_mbid_df = df.where(col("recording_mbid").isNotNull())
  recording_msid_df = df.where(col("recording_mbid").isNull())
  
  recording_mbid_df.createOrReplaceTempView('recording_mbid_df')
  #print('recording_mbid_df schema')
  #recording_mbid_df.printSchema()
  #print('recording_msid_df schema')
  recording_msid_df.createOrReplaceTempView('recording_msid_df')
  #recording_msid_df.printSchema()
  
  # Determine the number of partitions you want to use
  num_partitions = 1000
  
  # Repartition the DataFrame using the desired number of partitions
  recording_mbid_df = recording_mbid_df.repartition(num_partitions, "recording_mbid")
  
  # join the original DataFrame with the new DataFrame on recording_mbid
  window_spec = Window.partitionBy("recording_mbid").orderBy("timestamp")
  df_with_first_msid = recording_mbid_df.withColumn("first_msid", first("recording_msid").over(window_spec))
 
  #print('df_with_first_msid schema')
  df_with_first_msid.createOrReplaceTempView('df_with_first_msid')
  #df_with_first_msid.printSchema()
  #df_with_first_msid.show()
  
  # df_one_to_one solves recording_msid & recording_mbid relation
  df_one_to_one = recording_mbid_df.join(df_with_first_msid.select("recording_mbid", "first_msid"), on="recording_mbid")
  df_one_to_one = df_one_to_one.drop('recording_mbid','recording_msid')
  df_one_to_one = df_one_to_one.dropDuplicates()
  df_one_to_one = df_one_to_one.withColumnRenamed("first_msid", "recording_msid")
  
  #print('df_one_to_one schema')
  df_one_to_one.createOrReplaceTempView('df_one_to_one')
  #df_one_to_one.printSchema()
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
  #final_df = final_df.filter(final_df.listens > 1)
  final_df.createOrReplaceTempView('final_df')
  print('printing final_df schema')
  final_df.printSchema()
  #final_df.show(10)
  
  print('Getting a list of unique user_ids in the data')
  # Get a list of unique user_ids in the data
  user_ids = final_df.select('user_id').distinct().rdd.map(lambda r: r[0]).collect()
  
  print('Randomly assigning user_ids to either train or val')
  # Randomly assign user_ids to either 'train' or 'val'
  random.shuffle(user_ids)
  split_idx = int(len(user_ids) * 0.7)
  train_user_ids = set(user_ids[:split_idx])
  val_user_ids = set(user_ids[split_idx:])
  
  print('Add a new column split to final_df, where each row is assigned to either train or val based on user_id')
  # Add a new column 'split' to final_df, where each row is assigned to either 'train' or 'val' based on user_id
  final_df = final_df.withColumn('split', F.when(F.col('user_id').isin(train_user_ids), 'train').otherwise('val'))
  
  print('Use the new split column to split final_df into train and validation sets')
  # Use the new 'split' column to split final_df into train and validation sets
  train_df = final_df.filter(F.col('split') == 'train').drop('split').orderBy(F.col('user_id'), F.col('recording_msid'))
  val_df = final_df.filter(F.col('split') == 'val').drop('split').orderBy(F.col('user_id'), F.col('recording_msid'))
  (validation_train_interactions,_) = val_df.randomSplit([0.6, 0.4])
  train_df = train_df.union(validation_train_interactions)
  
   
#   (trainUserId, validationUserId) =  final_df.select('user_id').randomSplit([0.8, 0.2])
    
#   print("Randon splitting done in 80 - 20 % .")
#   print ("Now working on saving the data into parquet files.")
    

#   train_user_id = trainUserId.rdd.map(lambda x: x.user_id).collect()
#   validation_user_id = validationUserId.rdd.map(lambda x: x.user_id).collect()
  
#   # Broadcast the lists
#   sc =  SparkContext.getOrCreate()
#   broadcast_train_user_id = sc.broadcast(train_user_id)
#   broadcast_validation_user_id = sc.broadcast(validation_user_id)

#   print ("train filtering started.")
#   train = final_df.filter(F.col('user_id').isin(train_user_id))
#   print ("train filtering finished, validation filtering started.")
#   validation = final_df.filter(F.col('user_id').isin(validation_user_id))
  
  
  #print('about to print train and val count')
  #train_count = train_df.count()
  #validation_count = val_df.count()
  #print("train count is: ", train_count)
  #print("Validation count is: ", validation_count)
  
  print('about to start writing into hdfs')
  ## train_df.write.csv("hdfs:/user/ps4379_nyu_edu/train_small.csv")
  val_df.write.csv("hdfs:/user/ps4379_nyu_edu/validation_small.csv")
    
  print('Train and validation partitioning done and finished writing parquet files')
  
if __name__ == "__main__": 
    # Create Spark session object
    spark = SparkSession.builder.appName('preprocessing').getOrCreate()
    # Get file path for the dataset to split
    userID = os.environ['USER']
    # Calling the split function
    preprocessing(spark, userID)
