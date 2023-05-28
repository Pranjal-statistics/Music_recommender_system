from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import count
import sys
import os

def partition(spark,userID):

    #interactions_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    interactions_train = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    
    #tracks_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
    tracks_train = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet')
   
    #print('Printing interactions_train_small schema')
    #interactions_train_small.printSchema()
    #interactions_train_small.createOrReplaceTempView('interactions_train_small')
    
    print('Printing interactions_train schema')
    interactions_train.printSchema()
    interactions_train.createOrReplaceTempView('interactions_train')
    
    #print('Printing tracks_train_small schema')
    #tracks_train_small.printSchema()
    #tracks_train_small.createOrReplaceTempView('tracks_train_small')
    
    print('Printing tracks_train schema')
    tracks_train.printSchema()
    tracks_train.createOrReplaceTempView('tracks_train')
    
    print('Data loaded and dataframe created')
    
    #df = spark.sql('select int.*, trk.recording_mbid from interactions_train_small int, tracks trk where int.recording_msid = trk.recording_msid')
    
    df = spark.sql('select * from interactions_train')
    df = df.sort('user_id')
    df = df.repartition(10, 'user_id')
    df = df.dropDuplicates()
    
    print("Repartition done.")
    
    (trainUserId, validationUserId) =  df.select('user_id').distinct().randomSplit([0.6, 0.4])
    
    print("Randon splitting done in 60 - 40 % .")
    
    train_user_id = [x.user_id for x in trainUserId.collect()]
    validation_user_id = [x.user_id for x in validationUserId.collect()]
    
    train = df.filter(F.col('user_id').isin(train_user_id))
    validation = df.filter(F.col('user_id').isin(validation_user_id))
    
    validation_count = validation.select(count("user_id")).collect()
    print("Validation count is: ", validation_count)
    
    (validation_train_interactions,_) = validation.randomSplit([0.6, 0.4])
    train.union(validation_train_interactions)
    
    train_count = train.select(count("user_id")).collect()
    print("train count is: ", train_count)
   
    train.write.parquet("hdfs:/user/ps4379_nyu_edu/train_interaction.parquet")
    validation.write.parquet("hdfs:/user/ps4379_nyu_edu/validation_interaction.parquet")
    
    print('Train and validation partitioning done and finished writing parquet files')
   
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get file path for the dataset to split
    userID = os.environ['USER']

    # Calling the split function
    partition(spark, userID)
    
    
