import getpass
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import desc, count, col, countDistinct, sum

import os

def baseline(spark, userID):
    """
        Baseline popularity model -- version 1
    """
    #read dataset
    train_interaction_small = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/train_interaction_small.parquet')
    train_interaction = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/train_interaction.parquet')
    validation_interaction_small = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/validation_interaction_small.parquet')
    validation_interaction = spark.read.parquet(f'hdfs:/user/ps4379_nyu_edu/validation_interaction.parquet')
    interactions_test = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    
    #create temp view
    train_interaction_small.createOrReplaceTempView('train_interaction_small')
    train_interaction.createOrReplaceTempView('train_interaction')
    validation_interaction_small.createOrReplaceTempView('validation_interaction_small')
    validation_interaction.createOrReplaceTempView('validation_interaction')
    interactions_test.createOrReplaceTempView('interactions_test')
    
    #calculate total number of listens of a song for each user on the entire dataset
    #listen_train_small = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM train_interaction_small GROUP BY user_id, recording_msid order by user_id, total_listens desc")
    listen_val_small = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM validation_interaction_small GROUP BY user_id, recording_msid order by user_id, total_listens desc")
    #listen_train = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM train_interaction GROUP BY user_id, recording_msid order by user_id, total_listens desc")
    listen_val = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM validation_interaction GROUP BY user_id, recording_msid order by user_id, total_listens desc")
    listen_test = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS total_listens FROM interactions_test GROUP BY user_id, recording_msid order by user_id, total_listens desc")

    #Get the list of top 100 most played recording_msid for each user by rank
    val_rank_small = listen_val_small.withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy(F.col("total_listens").desc()))).filter(f"rn <= {100}").groupBy("user_id").agg(F.collect_list(F.col("recording_msid")).alias("user_recording_msid")).orderBy("user_id")
    val_rank = listen_val.withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy(F.col("total_listens").desc()))).filter(f"rn <= {100}").groupBy("user_id").agg(F.collect_list(F.col("recording_msid")).alias("user_recording_msid")).orderBy("user_id")
    test_rank = listen_test.withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy(F.col("total_listens").desc()))).filter(f"rn <= {100}").groupBy("user_id").agg(F.collect_list(F.col("recording_msid")).alias("user_recording_msid")).orderBy("user_id") 
    
    #create temp view of validation and test ranking
    val_rank_small.createOrReplaceTempView('val_rank_small')
    val_rank.createOrReplaceTempView('val_rank')
    test_rank.createOrReplaceTempView('test_rank')
    
    #print ('validation rank small:')
    #val_rank_small.show(20)
    #print ('validation rank:')
    #val_rank.show(20)
    #print ('test rank:')
    #test_rank.show(20)
    
    #calculate average listens of each song on the entire dataset and fetch top 100 songs
    #average_listens = spark.sql("select recording_msid, sum( total_listens)/count(recording_msid) as avg_listens_per_user from total_listens_per_song_per_user_train group by recording_msid order by avg_listens_per_user desc limit 100")
    average_listens_small = (train_interaction_small.groupBy("recording_msid").agg(count("recording_msid").alias("total_count"), countDistinct("user_id").alias("distinct_user")).withColumn("Avg_Listen", col("total_count") / col("distinct_user")).orderBy(col("Avg_Listen").desc()).limit(100).select("recording_msid", "Avg_Listen"))
    average_listens = (train_interaction.groupBy("recording_msid").agg(count("recording_msid").alias("total_count"), countDistinct("user_id").alias("distinct_user")).withColumn("Avg_Listen", col("total_count") / col("distinct_user")).orderBy(col("Avg_Listen").desc()).limit(100).select("recording_msid", "Avg_Listen"))
    #average_listens_small = spark.sql("select recording_msid, (sum(count(recording_msid))/count(distinct user_id)) as Avg_Listen from train_interaction_small group by recording_msid order by Avg_Listen desc limit 100")
    #average_listens = spark.sql("select recording_msid, (sum(count(recording_msid))/count(distinct user_id)) as Avg_Listen from train_interaction group by recording_msid order by Avg_Listen desc limit 100")
    average_listens_small.createOrReplaceTempView('average_listens_small')
    average_listens.createOrReplaceTempView('average_listens')

    
    #calculate the top 100 listened songs from the entire dataset
    total_listens_small = spark.sql("select recording_msid, count(*) from train_interaction_small group by recording_msid order by 2 desc limit 100")
    total_listens = spark.sql("select recording_msid, count(*) from train_interaction group by recording_msid order by 2 desc limit 100")
    total_listens_small.createOrReplaceTempView('total_listens_small')
    total_listens.createOrReplaceTempView('total_listens')
   
    #get the list of top 100 listened songs
    top_100_songs_small = total_listens_small.agg(F.collect_list(F.col("recording_msid")).alias("predicted_recording_msid"))
    top_100_songs = total_listens.agg(F.collect_list(F.col("recording_msid")).alias("predicted_recording_msid"))
    
    top_100_songs_avg_small = average_listens_small.agg(F.collect_list(F.col("recording_msid")).alias("predicted_recording_msid"))
    top_100_songs_avg = average_listens.agg(F.collect_list(F.col("recording_msid")).alias("predicted_recording_msid"))
    
    #calculate the predicted songs
    predicted_songs_val_small = val_rank_small.select("user_recording_msid").crossJoin(top_100_songs_small.select("predicted_recording_msid"))
    predicted_songs_val = val_rank.select("user_recording_msid").crossJoin(top_100_songs.select("predicted_recording_msid"))
    predicted_songs_test = test_rank.select("user_recording_msid").crossJoin(top_100_songs.select("predicted_recording_msid"))
    
    predicted_songs_val_avg_small = val_rank_small.select("user_recording_msid").crossJoin(top_100_songs_avg_small.select("predicted_recording_msid"))
    predicted_songs_val_avg = val_rank.select("user_recording_msid").crossJoin(top_100_songs_avg.select("predicted_recording_msid"))
    predicted_songs_test_avg = test_rank.select("user_recording_msid").crossJoin(top_100_songs_avg.select("predicted_recording_msid"))
    
    #print ('Predicted songs list for test_ranking:')
    #predicted_songs_test.show()
    
    print("Metrics of popularity baseline model with total listen: ")

    # Instantiate ranking metrics object
    ranking_metrics_val_small = RankingMetrics(predicted_songs_val_small.rdd)
    ranking_metrics_val = RankingMetrics(predicted_songs_val.rdd)
    ranking_metrics_test = RankingMetrics(predicted_songs_test.rdd)
    
    ranking_metrics_val_avg_small = RankingMetrics(predicted_songs_val_avg_small.rdd)
    ranking_metrics_val_avg = RankingMetrics(predicted_songs_val_avg.rdd)
    ranking_metrics_test_avg = RankingMetrics(predicted_songs_test_avg.rdd)
    
    #calculate MAP@100 
    print("Popularity Baseline Mean Average Precision at 100 for val_small split = %s" % ranking_metrics_val_small.meanAveragePrecisionAt(100))
    print("Popularity Baseline Mean Average Precision at 100 for val split = %s" % ranking_metrics_val.meanAveragePrecisionAt(100))
    print("Popularity Baseline Mean Average Precision at 100 for test = %s" % ranking_metrics_test.meanAveragePrecisionAt(100))
    
    #calculate NDCG@100
    print("Popularity Baseline NDCG at 100 for val_small split = %s" % ranking_metrics_val_small.ndcgAt(100))
    print("Popularity Baseline NDCG at 100 for val split = %s" % ranking_metrics_val.ndcgAt(100))
    print("Popularity Baseline NDCG at 100 for test split = %s" % ranking_metrics_test.ndcgAt(100))
    
    print("Metrics of popularity baseline model using average listen: ")
    
    #calculate MAP@100 with average listen
    print("Popularity Baseline Mean Average Precision at 100 for val_small split = %s" % ranking_metrics_val_avg_small.meanAveragePrecisionAt(100))
    print("Popularity Baseline Mean Average Precision at 100 for val split = %s" % ranking_metrics_val_avg.meanAveragePrecisionAt(100))
    print("Popularity Baseline Mean Average Precision at 100 for test = %s" % ranking_metrics_test_avg.meanAveragePrecisionAt(100))
    
    #calculate NDCG@100 with average listen
    print("Popularity Baseline NDCG at 100 for val_small split = %s" % ranking_metrics_val_avg_small.ndcgAt(100))
    print("Popularity Baseline NDCG at 100 for val split = %s" % ranking_metrics_val_avg.ndcgAt(100))
    print("Popularity Baseline NDCG at 100 for test split = %s" % ranking_metrics_test_avg.ndcgAt(100))
    
  
if __name__ == "__main__": 
    # Create Spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()
    # Get file path for the dataset to split
    userID = os.environ['USER']
    baseline(spark, userID)
