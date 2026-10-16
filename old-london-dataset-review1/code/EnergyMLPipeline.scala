//> using scala "2.12.18"
//> using dep "org.apache.spark::spark-sql:3.5.6"
//> using dep "org.apache.spark::spark-mllib:3.5.6"
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector


object EnergyMLPipeline {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("London Smart Meter Analytics and ML")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ============================================================
    // PATHS
    // ============================================================

    val rawInputPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/daily_dataset.csv"

    val inputPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/energy_daily"

    val predictionPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/predictions"

    val anomalyPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/anomalies"

    val houseDailyPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily"

    val housePredictionPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_predictions"


    // ============================================================
    // 1. LOAD PROCESSED AGGREGATE DATA FROM HDFS
    // ============================================================

    println("==============================================")
    println(" LONDON SMART METER ML PIPELINE")
    println("==============================================")

    println("\n[1] Loading processed data from HDFS...")

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    println(s"Rows loaded: ${data.count()}")

    data.printSchema()


    // ============================================================
    // 2. BASIC DATA VALIDATION
    // ============================================================

    println("\n[2] DATA VALIDATION")

    data.select(
      min("day").alias("start_date"),
      max("day").alias("end_date")
    ).show()

    data.select(
      "total_energy",
      "household_count",
      "avg_energy"
    ).describe().show()

    println("Weekend / weekday counts:")

    data.groupBy("is_weekend")
      .count()
      .orderBy("is_weekend")
      .show()


    // ============================================================
    // 3. BASIC ANALYTICS
    // ============================================================

    println("\n[3] BASIC ANALYTICS")

    println("Top 10 energy-consumption days:")

    data.orderBy(desc("total_energy"))
      .select(
        "day",
        "total_energy",
        "household_count",
        "avg_energy"
      )
      .show(10, false)

    println("Weekday vs weekend:")

    data.groupBy("is_weekend")
      .agg(
        avg("total_energy").alias("avg_total_energy"),
        avg("avg_energy").alias("avg_household_energy")
      )
      .orderBy("is_weekend")
      .show()

    println("Monthly average household energy:")

    data.groupBy("year", "month")
      .agg(
        avg("avg_energy").alias("avg_household_energy")
      )
      .orderBy("year", "month")
      .show(50, false)

    println("Correlation between total energy and household count:")

    println(
      data.stat.corr(
        "total_energy",
        "household_count"
      )
    )


    // ============================================================
    // 4. RANDOM FOREST FEATURE ASSEMBLY
    // ============================================================

    println("\n[4] CREATING AGGREGATE ML FEATURES")

    val assembler = new VectorAssembler()
      .setInputCols(
        Array(
          "household_count",
          "month",
          "day_of_month",
          "day_of_week",
          "is_weekend"
        )
      )
      .setOutputCol("features")

    val mlData = assembler
      .transform(data)
      .select(
        "day",
        "features",
        "avg_energy"
      )

    mlData.show(5, false)


    // ============================================================
    // 5. TIME-BASED TRAIN / TEST SPLIT
    // ============================================================

    println("\n[5] TIME-BASED TRAIN / TEST SPLIT")

    // Train on 2011-2013 and test on 2014.

    val trainData =
      mlData.filter("year(day) < 2014")

    val testData =
      mlData.filter("year(day) == 2014")

    println(
      s"Training observations: ${trainData.count()}"
    )

    println(
      s"Testing observations : ${testData.count()}"
    )


    // ============================================================
    // 6. RANDOM FOREST REGRESSION
    // ============================================================

    println("\n[6] TRAINING AGGREGATE RANDOM FOREST")

    val rf = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("avg_energy")
      .setNumTrees(100)
      .setSeed(42)

    val model = rf.fit(trainData)

    println(
      "Random Forest training completed."
    )


    // ============================================================
    // 7. AGGREGATE PREDICTIONS
    // ============================================================

    println("\n[7] GENERATING AGGREGATE PREDICTIONS")

    val predictions =
      model.transform(testData)

    predictions
      .select(
        "day",
        "avg_energy",
        "prediction"
      )
      .orderBy("day")
      .show(59, false)


    // ============================================================
    // 8. AGGREGATE MODEL EVALUATION
    // ============================================================

    println("\n[8] AGGREGATE MODEL EVALUATION")

    val rmse = new RegressionEvaluator()
      .setLabelCol("avg_energy")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      .evaluate(predictions)

    val r2 = new RegressionEvaluator()
      .setLabelCol("avg_energy")
      .setPredictionCol("prediction")
      .setMetricName("r2")
      .evaluate(predictions)

    println(
      f"RMSE = $rmse%.6f"
    )

    println(
      f"R2   = $r2%.6f"
    )


    // ============================================================
    // 9. AGGREGATE FEATURE IMPORTANCE
    // ============================================================

    println("\n[9] AGGREGATE FEATURE IMPORTANCE")

    val featureNames = Array(
      "household_count",
      "month",
      "day_of_month",
      "day_of_week",
      "is_weekend"
    )

    val importances =
      model.featureImportances

    featureNames.zipWithIndex.foreach {
      case (name, index) =>
        println(
          f"$name%-20s ${importances(index)}%.6f"
        )
    }


    // ============================================================
    // 10. ACTUAL VS PREDICTED SUMMARY
    // ============================================================

    println("\n[10] ACTUAL VS PREDICTED SUMMARY")

    predictions
      .agg(
        avg("avg_energy").alias("actual_avg"),
        avg("prediction").alias("predicted_avg")
      )
      .show()


    // ============================================================
    // 11. SAVE AGGREGATE PREDICTIONS TO HDFS
    // ============================================================

    println("\n[11] SAVING AGGREGATE PREDICTIONS TO HDFS")

    predictions
      .select(
        "day",
        "avg_energy",
        "prediction"
      )
      .orderBy("day")
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(predictionPath)

    println(
      "Predictions stored at:"
    )

    println(predictionPath)


    // ============================================================
    // 12. ANOMALY DETECTION USING K-MEANS
    // ============================================================

    println("\n[12] ANOMALY DETECTION")

    println(
      "Preparing features for K-Means anomaly detection..."
    )

    // Energy-related features are used for anomaly detection.
    //
    // 1. Total daily energy
    // 2. Number of households
    // 3. Average household energy
    // 4. Month
    // 5. Day of week
    // 6. Weekend indicator

    val anomalyAssembler =
      new VectorAssembler()
        .setInputCols(
          Array(
            "total_energy",
            "household_count",
            "avg_energy",
            "month",
            "day_of_week",
            "is_weekend"
          )
        )
        .setOutputCol("anomaly_features")

    val anomalyInput =
      anomalyAssembler.transform(data)

    // Standardize the features so that large-valued
    // variables such as total_energy do not dominate
    // the clustering algorithm.

    val scaler =
      new StandardScaler()
        .setInputCol("anomaly_features")
        .setOutputCol("scaled_features")
        .setWithMean(true)
        .setWithStd(true)

    val scalerModel =
      scaler.fit(anomalyInput)

    val scaledData =
      scalerModel.transform(anomalyInput)


    // ============================================================
    // 13. K-MEANS CLUSTERING
    // ============================================================

    println("\n[13] TRAINING K-MEANS")

    val kmeans =
      new KMeans()
        .setK(3)
        .setSeed(42)
        .setFeaturesCol("scaled_features")
        .setPredictionCol("cluster")

    val kmeansModel =
      kmeans.fit(scaledData)

    val clusteredData =
      kmeansModel.transform(scaledData)

    println(
      "K-Means training completed."
    )

    println(
      "Cluster distribution:"
    )

    clusteredData
      .groupBy("cluster")
      .count()
      .orderBy("cluster")
      .show()


    // ============================================================
    // 14. CALCULATE DISTANCE FROM CLUSTER CENTER
    // ============================================================

    println(
      "\n[14] CALCULATING DISTANCE FROM CLUSTER CENTERS"
    )

    val centers =
      kmeansModel.clusterCenters

    val distanceUDF =
      udf { (features: Vector, cluster: Int) =>

        val center = centers(cluster)

        var sum = 0.0

        var i = 0

        while (i < features.size) {

          val difference =
            features(i) - center(i)

          sum +=
            difference * difference

          i += 1
        }

        math.sqrt(sum)
      }

    val anomalyScored =
      clusteredData.withColumn(
        "distance_from_center",
        distanceUDF(
          col("scaled_features"),
          col("cluster")
        )
      )


    // ============================================================
    // 15. ANOMALY THRESHOLD
    // ============================================================

    println(
      "\n[15] CALCULATING ANOMALY THRESHOLD"
    )

    // The top 5% most distant observations
    // are classified as anomalies.

    val threshold =
      anomalyScored
        .stat
        .approxQuantile(
          "distance_from_center",
          Array(0.95),
          0.01
        )(0)

    println(
      f"Anomaly threshold = $threshold%.6f"
    )

    val anomalyResults =
      anomalyScored
        .withColumn(
          "is_anomaly",
          when(
            col("distance_from_center") >= threshold,
            1
          ).otherwise(0)
        )


    // ============================================================
    // 16. DISPLAY ANOMALIES
    // ============================================================

    println(
      "\n[16] DETECTED ANOMALIES"
    )

    val anomalyCount =
      anomalyResults
        .filter(
          col("is_anomaly") === 1
        )
        .count()

    println(
      s"Anomalies detected: $anomalyCount"
    )

    anomalyResults
      .filter(
        col("is_anomaly") === 1
      )
      .select(
        "day",
        "total_energy",
        "household_count",
        "avg_energy",
        "cluster",
        "distance_from_center",
        "is_anomaly"
      )
      .orderBy(
        desc("distance_from_center")
      )
      .show(
        20,
        false
      )


    // ============================================================
    // 17. SAVE ANOMALY RESULTS TO HDFS
    // ============================================================

    println(
      "\n[17] SAVING ANOMALY RESULTS TO HDFS"
    )

    anomalyResults
      .select(
        "day",
        "total_energy",
        "household_count",
        "avg_energy",
        "cluster",
        "distance_from_center",
        "is_anomaly"
      )
      .orderBy("day")
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(anomalyPath)

    println(
      "Anomaly results stored at:"
    )

    println(anomalyPath)


    // ============================================================
    // 18. HOUSE-LEVEL DATASET
    // ============================================================

    println("\n[18] CREATING HOUSE-LEVEL DATASET")

    println(
      "Loading raw household data from HDFS..."
    )

    val houseRaw = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .csv(rawInputPath)

    println(
      s"Raw house-level records: ${houseRaw.count()}"
    )

    val houseDaily = houseRaw
      .select(
        col("LCLid"),
        to_date(col("day")).alias("day"),
        col("energy_sum")
          .cast("double")
          .alias("energy")
      )
      .filter(
        col("LCLid").isNotNull &&
        col("day").isNotNull &&
        col("energy").isNotNull &&
        col("energy") >= 0
      )
      .dropDuplicates("LCLid", "day")

    println(
      s"Clean house-day records: ${houseDaily.count()}"
    )

    println(
      s"Unique houses: ${houseDaily.select("LCLid").distinct().count()}"
    )

    houseDaily
      .select(
        min("day").alias("start_date"),
        max("day").alias("end_date")
      )
      .show()

    println(
      "Saving house_daily dataset to HDFS..."
    )

    houseDaily
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(houseDailyPath)

    println(
      "House-level dataset stored at:"
    )

    println(houseDailyPath)


    // ============================================================
    // 19. HOUSE-LEVEL FEATURE ENGINEERING
    // ============================================================

    println(
      "\n[19] CREATING HOUSE-LEVEL ML FEATURES"
    )

    /*
     * Window is partitioned by house.
     *
     * lag_1_energy:
     * Previous observed consumption for the same house.
     *
     * rolling_7_energy:
     * Average consumption over the previous 7 observed
     * records for the same house.
     *
     * These features use only previous observations and
     * therefore do not use the current day's target value.
     */

    val houseWindow =
      Window
        .partitionBy("LCLid")
        .orderBy("day")

    val houseFeaturesBase =
      houseDaily
        .withColumn(
          "year",
          year(col("day"))
        )
        .withColumn(
          "month",
          month(col("day"))
        )
        .withColumn(
          "day_of_month",
          dayofmonth(col("day"))
        )
        .withColumn(
          "day_of_week",
          dayofweek(col("day"))
        )
        .withColumn(
          "is_weekend",
          when(
            dayofweek(col("day")).isin(1, 7),
            1
          ).otherwise(0)
        )
        .withColumn(
          "lag_1_energy",
          lag("energy", 1).over(houseWindow)
        )
        .withColumn(
          "rolling_7_energy",
          avg("energy")
            .over(
              houseWindow
                .rowsBetween(-7, -1)
            )
        )

    println(
      "House-level feature sample:"
    )

    houseFeaturesBase
      .select(
        "LCLid",
        "day",
        "energy",
        "month",
        "day_of_week",
        "is_weekend",
        "lag_1_energy",
        "rolling_7_energy"
      )
      .orderBy("LCLid", "day")
      .show(10, false)


    // ============================================================
    // 20. REMOVE RECORDS WITHOUT HISTORICAL FEATURES
    // ============================================================

    println(
      "\n[20] PREPARING HOUSE-LEVEL ML DATA"
    )

    val houseMLData =
      houseFeaturesBase
        .filter(
          col("lag_1_energy").isNotNull &&
          col("rolling_7_energy").isNotNull
        )

    println(
      s"House ML records: ${houseMLData.count()}"
    )


    // ============================================================
    // 21. HOUSE-LEVEL TIME-BASED TRAIN / TEST SPLIT
    // ============================================================

    println(
      "\n[21] HOUSE-LEVEL TIME-BASED TRAIN / TEST SPLIT"
    )

    val houseTrain =
      houseMLData
        .filter("year < 2014")

    val houseTest =
      houseMLData
        .filter("year == 2014")

    println(
      s"House training observations: ${houseTrain.count()}"
    )

    println(
      s"House testing observations : ${houseTest.count()}"
    )

    houseTest
      .select(
        min("day").alias("test_start"),
        max("day").alias("test_end")
      )
      .show()


        // ============================================================
    // 22. HOUSE-LEVEL FEATURE PREPARATION
    // ============================================================

    println(
      "\n[22] PREPARING HOUSE-LEVEL FEATURES"
    )

    /*
     * LCLid is kept as an identifier only.
     *
     * It is NOT used as a Random Forest feature because
     * there are thousands of unique houses. Using LCLid as
     * a categorical feature would require a very large
     * maxBins value and causes excessive memory usage.
     *
     * Each prediction still retains its LCLid, so predictions
     * remain available separately for every house.
     */


    // ============================================================
    // 23. HOUSE-LEVEL RANDOM FOREST FEATURES
    // ============================================================

    println(
      "\n[23] ASSEMBLING HOUSE-LEVEL FEATURES"
    )

    val houseAssembler =
      new VectorAssembler()
        .setInputCols(
          Array(
            "month",
            "day_of_month",
            "day_of_week",
            "is_weekend",
            "lag_1_energy",
            "rolling_7_energy"
          )
        )
        .setOutputCol("house_features")

    val houseTrainML =
      houseAssembler
        .transform(houseTrain)
        .select(
          "LCLid",
          "day",
          "energy",
          "house_features"
        )

    val houseTestML =
      houseAssembler
        .transform(houseTest)
        .select(
          "LCLid",
          "day",
          "energy",
          "house_features"
        )

    println(
      "House-level feature sample:"
    )

    houseTrainML
      .show(5, false)


    // ============================================================
    // 24. HOUSE-LEVEL RANDOM FOREST
    // ============================================================

    println(
      "\n[24] TRAINING HOUSE-LEVEL RANDOM FOREST"
    )

    val houseRF =
      new RandomForestRegressor()
        .setFeaturesCol("house_features")
        .setLabelCol("energy")
        .setNumTrees(50)
        .setSeed(42)

    val houseModel =
      houseRF.fit(houseTrainML)

    println(
      "House-level Random Forest training completed."
    )


    // ============================================================
    // 25. HOUSE-LEVEL PREDICTIONS
    // ============================================================

    println(
      "\n[25] GENERATING HOUSE-LEVEL PREDICTIONS"
    )

    val housePredictions =
      houseModel
        .transform(houseTestML)
        .select(
          col("LCLid"),
          col("day"),
          col("energy").alias("actual_energy"),
          col("prediction")
        )
        .orderBy(
          col("LCLid"),
          col("day")
        )

    println(
      "Sample house-level predictions:"
    )

    housePredictions
      .show(20, false)
    // ============================================================
    // 26. HOUSE-LEVEL MODEL EVALUATION
    // ============================================================

    println(
      "\n[26] HOUSE-LEVEL MODEL EVALUATION"
    )

    val houseRMSE =
      new RegressionEvaluator()
        .setLabelCol("actual_energy")
        .setPredictionCol("prediction")
        .setMetricName("rmse")
        .evaluate(housePredictions)

    val houseR2 =
      new RegressionEvaluator()
        .setLabelCol("actual_energy")
        .setPredictionCol("prediction")
        .setMetricName("r2")
        .evaluate(housePredictions)

    println(
      f"House RMSE = $houseRMSE%.6f"
    )

    println(
      f"House R2   = $houseR2%.6f"
    )


    // ============================================================
    // 27. HOUSE-LEVEL PREDICTION SUMMARY
    // ============================================================

    println(
      "\n[27] HOUSE-LEVEL PREDICTION SUMMARY"
    )

    println(
      "Number of houses in test predictions:"
    )

    println(
      housePredictions
        .select("LCLid")
        .distinct()
        .count()
    )

    println(
      "Number of house-day predictions:"
    )

    println(
      housePredictions.count()
    )

    housePredictions
      .agg(
        avg("actual_energy")
          .alias("actual_avg_energy"),
        avg("prediction")
          .alias("predicted_avg_energy")
      )
      .show()


    // ============================================================
    // 28. SAVE HOUSE-LEVEL PREDICTIONS TO HDFS
    // ============================================================

    println(
      "\n[28] SAVING HOUSE-LEVEL PREDICTIONS TO HDFS"
    )

    housePredictions
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(housePredictionPath)

    println(
      "House-level predictions stored at:"
    )

    println(housePredictionPath)


    // ============================================================
    // 29. HOUSE-LEVEL FEATURE IMPORTANCE
    // ============================================================

    println(
      "\n[29] HOUSE-LEVEL FEATURE IMPORTANCE"
    )

    val houseFeatureNames =
      Array(
        "month",
        "day_of_month",
        "day_of_week",
        "is_weekend",
        "lag_1_energy",
        "rolling_7_energy"
      )

    val houseImportances =
      houseModel.featureImportances

    houseFeatureNames.zipWithIndex.foreach {
      case (name, index) =>
        println(
          f"$name%-20s ${houseImportances(index)}%.6f"
        )
    }


    // ============================================================
    // 30. FINAL PIPELINE SUMMARY
    // ============================================================

    println(
      "\n=============================================="
    )

    println(
      " PIPELINE COMPLETED"
    )

    println(
      "=============================================="
    )

    println(
      s"Processed rows          : ${data.count()}"
    )

    println(
      s"Aggregate train rows    : ${trainData.count()}"
    )

    println(
      s"Aggregate test rows     : ${testData.count()}"
    )

    println(
      f"Aggregate RMSE          : $rmse%.6f"
    )

    println(
      f"Aggregate R2            : $r2%.6f"
    )

    println(
      s"Anomalies               : $anomalyCount"
    )

    println(
      f"Anomaly threshold       : $threshold%.6f"
    )

    println(
      s"House-day records       : ${houseDaily.count()}"
    )

    println(
      s"House ML train rows     : ${houseTrain.count()}"
    )

    println(
      s"House ML test rows      : ${houseTest.count()}"
    )

    println(
      s"Houses predicted       : ${
        housePredictions
          .select("LCLid")
          .distinct()
          .count()
      }"
    )

    println(
      s"House predictions      : ${housePredictions.count()}"
    )

    println(
      f"House RMSE             : $houseRMSE%.6f"
    )

    println(
      f"House R2               : $houseR2%.6f"
    )

    println(
      "=============================================="
    )


    spark.stop()
  }
}