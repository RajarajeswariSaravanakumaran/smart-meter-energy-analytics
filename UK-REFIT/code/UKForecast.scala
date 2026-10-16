import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val base = "hdfs://namenode:9000/data/processed/UKrefit"
val out = "hdfs://namenode:9000/data/results/UKrefit"

val hourly = spark.read.parquet(base + "/appliance_hourly_summary")

val prepared = hourly
  .withColumn("Timestamp", to_timestamp(concat_ws(" ", col("Date").cast("string"), format_string("%02d:00:00", col("Hour")))))
  .withColumn("ApplianceId", regexp_extract(col("Appliance"), "([0-9]+)$", 1).cast("double"))
  .withColumn("DayOfWeek", dayofweek(col("Date")).cast("double"))
  .withColumn("HourFeature", col("Hour").cast("double"))
  .withColumn("HouseFeature", col("House").cast("double"))

val w = Window.partitionBy("House", "Appliance").orderBy("Timestamp")

val featured = prepared
  .withColumn("PreviousPower", lag(col("MeanW"), 1).over(w))
  .withColumn("PreviousObservedPower", lag(col("MeanW"), 1).over(w))
  .withColumn("Previous24Observations", lag(col("MeanW"), 24).over(w))
  .filter(col("Timestamp").isNotNull && col("MeanW").isNotNull && col("PreviousPower").isNotNull)
  .filter(col("MeanW") >= 0 && col("PreviousPower") >= 0)

val cutoff = "2015-05-01 00:00:00"

val train = featured.filter(col("Timestamp") < to_timestamp(lit(cutoff)))
val test = featured.filter(col("Timestamp") >= to_timestamp(lit(cutoff)))

println("Training rows: " + train.count())
println("Test rows: " + test.count())

val assembler = new VectorAssembler()
  .setInputCols(Array("HouseFeature", "ApplianceId", "HourFeature", "DayOfWeek", "PreviousPower"))
  .setOutputCol("features")
  .setHandleInvalid("skip")

val trainVec = assembler.transform(train).select(col("features"), col("MeanW").alias("label"))
val testVec = assembler.transform(test)

val rf = new RandomForestRegressor()
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setNumTrees(30)
  .setMaxDepth(8)
  .setSeed(42)

val model = rf.fit(trainVec)
val scored = model.transform(testVec)
  .withColumn("BaselinePrediction", col("PreviousPower"))
  .withColumn("AbsoluteError", abs(col("MeanW") - col("prediction")))
  .withColumn("BaselineAbsoluteError", abs(col("MeanW") - col("BaselinePrediction")))

val rmseEval = new RegressionEvaluator().setLabelCol("MeanW").setPredictionCol("prediction").setMetricName("rmse")
val maeEval = new RegressionEvaluator().setLabelCol("MeanW").setPredictionCol("prediction").setMetricName("mae")
val baselineRmseEval = new RegressionEvaluator().setLabelCol("MeanW").setPredictionCol("BaselinePrediction").setMetricName("rmse")
val baselineMaeEval = new RegressionEvaluator().setLabelCol("MeanW").setPredictionCol("BaselinePrediction").setMetricName("mae")

val metrics = Seq(
  ("RandomForest", "RMSE_W", rmseEval.evaluate(scored)),
  ("RandomForest", "MAE_W", maeEval.evaluate(scored)),
  ("PersistenceBaseline", "RMSE_W", baselineRmseEval.evaluate(scored)),
  ("PersistenceBaseline", "MAE_W", baselineMaeEval.evaluate(scored))
).toDF("Model", "Metric", "Value")

model.write.overwrite().save(out + "/models/hourly_random_forest")
scored.select("House", "Appliance", "Date", "Hour", "Timestamp", "MeanW", "prediction", "BaselinePrediction", "AbsoluteError")
  .write.mode("overwrite").parquet(out + "/forecasts/hourly_test_predictions")
metrics.write.mode("overwrite").option("header", "true").csv(out + "/metrics/hourly_forecast_metrics")

println("FORECAST PIPELINE COMPLETE")
metrics.show(false)
