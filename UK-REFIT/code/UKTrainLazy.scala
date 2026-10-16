
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame

//============================================================
// UK-REFIT: Spark-native chronological model comparison
// Spark 3.5.x / Scala 2.12
//============================================================

val inputPath ="hdfs://namenode:9000/data/processed/UKrefit/appliance_hourly_summary"

val outputRoot =
  "hdfs://namenode:9000/data/results/UKrefit"

println("\n[1/7] Loading hourly UK-REFIT data...")

val raw = spark.read.parquet(inputPath)
  .select(
    col("House").cast("double").as("House"),
    to_date(col("Date")).as("Date"),
    col("Hour").cast("double").as("Hour"),
    col("Appliance"),
    col("MeanW").cast("double").as("label")
  )
  .filter(
    col("House").isNotNull &&
    col("Date").isNotNull &&
    col("Hour").isNotNull &&
    col("Appliance").isNotNull &&
    col("label").isNotNull &&
    col("label") >= 0
  )

println(s"Hourly rows loaded: ${raw.count()}")

// Create a timestamp-like ordering for each house/appliance series.
// Do not fill missing hours with zero.
val w = Window
  .partitionBy("House", "Appliance")
  .orderBy(col("Date"), col("Hour"))

println("\n[2/7] Creating lag and time features...")

val data = raw
  .withColumn("PreviousPower", lag(col("label"), 1).over(w))
  .withColumn(
    "ApplianceId",
    regexp_extract(col("Appliance"), "(\\d+)", 1).cast("double")
  )
  .withColumn(
    "HourSin",
    sin(lit(2.0 * math.Pi) * col("Hour") / lit(24.0))
  )
  .withColumn(
    "HourCos",
    cos(lit(2.0 * math.Pi) * col("Hour") / lit(24.0))
  )
  .withColumn(
    "DayOfWeek",
    dayofweek(col("Date")).cast("double")
  )
  .withColumn(
    "DaySin",
    sin(lit(2.0 * math.Pi) * col("DayOfWeek") / lit(7.0))
  )
  .withColumn(
    "DayCos",
    cos(lit(2.0 * math.Pi) * col("DayOfWeek") / lit(7.0))
  )
  .filter(
    col("PreviousPower").isNotNull &&
    col("ApplianceId").isNotNull
  )
  .cache()

println(s"Rows after feature creation: ${data.count()}")

// Chronological split:
// Training:   before 2014-12-10
// Validation: 2014-12-10 through 2015-02-10
// Test:       2015-02-11 onward
// The test period is not used for model selection.

val train = data.filter(col("Date") < lit("2014-12-10"))
  .cache()

val validation = data.filter(
  col("Date") >= lit("2014-12-10") &&
  col("Date") < lit("2015-02-11")
).cache()

val test = data.filter(col("Date") >= lit("2015-02-11"))
  .cache()

println("\n[3/7] Chronological split sizes...")
println(s"Training rows:   ${train.count()}")
println(s"Validation rows: ${validation.count()}")
println(s"Test rows:       ${test.count()}")

if (train.isEmpty || validation.isEmpty || test.isEmpty) {
  throw new IllegalStateException(
    "A chronological split is empty. Check the input dates and split boundaries."
  )
}

val featureCols = Array(
  "House",
  "ApplianceId",
  "HourSin",
  "HourCos",
  "DaySin",
  "DayCos",
  "PreviousPower"
)

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")
  .setHandleInvalid("skip")

// Train and compare candidate models using validation RMSE.
// Each candidate gets its own fitted pipeline.

println("\n[4/7] Training and comparing Spark ML regressors...")

val candidates: Seq[(String, org.apache.spark.ml.Estimator[_])] = Seq(
  ("RandomForest", new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(50)
    .setMaxDepth(8)
    .setSeed(42)),

  ("GradientBoostedTrees", new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(30)
    .setMaxDepth(5)
    .setSeed(42)),

  ("DecisionTree", new DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxDepth(8)
    .setSeed(42)),

  ("LinearRegression", new LinearRegression()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(30)
    .setRegParam(0.1))
)

val rmseEvaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val maeEvaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mae")

case class CandidateResult(
  name: String,
  pipeline: Pipeline,
  model: org.apache.spark.ml.PipelineModel,
  validationMAE: Double,
  validationRMSE: Double
)

var results = Seq.empty[CandidateResult]

candidates.foreach { case (name, estimator) =>
  println(s"\nTraining $name ...")

  val pipeline = new Pipeline()
    .setStages(Array(assembler, estimator.asInstanceOf[org.apache.spark.ml.PipelineStage]))

  val fitted = pipeline.fit(train)

  val validationPredictions = fitted.transform(validation)
    .filter(
      col("prediction").isNotNull &&
      !isnan(col("prediction"))
    )
    .cache()

  val mae = maeEvaluator.evaluate(validationPredictions)
  val rmse = rmseEvaluator.evaluate(validationPredictions)

  println(f"$name validation MAE  = $mae%.4f W")
  println(f"$name validation RMSE = $rmse%.4f W")

  results = results :+ CandidateResult(
    name, pipeline, fitted, mae, rmse
  )

  validationPredictions.unpersist()
}

// Save validation leaderboard
val leaderboard = results.map { r =>
  (r.name, r.validationMAE, r.validationRMSE)
}.toDF("Model", "Validation_MAE_W", "Validation_RMSE_W")
  .orderBy(col("Validation_RMSE_W"))

leaderboard.write
  .mode("overwrite")
  .option("header", "true")
  .csv(outputRoot + "/metrics/spark_model_comparison")

println("\nValidation leaderboard:")
leaderboard.show(false)

// Select using validation RMSE only; do not select on test results.
val best = results.minBy(_.validationRMSE)

println(s"\nSelected model: ${best.name}")
println(f"Validation MAE: ${best.validationMAE}%.4f W")
println(f"Validation RMSE: ${best.validationRMSE}%.4f W")

println("\n[5/7] Evaluating selected model on the untouched test period...")

val testPredictions = best.model.transform(test)
  .filter(
    col("prediction").isNotNull &&
    !isnan(col("prediction"))
  )
  .cache()

val testMAE = maeEvaluator.evaluate(testPredictions)
val testRMSE = rmseEvaluator.evaluate(testPredictions)

// Persistence baseline: next prediction equals previous observed power.
val baselinePredictions = testPredictions
  .withColumn("prediction", col("PreviousPower"))

val baselineMAE = maeEvaluator.evaluate(baselinePredictions)
val baselineRMSE = rmseEvaluator.evaluate(baselinePredictions)

println(f"\n${best.name} TEST MAE:  $testMAE%.4f W")
println(f"${best.name} TEST RMSE: $testRMSE%.4f W")
println(f"Persistence TEST MAE:  $baselineMAE%.4f W")
println(f"Persistence TEST RMSE: $baselineRMSE%.4f W")

println("\n[6/7] Saving model, predictions and metrics...")

// Save the complete fitted pipeline, including feature assembly.
best.model.write.overwrite()
  .save(outputRoot + "/models/best_hourly_regression_model")

testPredictions
  .select(
    "House", "Date", "Hour", "Appliance",
    "label", "PreviousPower", "prediction"
  )
  .write.mode("overwrite")
  .parquet(outputRoot + "/forecasts/hourly_test_predictions")

val metrics = Seq(
  ("SelectedModel", best.name),
  ("ValidationMAE_W", best.validationMAE.toString),
  ("ValidationRMSE_W", best.validationRMSE.toString),
  ("TestMAE_W", testMAE.toString),
  ("TestRMSE_W", testRMSE.toString),
  ("PersistenceMAE_W", baselineMAE.toString),
  ("PersistenceRMSE_W", baselineRMSE.toString),
  ("TestRows", testPredictions.count().toString),
  ("MAE_Better_Than_Persistence", (testMAE < baselineMAE).toString),
  ("RMSE_Better_Than_Persistence", (testRMSE < baselineRMSE).toString)
).toDF("Metric", "Value")

metrics.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv(outputRoot + "/metrics/best_model_test_metrics")

println("\n[7/7] Finished.")
println(s"Leaderboard: $outputRoot/metrics/spark_model_comparison")
println(s"Model:       $outputRoot/models/best_hourly_regression_model")
println(s"Predictions: $outputRoot/forecasts/hourly_test_predictions")
println(s"Metrics:     $outputRoot/metrics/best_model_test_metrics")

testPredictions.unpersist()
baselinePredictions.unpersist()
train.unpersist()
validation.unpersist()
test.unpersist()
data.unpersist()
