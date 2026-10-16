import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation.RegressionEvaluator

val inputPath =
  "hdfs://namenode:9000/data/processed/UKrefit/appliance_hourly_summary"

val outputRoot =
  "hdfs://namenode:9000/data/results/UKrefit/forecast_improved"

println("[1/6] Loading hourly data...")

val raw = spark.read.parquet(inputPath)
  .select(
    col("House").cast("double").as("House"),
    to_date(col("Date")).as("Date"),
    col("Hour").cast("int").as("Hour"),
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

// Use an hourly timestamp as the ordering key.
val timed = raw
  .withColumn(
    "Timestamp",
    to_timestamp(
      concat_ws(
        " ",
        date_format(col("Date"), "yyyy-MM-dd"),
        lpad(col("Hour").cast("string"), 2, "0")
      ),
      "yyyy-MM-dd HH"
    )
  )
  .filter(col("Timestamp").isNotNull)
  .dropDuplicates(Seq("House", "Appliance", "Timestamp"))

println("[2/6] Creating chronological lag features...")

val w = Window
  .partitionBy("House", "Appliance")
  .orderBy(col("Timestamp"))

val lagged = timed
  .withColumn("Lag1", lag(col("label"), 1).over(w))
  .withColumn("Lag2", lag(col("label"), 2).over(w))
  .withColumn("Lag3", lag(col("label"), 3).over(w))
  .withColumn("Lag6", lag(col("label"), 6).over(w))
  .withColumn("Lag24", lag(col("label"), 24).over(w))
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
    col("Lag1").isNotNull &&
    col("Lag2").isNotNull &&
    col("Lag3").isNotNull &&
    col("Lag6").isNotNull &&
    col("Lag24").isNotNull &&
    col("ApplianceId").isNotNull
  )
  .cache()

println(s"Rows with all lag features: ${lagged.count()}")

// Chronological train/validation/test periods.
// The test period is held back until model comparison is complete.
val train = lagged
  .filter(col("Date") < lit("2014-12-10"))
  .cache()

val validation = lagged
  .filter(
    col("Date") >= lit("2014-12-10") &&
    col("Date") < lit("2015-02-11")
  )
  .cache()

val test = lagged
  .filter(col("Date") >= lit("2015-02-11"))
  .cache()

println("[3/6] Split sizes:")
println(s"Training:   ${train.count()}")
println(s"Validation: ${validation.count()}")
println(s"Test:       ${test.count()}")

if (train.isEmpty || validation.isEmpty || test.isEmpty) {
  throw new IllegalStateException("A time split is empty.")
}

val featureCols = Array(
  "House",
  "ApplianceId",
  "HourSin",
  "HourCos",
  "DaySin",
  "DayCos",
  "Lag1",
  "Lag2",
  "Lag3",
  "Lag6",
  "Lag24"
)

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")
  .setHandleInvalid("skip")

val maeEval = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mae")

val rmseEval = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val models: Seq[(String, org.apache.spark.ml.Estimator[_])] = Seq(
  ("GradientBoostedTrees", new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(40)
    .setMaxDepth(5)
    .setSeed(42)),

  ("RandomForest", new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(60)
    .setMaxDepth(8)
    .setSeed(42))
)

case class ModelResult(
  name: String,
  model: org.apache.spark.ml.PipelineModel,
  validationMAE: Double,
  validationRMSE: Double
)

println("[4/6] Comparing models on validation data...")

var results = Seq.empty[ModelResult]

models.foreach { case (name, estimator) =>
  println(s"Training $name")

  val pipeline = new Pipeline()
    .setStages(
      Array(
        assembler,
        estimator.asInstanceOf[org.apache.spark.ml.PipelineStage]
      )
    )

  val fitted = pipeline.fit(train)

  val pred = fitted.transform(validation)
    .filter(
      col("prediction").isNotNull &&
      !isnan(col("prediction"))
    )
    .cache()

  val mae = maeEval.evaluate(pred)
  val rmse = rmseEval.evaluate(pred)

  println(f"$name validation MAE = $mae%.4f W")
  println(f"$name validation RMSE = $rmse%.4f W")

  results = results :+ ModelResult(name, fitted, mae, rmse)
  pred.unpersist()
}

val leaderboard = results.map(r =>
  (r.name, r.validationMAE, r.validationRMSE)
).toDF("Model", "Validation_MAE_W", "Validation_RMSE_W")
  .orderBy(col("Validation_RMSE_W"))

leaderboard.write
  .mode("overwrite")
  .option("header", "true")
  .csv(outputRoot + "/metrics/validation_leaderboard")

leaderboard.show(false)

// Choose model using validation RMSE, not test performance.
val best = results.minBy(_.validationRMSE)

println(s"Selected on validation: ${best.name}")

println("[5/6] Evaluating once on the held-out test period...")

val predictions = best.model.transform(test)
  .filter(
    col("prediction").isNotNull &&
    !isnan(col("prediction"))
  )
  .cache()

val modelMAE = maeEval.evaluate(predictions)
val modelRMSE = rmseEval.evaluate(predictions)

// Persistence predicts using the immediately preceding observed sample.
val persistence = predictions
  .withColumn("prediction", col("Lag1"))

val baselineMAE = maeEval.evaluate(persistence)
val baselineRMSE = rmseEval.evaluate(persistence)

println(f"Model test MAE:        $modelMAE%.4f W")
println(f"Model test RMSE:       $modelRMSE%.4f W")
println(f"Persistence test MAE:  $baselineMAE%.4f W")
println(f"Persistence test RMSE: $baselineRMSE%.4f W")

best.model.write.overwrite()
  .save(outputRoot + "/models/best_lagged_model")

predictions.select(
  "House", "Date", "Hour", "Appliance",
  "label", "Lag1", "Lag2", "Lag3", "Lag6", "Lag24",
  "prediction"
).write.mode("overwrite")
  .parquet(outputRoot + "/forecasts/test_predictions")

val metrics = Seq(
  ("SelectedModel", best.name),
  ("TestMAE_W", modelMAE.toString),
  ("TestRMSE_W", modelRMSE.toString),
  ("PersistenceMAE_W", baselineMAE.toString),
  ("PersistenceRMSE_W", baselineRMSE.toString),
  ("MAE_Better_Than_Persistence", (modelMAE < baselineMAE).toString),
  ("RMSE_Better_Than_Persistence", (modelRMSE < baselineRMSE).toString),
  ("TestRows", predictions.count().toString)
).toDF("Metric", "Value")

metrics.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv(outputRoot + "/metrics/test_metrics")

println("[6/6] Finished.")
println(s"Leaderboard: $outputRoot/metrics/validation_leaderboard")
println(s"Model:       $outputRoot/models/best_lagged_model")
println(s"Predictions: $outputRoot/forecasts/test_predictions")
println(s"Metrics:     $outputRoot/metrics/test_metrics")

predictions.unpersist()
persistence.unpersist()
train.unpersist()
validation.unpersist()
test.unpersist()
lagged.unpersist()