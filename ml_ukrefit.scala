import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator

val path = "hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary"

val raw = spark.read.parquet(path)

val data = raw
  .withColumn("HouseFeature", col("House").cast("string"))
  .withColumn("ApplianceFeature", col("Appliance").cast("string"))
  .withColumn("MonthFeature", month(col("Date")).cast("double"))
  .withColumn("DayFeature", dayofmonth(col("Date")).cast("double"))
  .withColumn("WeekdayFeature", dayofweek(col("Date")).cast("double"))
  .filter(
    col("EnergyKWh").isNotNull &&
    col("HouseFeature").isNotNull &&
    col("ApplianceFeature").isNotNull &&
    col("Date").isNotNull
  )
  .select(
    col("HouseFeature"),
    col("ApplianceFeature"),
    col("MonthFeature"),
    col("DayFeature"),
    col("WeekdayFeature"),
    col("Date"),
    col("EnergyKWh").cast("double").alias("label")
  )
  .filter(col("label") >= 0)

val splitDate = data
  .selectExpr("percentile_approx(Date, 0.8) as split_date")
  .first().getDate(0)

val train = data.filter(col("Date") <= lit(splitDate)).drop("Date")
val test = data.filter(col("Date") > lit(splitDate)).drop("Date")

println("=== UK-REFIT SPARK ML PIPELINE ===")
println("Training rows: " + train.count())
println("Testing rows: " + test.count())
println("Chronological split date: " + splitDate)

val houseIndexer = new StringIndexer()
  .setInputCol("HouseFeature")
  .setOutputCol("HouseIndex")
  .setHandleInvalid("keep")

val applianceIndexer = new StringIndexer()
  .setInputCol("ApplianceFeature")
  .setOutputCol("ApplianceIndex")
  .setHandleInvalid("keep")

val assembler = new VectorAssembler()
  .setInputCols(Array(
    "HouseIndex",
    "ApplianceIndex",
    "MonthFeature",
    "DayFeature",
    "WeekdayFeature"
  ))
  .setOutputCol("features")

val evaluatorRMSE = new RegressionEvaluator()
  .setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val evaluatorMAE = new RegressionEvaluator()
  .setLabelCol("label").setPredictionCol("prediction").setMetricName("mae")

val evaluatorR2 = new RegressionEvaluator()
  .setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")

def evaluateModel(name: String, estimator: org.apache.spark.ml.Estimator[_]): Unit = {
  val pipeline = new Pipeline().setStages(Array(
    houseIndexer, applianceIndexer, assembler, estimator
  ))
  val model = pipeline.fit(train)
  val predictions = model.transform(test).cache()

  println("=== " + name + " ===")
  println("RMSE: " + evaluatorRMSE.evaluate(predictions))
  println("MAE:  " + evaluatorMAE.evaluate(predictions))
  println("R2:   " + evaluatorR2.evaluate(predictions))
  println("=== SAMPLE ACTUAL VS PREDICTED ===")
  predictions.select("label", "prediction").show(10, false)

  predictions.unpersist()
}

evaluateModel(
  "LINEAR REGRESSION",
  new LinearRegression().setMaxIter(30).setRegParam(0.1)
)

evaluateModel(
  "RANDOM FOREST REGRESSION",
  new RandomForestRegressor()
    .setNumTrees(20)
    .setMaxDepth(5)
    .setSeed(42)
)

System.exit(0)