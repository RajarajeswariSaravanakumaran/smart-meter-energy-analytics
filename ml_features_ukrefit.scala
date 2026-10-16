import org.apache.spark.sql.functions._

val path = "hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary"
val raw = spark.read.parquet(path)

println("=== UK-REFIT: FEATURE ENGINEERING ===")
println("Input rows: " + raw.count())
raw.printSchema()

val features = raw
  .withColumn("DayOfMonth", dayofmonth(col("Date")))
  .withColumn("Month", month(col("Date")))
  .withColumn("DayOfWeek", dayofweek(col("Date")))
  .withColumn("IsWeekend", when(dayofweek(col("Date")).isin(1, 7), 1).otherwise(0))
  .filter(
    col("EnergyKWh").isNotNull &&
    col("MeanW").isNotNull &&
    col("ReadingCount").isNotNull
  )
  .select(
    col("House"),
    col("Appliance"),
    col("Date"),
    col("DayOfMonth"),
    col("Month"),
    col("DayOfWeek"),
    col("IsWeekend"),
    col("ReadingCount"),
    col("MeanW"),
    col("EnergyKWh").alias("Label")
  )

val splitDate = features
  .selectExpr("percentile_approx(Date, 0.8) as split_date")
  .first()
  .getDate(0)

val train = features.filter(col("Date") <= lit(splitDate))
val test = features.filter(col("Date") > lit(splitDate))

println("=== FEATURE DATASET ===")
println("Feature rows: " + features.count())
println("Training rows: " + train.count())
println("Testing rows: " + test.count())
println("Chronological split date: " + splitDate)
println("=== FEATURE SAMPLE ===")
features.show(10, false)
println("=== FEATURE SUMMARY ===")
features.describe("ReadingCount", "MeanW", "Label").show(false)

System.exit(0)