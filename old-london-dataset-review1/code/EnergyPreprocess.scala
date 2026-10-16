import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object EnergyPreprocess {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Smart Meter Energy Preprocessing")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ----------------------------------------------------
    // 1. Input and output paths
    // ----------------------------------------------------

    val inputPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/daily_dataset.csv"

    val outputPath =
      "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/energy_daily"

    println("==============================================")
    println("     SMART METER ENERGY PREPROCESSING")
    println("==============================================")

    // ----------------------------------------------------
    // 2. Read daily energy dataset from HDFS
    // ----------------------------------------------------

    println("\n[1] Reading daily energy dataset from HDFS...")

    val rawDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .csv(inputPath)

    println("Dataset loaded successfully.")

    // ----------------------------------------------------
    // 3. Display schema
    // ----------------------------------------------------

    println("\n[2] ORIGINAL SCHEMA")

    rawDF.printSchema()

    // ----------------------------------------------------
    // 4. Count records
    // ----------------------------------------------------

    val originalCount = rawDF.count()

    println(s"\nOriginal records: $originalCount")

    // ----------------------------------------------------
    // 5. Select required columns
    // ----------------------------------------------------

    println("\n[3] Selecting required energy columns...")

    val df = rawDF
      .select(
        col("LCLid"),
        to_date(col("day")).alias("day"),
        col("energy_sum").cast("double").alias("energy_sum")
      )
      .filter(
        col("day").isNotNull &&
        col("energy_sum").isNotNull &&
        col("energy_sum") >= 0
      )

    // ----------------------------------------------------
    // 6. Remove duplicate household-day records
    // ----------------------------------------------------

    println("\n[4] Removing duplicate records...")

    val beforeDuplicates = df.count()

    val cleanDF = df.dropDuplicates("LCLid", "day")

    val afterDuplicates = cleanDF.count()

    println(s"Records before duplicate removal: $beforeDuplicates")
    println(s"Records after duplicate removal : $afterDuplicates")
    println(s"Duplicates removed              : ${beforeDuplicates - afterDuplicates}")

    // ----------------------------------------------------
    // 7. Aggregate daily energy consumption
    // ----------------------------------------------------

    println("\n[5] Calculating daily energy consumption...")

    val dailyEnergyDF = cleanDF
      .groupBy("day")
      .agg(
        sum("energy_sum").alias("total_energy"),
        countDistinct("LCLid").alias("household_count")
      )
      .withColumn(
        "avg_energy",
        col("total_energy") / col("household_count")
      )
      .orderBy("day")

    // ----------------------------------------------------
    // 8. Add calendar features
    // ----------------------------------------------------

    println("\n[6] Creating calendar features...")

    val featureDF = dailyEnergyDF
      .withColumn("year", year(col("day")))
      .withColumn("month", month(col("day")))
      .withColumn("day_of_month", dayofmonth(col("day")))
      .withColumn("day_of_week", dayofweek(col("day")))
      .withColumn(
        "is_weekend",
        when(
          dayofweek(col("day")).isin(1, 7),
          1
        ).otherwise(0)
      )

    // ----------------------------------------------------
    // 9. Display final dataset
    // ----------------------------------------------------

    println("\n[7] FINAL DATASET")

    featureDF.printSchema()

    println("\nSample processed records:")

    featureDF.show(10, truncate = false)

    // ----------------------------------------------------
    // 10. Final statistics
    // ----------------------------------------------------

    val finalCount = featureDF.count()

    println("\n==============================================")
    println("        PREPROCESSING SUMMARY")
    println("==============================================")

    println(s"Original records : $originalCount")
    println(s"Final daily rows : $finalCount")

    println("\nDaily energy statistics:")

    featureDF
      .select(
        "total_energy",
        "household_count",
        "avg_energy"
      )
      .describe()
      .show()

    // ----------------------------------------------------
    // 11. Write processed data to HDFS
    // ----------------------------------------------------

    println("\n[8] Writing processed dataset to HDFS...")

    featureDF
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(outputPath)

    println("\nProcessed dataset successfully stored at:")
    println(outputPath)

    println("\n==============================================")
    println("       PREPROCESSING COMPLETED")
    println("==============================================")

    spark.stop()
  }
}
