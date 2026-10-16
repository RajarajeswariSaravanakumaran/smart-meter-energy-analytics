def inspectDataset(name: String, path: String, format: String): Unit = {
  println("\n========================================")
  println("DATASET: " + name)
  println("========================================")

  val start = System.nanoTime()

  val df = if (format == "csv") {
    spark.read.option("header", "true")
      .option("inferSchema", "true").csv(path)
  } else {
    spark.read.parquet(path)
  }

  println("=== SCHEMA ===")
  df.printSchema()

  println("=== ROW COUNT === " + df.count())
  println("=== FIRST 5 ROWS ===")
  df.show(5, false)

  println("=== NUMERIC/DATA SUMMARY ===")
  df.describe().show(false)

  println("=== ELAPSED SECONDS === " +
    (System.nanoTime() - start) / 1e9)
}

inspectDataset(
  "LONDON SMART METER",
  "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily",
  "csv"
)

inspectDataset(
  "UK-REFIT",
  "hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary",
  "parquet"
)

System.exit(0)