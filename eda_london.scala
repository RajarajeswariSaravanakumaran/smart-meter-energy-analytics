val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily")

println("=== LONDON SMART METER: EDA ===")
println("Total rows: " + df.count())
println("Total columns: " + df.columns.length)

println("=== NUMERIC SUMMARY ===")
df.describe().show(false)

println("=== SAMPLE RECORDS ===")
df.show(10, false)

System.exit(0)