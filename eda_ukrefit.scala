val df = spark.read.parquet("hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary")

println("=== UK-REFIT: EDA ===")
println("Total rows: " + df.count())
println("Total columns: " + df.columns.length)

println("=== SCHEMA ===")
df.printSchema()

println("=== NUMERIC SUMMARY ===")
df.describe().show(false)

println("=== SAMPLE RECORDS ===")
df.show(10, false)

System.exit(0)