val df = spark.read.parquet("hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary")

println("=== UK-REFIT DAILY SUMMARY: SCHEMA ===")
df.printSchema()

println("=== ROW COUNT === " + df.count())

println("=== FIRST 5 ROWS ===")
df.show(5, false)

System.exit(0)