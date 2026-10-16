val df = spark.read.parquet("hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary")

println("=== UK-REFIT DATA QUALITY CHECK ===")
println("Total rows: " + df.count())
println("Total columns: " + df.columns.length)

df.columns.foreach { c =>
  println(c + " null_count=" + df.filter(df(c).isNull).count())
}

System.exit(0)