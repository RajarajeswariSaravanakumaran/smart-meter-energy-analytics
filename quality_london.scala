val df = spark.read.option("header", "true").option("inferSchema", "true").csv("hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily")

println("=== LONDON DATA QUALITY CHECK ===")
println("Total rows: " + df.count())
println("Total columns: " + df.columns.length)

df.columns.foreach { c =>
  println(c + " null_count=" + df.filter(df(c).isNull).count())
}

System.exit(0)