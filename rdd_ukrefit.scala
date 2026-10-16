val path = "hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary"

val df = spark.read.parquet(path)
val rdd = df.rdd

println("=== UK-REFIT: RDD PROCESSING ===")
println("Total records: " + rdd.count())

println("=== FIRST 5 RECORDS ===")
rdd.take(5).foreach(println)

val nonEmpty = rdd.filter(row => !row.anyNull)
println("Records with no null fields: " + nonEmpty.count())

println("=== RDD TRANSFORMATION: MAP ===")
rdd.map(row => row.toString).take(5).foreach(println)

System.exit(0)