val path = "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily"

val rdd = sc.textFile(path)

println("=== LONDON: RDD PROCESSING ===")
println("Total lines: " + rdd.count())

println("=== FIRST 5 LINES ===")
rdd.take(5).foreach(println)

val nonEmpty = rdd.filter(line => line.trim.nonEmpty)
println("Non-empty lines: " + nonEmpty.count())

val wordCounts = rdd
  .flatMap(line => line.split(","))
  .map(value => (value.trim, 1))
  .reduceByKey(_ + _)

println("=== TOP 10 FREQUENT FIELD VALUES ===")
wordCounts.takeOrdered(10)(Ordering[Int].reverse.on(_._2))
  .foreach(println)

System.exit(0)