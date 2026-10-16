import org.apache.spark.sql.functions._

val path = "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily"

val raw = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(path)

println("=== LONDON SMART METER: ML FEATURE PREPARATION ===")
println("Input rows: " + raw.count())

println("=== ACTUAL DATA SCHEMA ===")
raw.printSchema()

println("=== COLUMN NAMES ===")
raw.columns.foreach(println)

println("=== SAMPLE RECORDS ===")
raw.show(5, false)

val energyCandidates = raw.columns.filter { c =>
  val n = c.toLowerCase
  n.contains("energy") || n.contains("consumption") ||
  n.contains("kwh") || n.contains("power") || n.contains("mean")
}

println("=== POSSIBLE ENERGY/TARGET COLUMNS ===")
energyCandidates.foreach(println)

println("=== NUMERIC SUMMARY ===")
val numericColumns = raw.schema.fields
  .filter(f => Set("integer", "long", "double", "float", "short", "decimal")
    .contains(f.dataType.typeName.toLowerCase))
  .map(_.name)

if (numericColumns.nonEmpty) {
  raw.select(numericColumns.map(c => col("`" + c.replace("`", "``") + "`")): _*)
    .describe()
    .show(false)
}

System.exit(0)