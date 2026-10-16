import org.apache.spark.sql.functions._

val path = "hdfs://namenode:9000/data/raw/old-london-dataset-review1/processed/house_daily"

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(path)

println("=== LONDON SMART METER: DATAFRAME ANALYSIS ===")
println("Total records: " + df.count())

println("=== ACTUAL SCHEMA ===")
df.printSchema()

println("=== COLUMN NAMES ===")
df.columns.foreach(println)

println("=== SAMPLE RECORDS ===")
df.show(5, false)

val numericTypes = Set("integer", "long", "double", "float", "short", "decimal")
val numericColumns = df.schema.fields
  .filter(f => numericTypes.exists(t => f.dataType.typeName.toLowerCase.contains(t)))
  .map(_.name)

println("=== NUMERIC COLUMN SUMMARY ===")
if (numericColumns.nonEmpty) {
  df.select(numericColumns.map(c => col("`" + c.replace("`", "``") + "`")): _*)
    .describe()
    .show(false)
} else {
  println("No numeric columns detected. Check the printed schema.")
}

System.exit(0)