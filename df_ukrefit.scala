import org.apache.spark.sql.functions._

val path = "hdfs://namenode:9000/data/processed/UKrefit/appliance_daily_summary"
val df = spark.read.parquet(path)

println("=== UK-REFIT: DATAFRAME ENERGY ANALYSIS ===")
println("Total records: " + df.count())

println("=== SCHEMA ===")
df.printSchema()

println("=== SAMPLE RECORDS ===")
df.show(5, false)

println("=== ENERGY SUMMARY BY APPLIANCE ===")
df.groupBy("Appliance")
  .agg(
    count("*").alias("RecordCount"),
    round(avg("EnergyKWh"), 4).alias("AverageEnergyKWh"),
    round(sum("EnergyKWh"), 4).alias("TotalEnergyKWh"),
    round(max("EnergyKWh"), 4).alias("MaximumEnergyKWh")
  )
  .orderBy(desc("TotalEnergyKWh"))
  .show(30, false)

System.exit(0)