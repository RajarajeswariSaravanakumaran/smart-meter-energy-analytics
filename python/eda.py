from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ============================================================
# SMART METER ENERGY ANALYSIS - PYSPARK EDA
# ============================================================

spark = SparkSession.builder \
    .appName("Smart Meter Energy EDA") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("       SMART METER ENERGY - PYSPARK EDA")
print("=" * 60)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

input_path = "hdfs://namenode:9000/data/processed/energy_daily"
output_path = "hdfs://namenode:9000/data/eda/energy_daily"

# ------------------------------------------------------------
# 2. Read processed data
# ------------------------------------------------------------

print("\n[1] Reading processed dataset from HDFS...")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(input_path)

print("Dataset loaded successfully.")

# ------------------------------------------------------------
# 3. Schema
# ------------------------------------------------------------

print("\n[2] DATASET SCHEMA")
df.printSchema()

# ------------------------------------------------------------
# 4. Dataset dimensions
# ------------------------------------------------------------

row_count = df.count()
column_count = len(df.columns)

print("\n[3] DATASET DIMENSIONS")
print("Rows    :", row_count)
print("Columns :", column_count)

print("\nColumns:")
for column in df.columns:
    print("-", column)

# ------------------------------------------------------------
# 5. Sample records
# ------------------------------------------------------------

print("\n[4] SAMPLE RECORDS")
df.show(10, truncate=False)

# ------------------------------------------------------------
# 6. Missing-value analysis
# ------------------------------------------------------------

print("\n[5] MISSING VALUE ANALYSIS")

missing_values = df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

missing_values.show()

# ------------------------------------------------------------
# 7. Descriptive statistics
# ------------------------------------------------------------

print("\n[6] DESCRIPTIVE STATISTICS")

df.select(
    "total_energy",
    "household_count",
    "avg_energy"
).describe().show()

# ------------------------------------------------------------
# 8. Minimum / maximum energy
# ------------------------------------------------------------

print("\n[7] ENERGY RANGE")

df.select(
    min("total_energy").alias("minimum_energy"),
    max("total_energy").alias("maximum_energy"),
    avg("total_energy").alias("mean_energy"),
    stddev("total_energy").alias("std_energy")
).show()

# ------------------------------------------------------------
# 9. Monthly energy consumption
# ------------------------------------------------------------

print("\n[8] MONTHLY ENERGY CONSUMPTION")

monthly_df = df.groupBy(
    "year",
    "month"
).agg(
    sum("total_energy").alias("monthly_energy"),
    avg("avg_energy").alias("monthly_avg_energy"),
    avg("household_count").alias("avg_households")
).orderBy(
    "year",
    "month"
)

monthly_df.show(20, truncate=False)

# ------------------------------------------------------------
# 10. Weekday vs weekend
# ------------------------------------------------------------

print("\n[9] WEEKDAY VS WEEKEND ENERGY")

weekend_df = df.groupBy(
    "is_weekend"
).agg(
    avg("total_energy").alias("average_total_energy"),
    avg("avg_energy").alias("average_household_energy"),
    count("*").alias("number_of_days")
).orderBy("is_weekend")

weekend_df.show()

# ------------------------------------------------------------
# 11. Day-of-week analysis
# ------------------------------------------------------------

print("\n[10] DAY-OF-WEEK ENERGY ANALYSIS")

weekday_df = df.groupBy(
    "day_of_week"
).agg(
    avg("total_energy").alias("average_total_energy"),
    avg("avg_energy").alias("average_household_energy"),
    count("*").alias("number_of_days")
).orderBy("day_of_week")

weekday_df.show()

# ------------------------------------------------------------
# 12. Highest consumption days
# ------------------------------------------------------------

print("\n[11] TOP 10 HIGHEST ENERGY CONSUMPTION DAYS")

top_days = df.orderBy(
    col("total_energy").desc()
).select(
    "day",
    "total_energy",
    "household_count",
    "avg_energy"
).limit(10)

top_days.show(truncate=False)

# ------------------------------------------------------------
# 13. Lowest consumption days
# ------------------------------------------------------------

print("\n[12] TOP 10 LOWEST ENERGY CONSUMPTION DAYS")

low_days = df.orderBy(
    col("total_energy").asc()
).select(
    "day",
    "total_energy",
    "household_count",
    "avg_energy"
).limit(10)

low_days.show(truncate=False)

# ------------------------------------------------------------
# 14. Correlation analysis
# ------------------------------------------------------------

print("\n[13] CORRELATION ANALYSIS")

energy_household_corr = df.stat.corr(
    "total_energy",
    "household_count"
)

energy_average_corr = df.stat.corr(
    "total_energy",
    "avg_energy"
)

print(
    "Correlation: Total Energy vs Household Count =",
    energy_household_corr
)

print(
    "Correlation: Total Energy vs Average Household Energy =",
    energy_average_corr
)

# ------------------------------------------------------------
# 15. Create ML-ready dataset
# ------------------------------------------------------------

print("\n[14] CREATING ML-READY DATASET")

ml_df = df.select(
    "day",
    "total_energy",
    "household_count",
    "avg_energy",
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "is_weekend"
).orderBy("day")

ml_df.show(10, truncate=False)

# ------------------------------------------------------------
# 16. Write EDA dataset
# ------------------------------------------------------------

print("\n[15] WRITING EDA DATASET TO HDFS")

ml_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)

print("EDA dataset stored at:")
print(output_path)

# ------------------------------------------------------------
# 17. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              EDA COMPLETED")
print("=" * 60)

print("Rows analyzed       :", row_count)
print("Columns analyzed    :", column_count)
print("ML-ready rows       :", ml_df.count())
print("Output location     :", output_path)

print("=" * 60)

spark.stop()