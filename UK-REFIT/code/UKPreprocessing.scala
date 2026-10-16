
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window

// --------------------------------------------------
// 1. CONFIGURATION
// --------------------------------------------------

val inputPath =
  "hdfs://namenode:9000/data/raw/UKrefit-new/csv/"

val outputRoot =
  "hdfs://namenode:9000/data/processed/UKrefit"

val maxGapSeconds = 300L

val applianceCols = (1 to 9).map(i => s"Appliance$i").toSeq

// --------------------------------------------------
// 2. EXPLICIT INPUT SCHEMA
// --------------------------------------------------

val refitSchema = StructType(
  Seq(
    StructField("Time", TimestampType, nullable = true),
    StructField("Unix", LongType, nullable = true),
    StructField("Aggregate", LongType, nullable = true)
  ) ++ applianceCols.map(c =>
    StructField(c, LongType, nullable = true)
  )
)

// --------------------------------------------------
// 3. READ AND STANDARDIZE SOURCE DATA
// --------------------------------------------------

val source = spark.read
  .option("header", "true")
  .option("mode", "PERMISSIVE")
  .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
  .option("pathGlobFilter", "House_*.csv")
  .schema(refitSchema)
  .csv(inputPath)
  .withColumn(
    "House",
    regexp_extract(
      input_file_name(),
      ".*/House_(\\d+)\\.csv",
      1
    ).cast("int")
  )
  .filter(
    col("House").isNotNull &&
    col("Time").isNotNull &&
    col("Unix").isNotNull
  )
  .withColumn("Date", to_date(col("Time")))
  .withColumn("Hour", hour(col("Time")))

val cleaned = source.select(
  (
    Seq(
      col("House"),
      col("Time"),
      col("Unix"),
      col("Date"),
      col("Hour"),
      col("Aggregate")
    ) ++ applianceCols.map(c => col(c))
  ): _*
)

println("Input schema:")
cleaned.printSchema()

// --------------------------------------------------
// 4. SAVE CLEANED READINGS
// --------------------------------------------------

cleaned.write
  .mode("overwrite")
  .parquet(s"$outputRoot/cleaned_readings")

println("Cleaned readings saved.")

// --------------------------------------------------
// 5. ESTIMATE ENERGY CONSUMPTION
// --------------------------------------------------

// Calculate differences within each house and day.
// The first reading each day has no previous reading
// in that day, so cross-midnight intervals are omitted.

val houseDayWindow = Window
  .partitionBy("House", "Date")
  .orderBy(col("Unix"))

val withPrevious = applianceCols.foldLeft(
  cleaned
    .withColumn(
      "PreviousUnix",
      lag(col("Unix"), 1).over(houseDayWindow)
    )
    .withColumn(
      "IntervalSeconds",
      col("Unix") - col("PreviousUnix")
    )
) { (df, appliance) =>
  df.withColumn(
    s"Previous_$appliance",
    lag(col(appliance), 1).over(houseDayWindow)
  )
}

// Power readings are in watts.
// Energy (kWh) = average power (W) * seconds / 3,600,000.

val energyColumns = applianceCols.map { appliance =>
  val previousPower =
    col(s"Previous_$appliance").cast("double")

  val currentPower =
    col(appliance).cast("double")

  val validInterval =
    col("PreviousUnix").isNotNull &&
    col("IntervalSeconds") > 0 &&
    col("IntervalSeconds") <= lit(maxGapSeconds) &&
    previousPower.isNotNull &&
    currentPower.isNotNull

  when(
    validInterval,
    ((previousPower + currentPower) / lit(2.0)) *
      col("IntervalSeconds").cast("double") /
      lit(3600000.0)
  ).otherwise(lit(0.0))
    .as(s"${appliance}_EnergyKWh")
}

val energyReadings = withPrevious.select(
  (
    Seq(
      col("House"),
      col("Date"),
      col("Hour"),
      col("Unix")
    ) ++ energyColumns
  ): _*
)

// --------------------------------------------------
// 6. DAILY ENERGY
// --------------------------------------------------

val dailyEnergyExpressions = applianceCols.map { appliance =>
  sum(col(s"${appliance}_EnergyKWh"))
    .as(s"${appliance}_EnergyKWh")
}

val dailyEnergy = energyReadings
  .groupBy("House", "Date")
  .agg(
    dailyEnergyExpressions.head,
    dailyEnergyExpressions.tail: _*
  )

// --------------------------------------------------
// 7. HOURLY ENERGY
// --------------------------------------------------

// Energy is assigned to the hour of the later reading.

val hourlyEnergyExpressions = applianceCols.map { appliance =>
  sum(col(s"${appliance}_EnergyKWh"))
    .as(s"${appliance}_EnergyKWh")
}

val hourlyEnergy = energyReadings
  .groupBy("House", "Date", "Hour")
  .agg(
    hourlyEnergyExpressions.head,
    hourlyEnergyExpressions.tail: _*
  )

// --------------------------------------------------
// 8. DAILY POWER STATISTICS
// --------------------------------------------------

val dailyPowerExpressions = applianceCols.flatMap { appliance =>
  Seq(
    avg(col(appliance).cast("double"))
      .as(s"${appliance}_MeanW"),

    min(col(appliance))
      .as(s"${appliance}_MinW"),

    max(col(appliance))
      .as(s"${appliance}_MaxW"),

    stddev_pop(col(appliance).cast("double"))
      .as(s"${appliance}_StdDevW"),

    avg(
      when(col(appliance) > 0, 1.0)
        .otherwise(0.0)
    ).as(s"${appliance}_ActiveRatio")
  )
}

val dailyAggregateExpressions =
  Seq(count(lit(1)).as("ReadingCount")) ++
    dailyPowerExpressions

val dailyPower = cleaned
  .groupBy("House", "Date")
  .agg(
    dailyAggregateExpressions.head,
    dailyAggregateExpressions.tail: _*
  )

val dailyWide = dailyPower.join(
  dailyEnergy,
  Seq("House", "Date"),
  "left"
)

// Convert the aggregated daily table to one row per appliance.

val dailyStackExpressions = applianceCols.map { appliance =>
  s"'$appliance', " +
    s"`${appliance}_MeanW`, " +
    s"`${appliance}_MinW`, " +
    s"`${appliance}_MaxW`, " +
    s"`${appliance}_StdDevW`, " +
    s"`${appliance}_ActiveRatio`, " +
    s"`${appliance}_EnergyKWh`"
}.mkString(", ")

val dailySummary = dailyWide.selectExpr(
  "House",
  "Date",
  "ReadingCount",
  s"""stack(9, $dailyStackExpressions) AS
      (Appliance, MeanW, MinW, MaxW, StdDevW,
       ActiveRatio, EnergyKWh)"""
)

dailySummary.write
  .mode("overwrite")
  .parquet(s"$outputRoot/appliance_daily_summary")

println("Daily appliance summaries saved.")

// --------------------------------------------------
// 9. HOURLY POWER STATISTICS
// --------------------------------------------------

val hourlyPowerExpressions = applianceCols.flatMap { appliance =>
  Seq(
    avg(col(appliance).cast("double"))
      .as(s"${appliance}_MeanW"),

    min(col(appliance))
      .as(s"${appliance}_MinW"),

    max(col(appliance))
      .as(s"${appliance}_MaxW"),

    avg(
      when(col(appliance) > 0, 1.0)
        .otherwise(0.0)
    ).as(s"${appliance}_ActiveRatio")
  )
}

val hourlyAggregateExpressions =
  Seq(count(lit(1)).as("ReadingCount")) ++
    hourlyPowerExpressions

val hourlyPower = cleaned
  .groupBy("House", "Date", "Hour")
  .agg(
    hourlyAggregateExpressions.head,
    hourlyAggregateExpressions.tail: _*
  )

val hourlyWide = hourlyPower.join(
  hourlyEnergy,
  Seq("House", "Date", "Hour"),
  "left"
)

val hourlyStackExpressions = applianceCols.map { appliance =>
  s"'$appliance', " +
    s"`${appliance}_MeanW`, " +
    s"`${appliance}_MinW`, " +
    s"`${appliance}_MaxW`, " +
    s"`${appliance}_ActiveRatio`, " +
    s"`${appliance}_EnergyKWh`"
}.mkString(", ")

val hourlySummary = hourlyWide.selectExpr(
  "House",
  "Date",
  "Hour",
  "ReadingCount",
  s"""stack(9, $hourlyStackExpressions) AS
      (Appliance, MeanW, MinW, MaxW,
       ActiveRatio, EnergyKWh)"""
)

hourlySummary.write
  .mode("overwrite")
  .parquet(s"$outputRoot/appliance_hourly_summary")

println("Hourly appliance summaries saved.")

// --------------------------------------------------
// 10. COMPLETION
// --------------------------------------------------

println("====================================")
println("UK-REFIT PREPROCESSING COMPLETED")
println("====================================")
println(s"Cleaned readings: $outputRoot/cleaned_readings")
println(s"Daily summaries:  $outputRoot/appliance_daily_summary")
println(s"Hourly summaries: $outputRoot/appliance_hourly_summary")
println("Energy units: kWh")
println(s"Maximum included interval: $maxGapSeconds seconds")
println("Energy estimates exclude cross-midnight intervals.")

