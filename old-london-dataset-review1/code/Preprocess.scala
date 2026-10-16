import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Preprocess {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ACORN Dataset Preprocessing")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ----------------------------------------------------
    // 1. Input and output paths
    // ----------------------------------------------------

    val inputPath = "hdfs://namenode:9000/data/raw/acorn_details.csv"
    val outputPath = "hdfs://namenode:9000/data/processed/acorn_details"

    println("==============================================")
    println("       ACORN DATASET PREPROCESSING")
    println("==============================================")

    // ----------------------------------------------------
    // 2. Read raw dataset from HDFS
    // ----------------------------------------------------

    println("\n[1] Reading raw dataset from HDFS...")

    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .csv(inputPath)

    println("Raw dataset loaded successfully.")

    // ----------------------------------------------------
    // 3. Display original schema
    // ----------------------------------------------------

    println("\n[2] ORIGINAL SCHEMA")
    df.printSchema()

    // ----------------------------------------------------
    // 4. Count original records
    // ----------------------------------------------------

    val originalCount = df.count()

    println(s"\nOriginal record count: $originalCount")

    // ----------------------------------------------------
    // 5. Trim categorical columns
    // ----------------------------------------------------

    println("\n[3] Trimming categorical columns...")

    df = df
      .withColumn(
        "MAIN CATEGORIES",
        trim(col("MAIN CATEGORIES"))
      )
      .withColumn(
        "CATEGORIES",
        trim(col("CATEGORIES"))
      )
      .withColumn(
        "REFERENCE",
        trim(col("REFERENCE"))
      )

    // ----------------------------------------------------
    // 6. Remove duplicate records
    // ----------------------------------------------------

    println("\n[4] Removing duplicate records...")

    val beforeDuplicates = df.count()

    df = df.dropDuplicates()

    val afterDuplicates = df.count()

    println(s"Records before duplicate removal: $beforeDuplicates")
    println(s"Records after duplicate removal : $afterDuplicates")
    println(s"Duplicates removed              : ${beforeDuplicates - afterDuplicates}")

    // ----------------------------------------------------
    // 7. ACORN numeric columns
    // ----------------------------------------------------

    val acornColumns = (('A' to 'Q')).map(letter => s"ACORN-$letter")

    // ----------------------------------------------------
    // 8. Cast ACORN columns to double
    // ----------------------------------------------------

    println("\n[5] Converting ACORN columns to numeric type...")

    acornColumns.foreach { columnName =>
      df = df.withColumn(
        columnName,
        col(columnName).cast("double")
      )
    }

    // ----------------------------------------------------
    // 9. Remove rows where all ACORN values are null
    // ----------------------------------------------------

    println("\n[6] Removing records with no ACORN values...")

    val allNullCondition =
      acornColumns
        .map(columnName => col(columnName).isNull)
        .reduce(_ && _)

    val beforeNullRemoval = df.count()

    df = df.filter(!allNullCondition)

    val afterNullRemoval = df.count()

    println(s"Records before null filtering: $beforeNullRemoval")
    println(s"Records after null filtering : $afterNullRemoval")
    println(s"Records removed              : ${beforeNullRemoval - afterNullRemoval}")

    // ----------------------------------------------------
    // 10. Replace remaining numeric null values with 0
    // ----------------------------------------------------

    println("\n[7] Handling remaining missing numeric values...")

    df = df.na.fill(
      0.0,
      acornColumns
    )

    // ----------------------------------------------------
    // 11. Remove negative numeric values
    // ----------------------------------------------------

    println("\n[8] Filtering invalid negative ACORN values...")

    val nonNegativeCondition =
      acornColumns
        .map(columnName => col(columnName) >= 0)
        .reduce(_ && _)

    val beforeNegativeFilter = df.count()

    df = df.filter(nonNegativeCondition)

    val afterNegativeFilter = df.count()

    println(s"Records before validation: $beforeNegativeFilter")
    println(s"Records after validation : $afterNegativeFilter")
    println(s"Invalid records removed  : ${beforeNegativeFilter - afterNegativeFilter}")

    // ----------------------------------------------------
    // 12. Feature engineering
    // ----------------------------------------------------

    println("\n[9] Creating derived features...")

    val totalExpression =
      acornColumns
        .map(columnName => col(columnName))
        .reduce(_ + _)

    val averageExpression =
      totalExpression / lit(acornColumns.length)

    df = df
      .withColumn(
        "TOTAL_ACORN",
        totalExpression
      )
      .withColumn(
        "AVERAGE_ACORN",
        round(averageExpression, 2)
      )

    // ----------------------------------------------------
    // 13. Final record count
    // ----------------------------------------------------

    val finalCount = df.count()

    println("\n==============================================")
    println("           PREPROCESSING SUMMARY")
    println("==============================================")

    println(s"Original records : $originalCount")
    println(s"Final records    : $finalCount")
    println(s"Records removed  : ${originalCount - finalCount}")

    // ----------------------------------------------------
    // 14. Display final schema
    // ----------------------------------------------------

    println("\n[10] FINAL SCHEMA")
    df.printSchema()

    // ----------------------------------------------------
    // 15. Display sample processed records
    // ----------------------------------------------------

    println("\n[11] SAMPLE PROCESSED RECORDS")

    df.show(10, truncate = false)

    // ----------------------------------------------------
    // 16. Write processed dataset to HDFS
    // ----------------------------------------------------

    println("\n[12] Writing processed dataset to HDFS...")

    df.write
      .mode("overwrite")
      .option("header", "true")
      .csv(outputPath)

    println(s"\nProcessed dataset successfully stored at:")
    println(outputPath)

    println("\n==============================================")
    println("       PREPROCESSING COMPLETED")
    println("==============================================")

    spark.stop()
  }
}