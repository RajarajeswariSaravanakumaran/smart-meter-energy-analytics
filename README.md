# Smart Meter Energy Analytics Using Big Data

An Apache Spark and Scala project for analyzing household electricity
consumption, forecasting energy usage, and identifying unusual patterns
using London Smart Meter and UK-REFIT data.

## Technology Stack

- Apache Hadoop HDFS
- Apache Spark 3.5.6
- Scala 2.12
- Java 17
- Python and Streamlit
- Docker

## Project Pipelines

### London Smart Meter

- `old-london-dataset-review1/code/EnergyPreprocess.scala`
  - Cleans daily household consumption records.
  - Aggregates daily energy consumption.
  - Generates calendar features.
- `old-london-dataset-review1/code/EnergyMLPipeline.scala`
  - Runs aggregate energy prediction.
  - Performs anomaly detection.
  - Creates household-level data and predictions.

### UK-REFIT

- `UK-REFIT/code/UKPreprocessing.scala`
  - Processes household readings.
  - Generates cleaned readings and daily/hourly appliance summaries.
- `UK-REFIT/code/UKForecastImproved.scala`
  - Compares forecasting models.
  - Evaluates the selected model against a persistence baseline.
  - Stores predictions and evaluation metrics.

## Data Locations

London input in HDFS:

`/data/raw/old-london-dataset-review1/daily_dataset.csv`

UK-REFIT input in HDFS:

`/data/raw/UKrefit-new/csv/`

Processed datasets and model outputs are stored in HDFS.
Raw datasets and generated outputs are intentionally excluded from
this repository.

## Dashboard

The combined Streamlit dashboard is located at:

`combined-dashboard/app.py`

Launch it from the project environment with:

`streamlit run "combined-dashboard/app.py"`

The dashboard expects the configured HDFS outputs and/or local UK-REFIT
CSV exports to be available.

## Execution Notes

The London Scala applications define `main` methods and can be compiled
and submitted with `spark-submit`.

The UK-REFIT Scala files contain top-level statements intended for
interactive Spark Shell execution.

Use the configured Docker, Hadoop, and Spark environment and verify each
pipeline's outputs before running dependent stages.

## Data and Reproducibility

This repository contains source code and documentation, not the raw
research datasets or generated results. Obtain the datasets separately
and configure their HDFS paths before execution.

Model metrics and dashboard results depend on the dataset version,
pipeline execution, and available output files.
