---
title: Vancouver Land Value Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: false
short_description: A data analysis and regression ML project
---

# Vancouver Land Value Predictor

Predicts Vancouver property values by neighbourhood using property tax data and (histogram-based) gradient boosted regression.

---

## Features

- Uses 2020-2025 Vancouver property tax data.
- Visualizer using Matplotlib on neighbourhood zoning.
- Fast model training using Histogram Gradient Boosted Regression.
- Interactive web UI via [Gradio](https://gradio.app/).
- Predicts land values based on zoning type, legal classification, and neighbourhood.


## Project Overview

This project is an interactive predictor and visualizer based on the **Vancouver Property Tax** dataset. It uses robust cleaning measures to clean a 1.2 million dataset for neighbourhood categories (which weren't provided), and estimates the average price of property based on each neighbourhood. 

## Usage

**Try it out here on [HuggingFace](https://huggingface.co/spaces/nesteagle/Vancouver-Land-Value-Predictor):**

*Note: Accuracy may be limited due to possible dataset or data fitting inaccuracies.*

## Data

- Source data: [Vancouver Property Tax Dataset](https://opendata.vancouver.ca/explore/dataset/property-tax-report/information/) and [GeoNames Canadian Postal Codes](https://www.geonames.org/), hosted on my [HuggingFace dataset](https://huggingface.co/datasets/nesteagle/VanProperty).
- Please follow **[Vancouver Open Government License](https://opendata.vancouver.ca/pages/licence/)**.
- Data may contain inaccuracies.

## How It Works

- Combines Vancouver property tax records with Canadian postal code geolocation data to map properties to coordinates.

- Cleans and analyzes 1.2 million entries of land, strata, and various other property types.

- Since BCAssessment doesn't provide neighbourhood boundaries, this project:
   - Maps postal codes to latitude/longitude coordinates
   - Filters outliers using z-score analysis to remove erroneous coordinates
   - Generates neighbourhood boundaries using convex hull algorithms
   - Visualizes results on real Vancouver map imagery

- Contains feature engineering including property age, zoning classifications, legal types, and neighbourhood codes using one-hot encoding.

- Trains a Histogram Gradient Boosting Regressor on log-transformed land values for better handling of price distributions.

## Limitations

- Predictions do not use property size, which wasn't provided in the dataset.
- Neighbourhood codes were not given by BCAssessment and had to be categorized manually.
- Inaccurate entries in the dataset and possible entries that escaped cleaning.
