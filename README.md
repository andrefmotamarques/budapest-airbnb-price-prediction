# Budapest Airbnb Price Prediction (R)

Predicting the nightly price of Airbnb listings in Budapest from location, room type, availability and review activity, comparing linear regression with tree-based ensembles. The project follows the CRISP-DM structure (business understanding, data understanding, preparation, modelling, evaluation).

Team project for the *Forecasting Models* course, MSc in Data Science, ISCTE (2025/26), developed with Afonso Abreu and Afonso Claudino.

## Data

`listings_Budapest.csv`, a public Airbnb listings snapshot for Budapest in the Inside Airbnb format (neighbourhood, coordinates, room type, minimum nights, availability, reviews). Prices were converted from HUF to EUR.

**Data availability:** the original snapshot used in this project is no longer available to me, so it is not included in the repository. The code, the report and the results above are kept as a record of the work. To re-run the pipeline, download a current Budapest `listings.csv` from [Inside Airbnb](https://insideairbnb.com/get-the-data/), save it as `data/listings_Budapest.csv` and run the script. Because the listings change over time, the numbers will differ from those reported here.

## Method

- **Feature engineering:** `days_since_last_review` derived from the last review date, irrelevant identifiers and free text dropped.
- **Outliers:** prices outside 1.5 × IQR removed, which caps the target at roughly €150 per night.
- **Models** (10-fold cross-validation with `caret`):
  - Multiple linear regression
  - Bagging (100 trees)
  - Random Forest (100 trees, `mtry` tuned from 1 to 10)
  - Gradient Boosting (GBM, tuned depth and number of trees)

## Results

| Model (10-fold CV) | RMSE (€) | MAE (€) | R² |
|---|---|---|---|
| Linear regression | 25.22 | 19.51 | 0.13 |
| Bagging | 25.16 | 19.29 | 0.14 |
| GBM | 23.95 | 18.17 | 0.22 |
| Random Forest (mtry = 10) | **23.08** | **17.23** | **0.28** |

On the held-out test set the Random Forest reached RMSE €24.17, MAE €17.79 and R² 0.26.

The modest R² reflects the data more than the models: the listings have no size, number of bedrooms or bathrooms, or amenities, which are the main drivers of price.

## Repository structure

```
airbnb_price_models.R   # full pipeline: cleaning, EDA, models, evaluation
install_packages.R      # installs the R packages used
data/                   # place listings_Budapest.csv here (see Data availability)
report_en.pdf           # full written report (English)
report_pt.pdf           # original report (Portuguese)
```

## How to run

```r
source("install_packages.R")
source("airbnb_price_models.R")
```

The script moves to its own folder before reading `data/listings_Budapest.csv` (see *Data availability* above), so it works from RStudio, `Rscript` or `source()` wherever the repository was downloaded.

## Tech stack

R · caret · randomForest · gbm · ipred · tidyverse · ggplot2

## Team

André Marques, Afonso Abreu and Afonso Claudino. The full report with the team's methodology and discussion is in [`report_en.pdf`](report_en.pdf) (English) and [`report_pt.pdf`](report_pt.pdf) (original, Portuguese).
