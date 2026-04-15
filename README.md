# Wake County Housing Price Predictor

ML regression model that predicts residential home prices in Wake County, NC using Random Forest. Includes a future price estimator with annual appreciation compounding.

---

## Results

| Metric | Value |
|---|---|
| R² | 0.95 |
| RMSE | ~$18,000 |
| MAE | — |
| MAPE | — |

---

## Features Used

| Feature | Description |
|---|---|
| `Heated_Area` | Interior square footage |
| `Bath` | Number of bathrooms |
| `Deeded_Acreage` | Lot size in acres |
| `Lot_SqFt` | Lot size in sq ft (derived) |
| `Year_Built` | Year of construction |
| `Large_Lot` | Binary flag for lots > 1 acre |
| `Zip_Avg_Price_per_sqft` | ZIP-level avg $/sqft from training data |
| `Physical_City` | City (one-hot encoded) |
| `Physical_Zip` | ZIP code (one-hot encoded, top 25 only) |

---

## Model

- **Algorithm**: Random Forest Regressor (200 trees, max depth 20)
- **Target**: `log1p(Total_Sale_Price)` — log-transformed to reduce skew
- **Preprocessing**: `StandardScaler` on numerics, `OneHotEncoder` on categoricals via `ColumnTransformer`
- **Split**: 80/20 train/test

---

## Data

- Source: Wake County public housing records (`WakeCountyHousing.csv`)
- Filters applied: built after 2005, sale price $100K–$2.5M, top 25 ZIP codes by volume
- ~300,000 property records

---

## Future Price Estimator

Interactive widget (Jupyter) that predicts a property's value in a future year using:

```
future_price = predicted_price × (1 + 0.038)^years_out
```

Annual appreciation rate: **3.8%**

Input fields: sq ft, baths, acreage, year built, ZIP code, city, target year.

---

## Stack

- Python, pandas, NumPy
- scikit-learn (RandomForest, Pipeline, ColumnTransformer)
- ipywidgets (interactive UI)

---

## Usage

```bash
pip install pandas numpy scikit-learn ipywidgets
jupyter notebook RANDFOR_Housing_Prices_Predictor.ipynb
```

Place `WakeCountyHousing.csv` in the same directory before running.

---

## Author

**Suhaan Temkar**
