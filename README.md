# PakWheels Used Car Price Predictor — GUI

A Streamlit GUI for the trained SVR pipeline that predicts used-car
prices (in PKR) from PakWheels listing data.

**Author:** Abdul Rehman
[GitHub](https://github.com/datawithabdulrehman) ·
[LinkedIn](https://www.linkedin.com/in/datawithabdulrehman) ·
[Kaggle](https://www.kaggle.com/datawithabxrehman)

## Folder contents

Put these four files in the **same folder**:

```
app.py
requirements.txt
pakwheels_best_price_model.joblib
pakwheels_model_metadata.joblib
```

## Setup & run

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. install exact dependencies (matches the training environment)
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Inputs

| Field         | Description                                  |
|---------------|-----------------------------------------------|
| Brand         | Car manufacturer (61 brands from training data) |
| Year          | Manufacturing year                            |
| Fuel Type     | petrol / diesel / hybrid / electric / cng / phev / reev |
| Transmission  | automatic / manual                            |
| Engine (CC)   | Engine displacement in cc                     |
| Mileage (KM)  | Total kilometers driven                       |

`vehicle_age` is computed automatically as `current_year - year`, exactly
as it was engineered during training — you don't need to enter it.

## Notes

- The model is a scikit-learn `Pipeline` (imputation → scaling / one-hot
  encoding → **SVR**). `requirements.txt` pins `scikit-learn==1.6.1` to
  match the version the `.joblib` file was saved with — installing a
  different version can cause a load error, since scikit-learn pickles
  are not guaranteed to be forward/backward compatible.
- Predictions are estimates (test MAE ≈ 1.06M PKR, R² ≈ 0.90) — real
  listing prices vary with condition, location, negotiation, and market
  timing.
