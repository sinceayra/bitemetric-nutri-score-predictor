
# BiteMetric: Nutri-Score Predictor

BiteMetric is a machine learning web application that predicts the Nutri-Score grade (**A** through **E**) of packaged food products based on their nutritional composition per 100g.

## Overview

Nutri-Score translates complex nutrition labels into a simple five-letter scale from **A** (highest nutritional quality) to **E** (lowest nutritional quality). This project trains an XGBoost multi-class classifier on product nutrition data and serves real-time predictions through a Streamlit interface.

## Project Structure

```text
bitemetric-nutri-score-predictor/
├── backend/                # Data preprocessing and XGBoost model training scripts
├── frontend/               # Streamlit web application
├── saved_models/           # Serialized XGBoost model and label encoder (.pkl)
├── requirements.txt        # Python dependencies
└── README.md

```

## Features & Methodology

### Input Features (per 100g)

* **Energy** (`energy_100g`, in kJ)
* **Sugars** (`sugars_100g`, in g)
* **Saturated Fat** (`saturated-fat_100g`, in g)
* **Sodium** (`sodium_100g`, in g)
* **Fiber** (`fiber_100g`, in g)
* **Proteins** (`proteins_100g`, in g)

### Engineered Features

To capture nutrient balance rather than relying solely on raw totals, two domain-specific ratios are computed during preprocessing and inference:

* **Sugar-to-Fiber Ratio**: `sugars / (fiber + 1.0)`
* **Protein-to-Fat Ratio**: `proteins / (saturated_fat + 1.0)`

### Model

* **Algorithm**: XGBoost Classifier (`XGBClassifier`)
* **Dataset**: Open Food Facts nutritional records
* **Target**: Nutri-Score grades (`A`, `B`, `C`, `D`, `E`) encoded via `LabelEncoder`

## Local Setup

1. **Clone the repository**
```bash
git clone [https://github.com/sinceayra/bitemetric-nutri-score-predictor.git](https://github.com/sinceayra/bitemetric-nutri-score-predictor.git)
cd bitemetric-nutri-score-predictor

```


2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the Streamlit app**
```bash
streamlit run frontend/app.py

```


*(Replace `app.py` with the exact filename inside your `frontend/` folder if named differently.)*

## Tech Stack

* **Language**: Python
* **Data Processing**: pandas, NumPy
* **Machine Learning**: XGBoost, scikit-learn, joblib
* **Frontend**: Streamlit

```

```
