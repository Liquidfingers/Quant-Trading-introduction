# Bitcoin Next-Day Return Prediction: Machine Learning & Manim Visualization

## Project Overview

This project explores the use of machine learning to predict the next-day return of Bitcoin using historical market data, blockchain activity, and macroeconomic indicators.

It compares **Linear Regression** and **Random Forest Regression** using a chronological train-test split. It also uses **Manim** to create animated explanations of the data pipeline, model structures, evaluation metrics, and prediction results.

The goal is to investigate whether the selected features contain useful information for forecasting Bitcoin's next-day returns.

## Objectives

- Prepare and clean historical Bitcoin and financial market data.
- Calculate daily returns using `pct_change()`.
- Build a forward-looking next-day prediction target.
- Train and compare Linear Regression and Random Forest models.
- Evaluate predictions using regression and directional performance indicators.
- Examine Random Forest feature importance.
- Visualize the modelling process through Manim animations.

## Dataset

The dataset contains daily observations and combines several categories of information:

| Category | Examples |
|---|---|
| Bitcoin market data | Closing price, volume, volatility |
| Blockchain indicators | Transactions, block size, hash rate, mining difficulty |
| Macroeconomic data | Exchange rates, S&P 500, Dow Jones, NASDAQ |
| Other financial indicators | VIX, gold, silver, oil |
| Search interest | Google Trends Bitcoin data |

The model uses 25 input features.

**Note:** The notebook expects a file named `btc_final.csv` in the same directory. The dataset is not included in this repository unless its redistribution is permitted.

## Methodology

### 1. Data Preparation

Daily Bitcoin and S&P 500 returns are calculated using percentage change:

```python
df["btc_returns"] = df["bt_close"].pct_change()
df["sp_returns"] = df["sp_close"].pct_change()
```

Invalid observations associated with zero prices are handled before model training.

### 2. Target Construction

The prediction target is the next-day Bitcoin return:

```python
df["btc_target"] = df["btc_returns"].shift(-1)
```

This means the model uses information available at day *t* to predict the Bitcoin return at day *t+1*.

### 3. Model Training

Two regression algorithms are compared:

- **Linear Regression:** Establishes a linear modelling baseline.
- **Random Forest Regression:** Uses an ensemble of decision trees to model nonlinear relationships.

The Random Forest configuration uses 500 estimators, a maximum depth of 8, and a minimum of 10 samples per leaf.

### 4. Validation Strategy

The dataset is divided chronologically using an 80/20 split. The observations are not shuffled, preserving the time-series order.

### 5. Evaluation Indicators

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Directional Accuracy
- Spearman Information Coefficient (IC)
- Spearman IC p-value

A training-mean prediction is also included as a naive baseline.

## Results

The following are the recomputed test-set results using the forward next-day target.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.355044 | 0.461412 | -82.126934 |
| Random Forest | 0.036836 | 0.051199 | -0.023510 |
| Training-mean baseline | 0.035775 | 0.050677 | -0.002728 |

Additional Random Forest diagnostics:

- Directional Accuracy: **54.31%**
- Spearman IC: **0.070754**
- Spearman IC p-value: **0.160999**

### Interpretation

Random Forest performs substantially better than Linear Regression. However, it does not outperform the training-mean baseline on MAE, RMSE, or R².

Its directional accuracy is modest, and the Spearman IC is not statistically significant at conventional significance levels.

Therefore, these results do not establish a reliable or profitable trading signal. They provide a baseline for further experimentation.

## Manim Visualization

The notebook includes an animated model explainer covering:

1. The financial data pipeline and prediction target.
2. The chronological training and testing split.
3. Linear Regression and its prediction process.
4. Random Forest decision-tree structure and ensemble averaging.
5. Actual versus predicted test-set returns.
6. Model performance comparisons and research conclusions.

The animations are generated from the notebook's model outputs rather than manually entered performance figures.
## Project Animation

The project includes a Manim-generated animation demonstrating the machine-learning workflow, model architectures, prediction results, and performance evaluation.

<video src="https://github.com/Liquidfingers/Quant-Trading-introduction/blob/main/BTCModelLab%402026-11-22%4011-19-38.mp4" controls></video>

**[▶ Watch the BTC Model Lab Animation (MP4)](https://github.com/Liquidfingers/Quant-Trading-introduction/blob/main/BTCModelLab%402026-11-22%4011-19-38.mp4)**


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Jupyter Notebook / JupyterLab
- Manim

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `btc_final.csv` in the same directory as the notebook.

### 5. Launch JupyterLab

```bash
jupyter lab
```

Open `BTC_ML_Manim_Explainer.ipynb` and execute the cells from top to bottom.

The final section renders the Manim animation. Manim may also require system-level dependencies such as FFmpeg and LaTeX.

## Future Improvements

- Experiment with XGBoost and other ensemble models.
- Introduce walk-forward validation.
- Audit feature publication times to reduce look-ahead bias.
- Investigate feature selection and hyperparameter tuning.
- Develop a signal-generation and backtesting framework.
- Incorporate transaction costs, slippage, and portfolio risk metrics.
- Evaluate performance across different market regimes.

## Disclaimer

This project is for educational and research purposes only. Its predictions are not financial advice or a recommendation to trade Bitcoin. The reported results do not demonstrate a profitable trading strategy.

## Author

**Victor Nnamani**

GitHub: [Liquidfingers](https://github.com/Liquidfingers)
