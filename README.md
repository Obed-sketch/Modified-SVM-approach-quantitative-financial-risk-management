# Application of Unbalanced Data Classification
# Algorithm in Quantitative Financial Risk Management
To cite this article: JinPeng Zhu and HanChen Wang 2020 J. Phys.: Conf. Ser. 1648 042093

**Modified-SVM-approach-quantitative-financial-risk-management**
The paper focuses on applying an unbalanced data classification algorithm, specifically a modified SVM approach, for quantitative financial risk management. The key components mentioned are the BADASYN algorithm for handling imbalanced data, integrating negative correlation learning with AdaBoost for SVM, and evaluating using metrics like G-mean, F-measure, and AUC.

1. **Data Preparation**: They selected data from the Chinese stock market from 2017 to 2018, focusing on extreme financial risks. The features include technical indicators and possibly sentiment data.
2.  **BADASYN Algorithm**: This is a variant of the SMOTE algorithm that focuses on boundary samples. The steps involve identifying boundary samples in the minority class and adaptively synthesizing new samples based on their distribution.
3.   **SVM Model with Adjustments**: The paper mentions a modified SVM (zSVM) that adjusts the decision boundary by weighting the support vectors of the minority class. Additionally, they integrate negative correlation learning with AdaBoost to improve the ensemble's performance.
4. **Evaluation Metrics**: They use G-mean, F-measure, and AUC instead of accuracy due to the imbalanced nature of the data.



   **1. Data Preparation**: I'll need financial data. Since the original paper uses Chinese stock data, maybe I can use Yahoo Finance or another source to get similar data. I'll also need to compute technical indicators like RSI, Moving Averages, and Volatility.

   RSI (Relative Strength Index)
What it is: A momentum indicator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions.
Range: 0–100.
Above 70: Overbought (potentially due for a pullback).
Below 30: Oversold (potentially due for a bounce).
Purpose: Identifies momentum shifts and potential reversals.
📊 Moving Averages (MA)
What it is: A trend-following indicator that smooths past price data to identify the direction of a trend.
Types:
SMA (Simple Moving Average) – equal weight to all data points.
EMA (Exponential Moving Average) – more weight to recent data.
Purpose: Highlights trends and possible support/resistance levels.
🌪️ Volatility
What it is: Measures the degree of variation in a security's price over time.
Indicators:
Bollinger Bands – wider bands mean higher volatility.
ATR (Average True Range) – shows average price range over a period.
Purpose: Assesses risk and potential price swings.
⚖️ RSI vs. Moving Averages
RSI: Momentum indicator—shows how strong or weak a price move is.
MA: Trend-following indicator—smooths out noise to reveal the trend.
Key Difference: RSI helps spot reversals, while MAs help confirm the trend direction.


# SMOTE (Synthetic Minority Over-sampling Technique) is a method used in machine learning to address class imbalance by generating synthetic samples for the minority class.

⚙️ How It Works:
Identify Minority Class: Select samples from the underrepresented class.
Find Nearest Neighbors: For each selected sample, find its k-nearest neighbors.
Generate Synthetic Samples: New samples are created by interpolating between the original sample and one of its neighbors.
📊 Why Use SMOTE?
Improves model performance on imbalanced datasets.
Reduces bias toward the majority class.
Works better than simple duplication of minority class samples.
🚨 Limitations:
Can create noisy or overlapping samples if not carefully applied.
May lead to overfitting if the dataset is too small.


2. **BADASYN Algorithm**: This is a variant of the SMOTE algorithm that focuses on boundary samples. The steps involve identifying boundary samples in the minority class and adaptively synthesizing new samples based on their distribution.

3. **SVM Model with Adjustments**: The paper mentions a modified SVM (zSVM) that adjusts the decision boundary by weighting the support vectors of the minority class. Additionally, they integrate negative correlation learning with AdaBoost to improve the ensemble's performance.

4. **Evaluation Metrics**: They use G-mean, F-measure, and AUC instead of accuracy due to the imbalanced nature of the data.

5. **4. Negative Correlation Learning with AdaBoost**: The paper combines AdaBoost with negative correlation learning to diversify the ensemble. This involves training multiple SVMs where each subsequent model focuses on samples that previous models found hard to classify, while also penalizing correlation between models. Implementing this would require customizing the AdaBoost algorithm to include a penalty term for correlation.

**6. Key implementation notes:**

**Data Preparation:**
Uses Yahoo Finance to get stock data
Calculates technical indicators (MA, RSI, Volatility)
Creates binary labels for extreme risk events
**BADASYN:**
Identifies borderline minority samples
Generates synthetic samples near decision boundaries
Balances class distribution
**zSVM:**
Inherits from sklearn's SVC
Adjusts decision boundary using z parameter
Focuses on minority class prediction
**NCAB-SVM:**
Implements negative correlation learning
Combines AdaBoost with diversity penalty
Uses weighted ensemble of zSVM models
Evaluation:
Implements G-mean metric
Uses F1-score and AUC for performance assessment

**5. Evaluation Metrics**: Implement functions to calculate G-mean, F-measure, and AUC. Scikit-learn has some of these, but G-mean might need to be custom-coded.
        

---

## About the code

`SVM_Quantitative_risk_management.py` is a re-implementation of the algorithm
in Zhu & Wang (2020), cited at the top of the file: a custom BADASYN
oversampler, a modified zSVM, and a negative-correlation AdaBoost ensemble
(NCAB-SVM), applied to Shanghai Composite (^SSEC) daily data.

## How to run

```bash
pip install -r requirements.txt
python SVM_Quantitative_risk_management.py
```

Needs an internet connection (price data comes from Yahoo Finance via
`yfinance`). A full run takes ~3-4 minutes, almost all of it in the
50-member NCAB ensemble.

## Run status (verified Nov 2026)

The script runs end-to-end and prints its evaluation metrics. Two fixes
made that possible; the method itself (features, label, BADASYN, zSVM,
NCAB ensemble) is unchanged from the paper:

1. **Data layer.** Yahoo no longer serves `^SSEC` (0 rows), and in the
   paper's calm 2017-2018 window the extreme-risk label (5-day drop
   worse than -10%) never fires for a broad index, which makes a
   stratified split impossible. `prepare_financial_data()` therefore
   falls back, with printed notices, in this order: `^SSEC` →
   `000001.SS` (the same Shanghai Composite index under the symbol
   Yahoo actually serves) → `^GSPC`; and if a window yields fewer than
   2 positive labels it widens to 2015-2020, a span containing real
   crash episodes (2015 China crash, 2018 sell-offs, 2020 COVID crash).
   The verified run used `000001.SS`, 2015-01-01 to 2020-12-31:
   1,410 usable rows, 18 extreme-risk rows.
2. **Ensemble label fix.** zSVM's `predict()` returns sign-space values
   in {-1, 0, 1} while the training labels are {0, 1}. The NCAB
   ensemble originally compared them directly when computing AdaBoost
   errors, which degenerated the sample weights and crashed training.
   Predictions are now mapped onto {0, 1} before the error/weight
   update (see `_as_class_labels` and the comments in the code).

Verified output of the run described above:

```
G-Mean: 0.0000
F1-Score: 0.0000
AUC: 0.5000
```

That is an honest result, not a defect in the pipeline: with only 18
positive days in six years (5 in the test split), identified by a label
that uses the *trailing* 5-day return while the features describe the
same window, the ensemble finds no exploitable signal and predicts the
majority class. The paper's headline numbers are not reproducible from
public index data alone.
