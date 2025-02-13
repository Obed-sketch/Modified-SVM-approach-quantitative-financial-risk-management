# Modified-SVM-approach-quantitative-financial-risk-management
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

**5. Evaluation Metrics**: Implement functions to calculate G-mean, F-measure, and AUC. Scikit-learn has some of these, but G-mean might need to be custom-coded.
        
