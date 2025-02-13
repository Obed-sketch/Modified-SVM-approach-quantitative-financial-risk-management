# Algorithm in Quantitative Financial Risk Management
# To cite this article: JinPeng Zhu and HanChen Wang 2020 J. Phys.: Conf. Ser. 1648 042093

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# 1. Data Preparation & Feature Engineering
def prepare_financial_data(ticker='^SSEC', start='2017-01-01', end='2018-02-01'):
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end)
    
    # Calculate technical indicators
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].pct_change().rolling(7).std()
    
    # Create binary labels (1=extreme risk, 0=normal)
    data['Label'] = np.where(data['Close'].pct_change(5) < -0.10, 1, 0)
    
    return data.dropna()

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 2. BADASYN Implementation
class BADASYN:
    def __init__(self, k=5, ratio=0.5):
        self.k = k
        self.ratio = ratio

    def fit_resample(self, X, y):
        minority_class = 1
        X_min = X[y == minority_class]
        X_maj = X[y != minority_class]
        
        # Find borderline samples
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(X)
        neighbors = nn.kneighbors(X_min, return_distance=False)
        
        borderline = []
        for i, neigh in enumerate(neighbors):
            if np.mean(y[neigh] != minority_class) >= 0.5:
                borderline.append(X_min[i])
        
        # Generate synthetic samples
        n_samples = int(len(X_maj)*self.ratio - len(X_min))
        synthetic = []
        for _ in range(n_samples):
            sample = borderline[np.random.randint(len(borderline))]
            neighbor = borderline[np.random.randint(len(borderline))]
            synth = sample + (neighbor - sample) * np.random.rand()
            synthetic.append(synth)
            
        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.ones(len(synthetic))])
        return X_resampled, y_resampled

# 3. zSVM Implementation
class zSVM(SVC):
    def __init__(self, z=0.1, **kwargs):
        super().__init__(**kwargs)
        self.z = z
        
    def predict(self, X):
        dec = self.decision_function(X)
        return np.sign(dec + self.z*self.intercept_)

# 4. NCAB-SVM Ensemble
class NCAB_SVM:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        sample_weights = np.ones(len(X)) / len(X)
        
        for _ in range(self.n_estimators):
            svm = zSVM(kernel='rbf', probability=True)
            svm.fit(X, y, sample_weight=sample_weights)
            
            pred = svm.predict(X)
            err = np.sum(sample_weights * (pred != y)) + 1e-10
            
            alpha = self.learning_rate * np.log((1 - err) / err)
            sample_weights *= np.exp(alpha * (pred != y))
            sample_weights /= sample_weights.sum()
            
            # Negative correlation term
            if len(self.models) > 0:
                prev_pred = np.array([m.predict(X) for m in self.models])
                diversity = np.mean(np.cov(prev_pred))
                alpha *= (1 - diversity)
            
            self.models.append(svm)
            self.alphas.append(alpha)
            
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.sign(np.dot(self.alphas, preds))

# 5. Evaluation Metrics
def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    return np.sqrt(specificity * sensitivity)

# Main Workflow
if __name__ == "__main__":
    # Load and prepare data
    data = prepare_financial_data()
    X = data[['MA_50', 'RSI', 'Volatility']].values
    y = data['Label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Handle imbalance
    badasyn = BADASYN(k=5, ratio=0.7)
    X_res, y_res = badasyn.fit_resample(X_train, y_train)
    
    # Normalize data
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = NCAB_SVM(n_estimators=50)
    model.fit(X_res, y_res)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print(f"G-Mean: {g_mean(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")
