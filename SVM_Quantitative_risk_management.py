# Algorithm in Quantitative Financial Risk Management
# To cite this article: JinPeng Zhu and HanChen Wang 2020 J. Phys.: Conf. Ser. 1648 042093

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# 1. Data Preparation & Feature Engineering
def _features_and_labels(data):
    """Technical indicators + the paper's binary extreme-risk label."""
    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.copy()
    # Calculate technical indicators
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].pct_change().rolling(7).std()
    # Create binary labels (1=extreme risk, 0=normal): trailing 5-day
    # drop worse than -10% (the definition in Zhu & Wang 2020).
    data['Label'] = np.where(data['Close'].pct_change(5) < -0.10, 1, 0)
    return data.dropna()

def prepare_financial_data(ticker='^SSEC', start='2017-01-01', end='2018-02-01'):
    import yfinance as yf

    # Data-availability fallbacks (added Nov 2026 so the script runs
    # end-to-end again; the method -- features, label, models -- is
    # unchanged from the paper):
    #   * Yahoo no longer serves '^SSEC' (0 rows, "possibly delisted").
    #     '000001.SS' is the SAME Shanghai Composite index under the
    #     symbol Yahoo actually serves, so it is tried next; '^GSPC'
    #     (S&P 500) is the last-resort liquid alternative.
    #   * In the paper's calm 2017-2018 window the -10%/5-day label
    #     never fires for a broad index (0 positives), which makes a
    #     stratified train/test split impossible. If that happens, the
    #     window is widened to 2015-2020 -- a span containing genuine
    #     crash episodes (2015 China crash, 2018 sell-offs, 2020 COVID
    #     crash). Every substitution is printed, never silent.
    candidates = [ticker] + [t for t in ('000001.SS', '^GSPC') if t != ticker]
    for cand in candidates:
        for (s, e) in [(start, end), ('2015-01-01', '2020-12-31')]:
            raw = yf.download(cand, start=s, end=e, progress=False)
            if raw is None or len(raw) == 0:
                print(f"[data] {cand} {s}..{e}: no rows returned by Yahoo Finance -- trying next option")
                continue
            data = _features_and_labels(raw)
            n_pos = int(data['Label'].sum())
            print(f"[data] {cand} {s}..{e}: {len(data)} usable rows, {n_pos} extreme-risk (label=1) rows")
            if n_pos >= 2:
                if cand != ticker or (s, e) != (start, end):
                    print(f"[data] NOTE: using {cand} {s}..{e} as a substitute for the "
                          f"unavailable/unusable {ticker} {start}..{end} (see README).")
                return data
            print(f"[data] {cand} {s}..{e}: label never fires (<2 positives), cannot stratify -- trying next option")
    raise RuntimeError("No usable dataset found: every ticker/window fallback was exhausted.")

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

        if not borderline:
            # Robustness (Nov 2026): if no minority sample is borderline,
            # fall back to the full minority set instead of crashing on
            # an empty choice list.
            borderline = list(X_min)

        # Generate synthetic samples
        n_samples = int(len(X_maj)*self.ratio - len(X_min))
        if n_samples <= 0:
            return X, y
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

def _as_class_labels(raw_pred):
    """Map sign-space predictions ({-1, 0, 1}) onto class labels ({0, 1}).

    zSVM.predict() returns the sign of its (shifted) decision function,
    i.e. values in {-1, 0, 1}, while the training labels are {0, 1}.
    A positive sign means "class 1 side of the boundary", so threshold
    at 0. Used by NCAB_SVM both when scoring ensemble members during
    fitting and when producing the ensemble's final predictions.
    """
    return (np.asarray(raw_pred) > 0).astype(int)

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
            
            # Label-space fix (Nov 2026): svm.predict here returns
            # {-1, 0, 1} but y is {0, 1}; comparing them directly counts
            # nearly every prediction as an error, which degenerates the
            # AdaBoost sample weights (they collapse toward zero and SVC
            # then refuses to fit). Map to {0, 1} BEFORE the error and
            # weight update.
            pred = _as_class_labels(svm.predict(X))
            err = np.sum(sample_weights * (pred != y)) + 1e-10
            err = min(err, 1.0 - 1e-10)  # keep log((1 - err) / err) finite

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
        # Same label-space fix as in fit(): the weighted vote is in sign
        # space ({-1, 0, 1}); map it onto the {0, 1} label space so the
        # evaluation metrics compare like with like.
        return _as_class_labels(np.dot(self.alphas, preds))

# 5. Evaluation Metrics
def g_mean(y_true, y_pred):
    # labels=[0, 1] keeps the matrix 2x2 even if a class is missing
    # from the predictions (robustness fix, Nov 2026).
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    return np.sqrt(specificity * sensitivity)

# Main Workflow
if __name__ == "__main__":
    np.random.seed(42)  # reproducible BADASYN sampling (added Nov 2026)
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
