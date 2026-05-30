# 🧠 AI Training Assistant

A full-stack machine learning workbench that runs entirely on your local machine. Upload a CSV, explore and clean your data, configure a classical ML model or a custom deep neural network, train it with real-time progress streaming, and export the results — all through a browser UI backed by a Python/Flask API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn)

---

## ✨ Features

### 📂 Data Loading
- Upload any CSV file (chunked reading — handles large files)
- Load built-in sklearn datasets: **Iris, Wine, Diabetes, Breast Cancer**
- Automatic column type detection (numeric / categorical / text)

### 🔍 Data Explorer
- Paginated table view with sort and filter on any column
- Per-column statistics (min, max, mean, std, median, quartiles, top values)
- Inline column operations: rename, drop, change type, set as target
- Row operations: delete selected rows, remove duplicates
- Full dataset reset to original uploaded state

### ⚙️ Preprocessing Pipeline
| Option | Choices |
|---|---|
| Missing value strategy | Mean, Median, Mode, Zero fill, Drop rows, KNN Imputation |
| Categorical encoding | One-Hot, Label Encoding |
| Feature scaling | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| Outlier removal | IQR-based automatic removal |
| Duplicate removal | ✓ |
| Feature selection | SelectKBest (top-K) |
| Dimensionality reduction | PCA (configurable variance threshold) |

### 🤖 Classical ML (17 algorithms)
| Category | Algorithms |
|---|---|
| Classification | Random Forest, Gradient Boosting, XGBoost, SVM, KNN, Logistic Regression, Decision Tree, Naive Bayes, AdaBoost |
| Regression | Linear Regression, Ridge, Lasso, ElasticNet, SVR, Random Forest, Gradient Boosting, KNN |
| Clustering | K-Means, DBSCAN |
| Anomaly Detection | Isolation Forest |

### 🧬 Deep Learning (Keras/TensorFlow)
- Visual layer builder — add/remove Dense layers with per-layer neurons, activation, dropout
- Quick architecture presets: Tiny, Small, Medium, Large, XLarge
- **10 activations:** ReLU, GELU, Swish, Sigmoid, Tanh, LeakyReLU, ELU, Softmax, Linear, PReLU
- **7 optimizers:** Adam, AdamW, SGD, RMSprop, Adagrad, Nadam, Adadelta
- **6 weight initializers:** Xavier/Glorot, He Normal, He Uniform, LeCun Normal, Orthogonal, Random Normal
- **LR schedulers:** ReduceOnPlateau, CosineAnnealing, StepDecay, CyclicLR, WarmupLinear
- Early stopping with configurable patience
- L2 regularization, bias toggle, configurable batch size

### 📊 Real-Time Training
- Live loss/accuracy curves streamed epoch-by-epoch via SSE (Server-Sent Events)
- Staged progress for Gradient Boosting (per-estimator updates)
- Final metrics panel: Accuracy, F1, Precision, Recall, R², MAE, RMSE, Silhouette Score
- Confusion matrix + full classification report

### 💾 Exports
- Trained model file (`.pkl` for classical, `.h5` for Keras)
- Training history as CSV
- Processed/cleaned dataset as CSV

---

## 🗂️ Project Structure

```
AI Training Assistant/
├── backend.py              # Flask API server
├── requirements.txt        # Python dependencies
├── ml_uploads/             # Uploaded CSV files (auto-created)
├── ml_models/              # Saved trained models (auto-created)
└── frontend/               # Vite + React app
    ├── src/
    │   ├── App.jsx         # Main frontend (full UI)
    │   └── main.jsx        # React entry point
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm 9+

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-training-assistant.git
cd ai-training-assistant
```

---

### 2. Set up the Python backend

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Set up the React frontend

```bash
cd frontend
npm install
```

---

### 4. Run the app

Open **two terminals**:

**Terminal 1 — Backend** (from the project root):
```bash
# Windows
.venv\Scripts\python.exe backend.py

# macOS / Linux
python backend.py
```

You should see:
```
=======================================================
  AI Training Assistant — Backend
=======================================================
  TensorFlow : ✓  2.21.0
  XGBoost    : ✓
  Uploads    : /path/to/ml_uploads
  Models     : /path/to/ml_models
=======================================================
  Running at http://localhost:5000
```

**Terminal 2 — Frontend** (from the `frontend/` folder):
```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## 🖥️ How to Use

### Step 1 — Load Data
- Click **Upload CSV** to load your own dataset, or
- Click one of the built-in sample datasets (Iris, Wine, Diabetes, Breast Cancer)

### Step 2 — Explore & Clean
- Browse your data in the **Data Explorer** table
- Click any column header to sort; use the filter bar to search
- Right-click a column name to rename, drop, change its type, or view statistics
- Select rows and delete them, or use **Remove Duplicates**

### Step 3 — Set Target Column
- In the column list, click the column you want to predict and mark it as **Target**

### Step 4 — Preprocess
- Go to the **Preprocess** tab
- Choose your missing value strategy, encoding, scaling, and any optional steps (outlier removal, feature selection, PCA)
- Click **Apply Preprocessing** — a log shows exactly what was done

### Step 5 — Configure Model
- Choose **Problem Type** (Classification, Regression, Clustering, etc.)
- Toggle between **Classical ML** and **Deep Learning**
  - Classical: pick an algorithm from the dropdown
  - Deep Learning: build your layer stack visually or choose a preset
- Set hyperparameters (epochs, learning rate, batch size, optimizer, scheduler, etc.)

### Step 6 — Train
- Click **▶ Start Real Training**
- Watch the loss/accuracy curves update in real time
- When training completes, metrics are shown automatically

### Step 7 — Export
- Download the **trained model** (`.pkl` or `.h5`)
- Download **training history** as CSV
- Download the **processed dataset** as CSV

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a CSV file |
| `GET` | `/api/sample/<name>` | Load a built-in sklearn dataset |
| `GET` | `/api/dataset/preview` | Paginated table data |
| `GET` | `/api/dataset/stats/<col>` | Column statistics |
| `POST` | `/api/column/drop` | Drop a column |
| `POST` | `/api/column/rename` | Rename a column |
| `POST` | `/api/column/target` | Set target column |
| `POST` | `/api/column/type` | Change column dtype |
| `POST` | `/api/column/fillnull` | Fill nulls in a column |
| `POST` | `/api/rows/delete` | Delete rows by index |
| `POST` | `/api/rows/dedup` | Remove duplicate rows |
| `POST` | `/api/dataset/reset` | Reset to original data |
| `POST` | `/api/preprocess` | Run preprocessing pipeline |
| `POST` | `/api/train/stream` | Start training (SSE stream) |
| `GET` | `/api/model/export` | Download trained model |
| `GET` | `/api/history/csv` | Download training history |
| `GET` | `/api/dataset/export` | Download processed dataset |
| `GET` | `/api/health` | Backend health check |
| `GET` | `/api/session/info` | Current session metadata |

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, Recharts |
| Backend | Python, Flask, Flask-CORS |
| ML — Classical | scikit-learn 1.8, XGBoost 3.2 |
| ML — Deep Learning | TensorFlow 2.21 / Keras |
| Data | pandas, NumPy |
| Streaming | Server-Sent Events (SSE) |

---

## 🛠️ Troubleshooting

**Backend shows `404` on `/`**
This is normal. The backend has no homepage — all routes are under `/api/`. Visit `http://localhost:5000/api/health` to confirm it's running.

**`recharts` not found error in frontend**
```bash
cd frontend
npm install recharts
```

**TensorFlow not detected**
Deep learning is disabled but classical ML still works. To enable TF:
```bash
pip install tensorflow==2.21.0
```

**Port already in use**
Change the backend port at the bottom of `backend.py`:
```python
app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)
```
And update `frontend/src/App.jsx` line 8:
```js
const API = "http://localhost:5001/api";
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.
