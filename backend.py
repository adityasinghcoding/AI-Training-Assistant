"""
ML Training Assistant — Real Backend
====================================
Install:  pip install flask flask-cors pandas numpy scikit-learn xgboost tensorflow joblib
Run:      python backend.py
Port:     5000 (backend) and 3000 (frontend)
"""

import os, io, json, uuid, time, queue, threading, traceback, pickle, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from datetime import datetime
from flask      import Flask, request, jsonify, Response, send_file
from flask_cors import CORS

# ── Optional heavy deps (graceful fallback) ───────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.ensemble         import (RandomForestClassifier, RandomForestRegressor,
                                      GradientBoostingClassifier, GradientBoostingRegressor,
                                      IsolationForest, AdaBoostClassifier, AdaBoostRegressor)
from sklearn.linear_model     import (LogisticRegression, Ridge, Lasso, ElasticNet,
                                      LinearRegression, SGDClassifier)
from sklearn.svm              import SVC, SVR
from sklearn.neighbors        import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree             import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes      import GaussianNB
from sklearn.cluster          import KMeans, DBSCAN
from sklearn.preprocessing    import (StandardScaler, MinMaxScaler, RobustScaler,
                                      LabelEncoder, OneHotEncoder, MaxAbsScaler)
from sklearn.decomposition    import PCA
from sklearn.impute           import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, f1_score, precision_score, recall_score,
                                      r2_score, mean_squared_error, mean_absolute_error,
                                      classification_report, confusion_matrix,
                                      silhouette_score)
from sklearn.pipeline         import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_DIR = "./ml_uploads"
MODEL_DIR  = "./ml_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# sid → { df, raw_df, target, feature_cols, cat_cols, num_cols,
#         prep_config, model, metrics, history, model_type, algo, layers, hp }
sessions     = {}
train_queues = {}   # sid → queue.Queue  (for SSE)

# ─── helpers ──────────────────────────────────────────────────────────────────

def sid_ok(sid):
    return sid and sid in sessions

def col_dtype(series):
    nulls = int(series.isna().sum())
    uniq  = int(series.nunique())
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", nulls, uniq
    if uniq <= max(2, int(len(series) * 0.05)):
        return "categorical", nulls, uniq
    return "text", nulls, uniq

def col_stats(series, dtype):
    base = {"nulls": int(series.isna().sum()), "unique": int(series.nunique()), "type": dtype}
    if dtype == "numeric":
        s = series.dropna()
        if len(s):
            base.update({
                "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                "median": round(float(s.median()), 4),
                "q1": round(float(s.quantile(0.25)), 4),
                "q3": round(float(s.quantile(0.75)), 4),
            })
    else:
        vc = series.value_counts().head(10)
        base["top"] = [[str(k), int(v)] for k, v in vc.items()]
    return base

def push(sid, event, data):
    if sid in train_queues:
        train_queues[sid].put(json.dumps({"event": event, "data": data}))

def emit_done(sid):
    push(sid, "done", {})
    train_queues[sid].put(None)   # sentinel

# ─── UPLOAD ───────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    sid = request.form.get("sid", str(uuid.uuid4()))
    f   = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400

    path = os.path.join(UPLOAD_DIR, f"{sid}_{f.filename}")
    f.save(path)

    try:
        # read in chunks for huge files
        chunks = []
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False,
                                  encoding="utf-8", on_bad_lines="skip"):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        return jsonify({"error": f"Parse error: {e}"}), 400

    sessions[sid] = {
        "df": df.copy(), "raw_df": df.copy(),
        "target": None, "feature_cols": list(df.columns),
        "cat_cols": [], "num_cols": [],
        "prep_config": {}, "model": None,
        "metrics": {}, "history": [],
        "model_type": "deep", "algo": "Random Forest",
        "layers": [], "hp": {},
        "model_path": None,
    }

    cols = []
    for c in df.columns:
        dt, nulls, uniq = col_dtype(df[c])
        cols.append({"name": c, "type": dt, "nulls": nulls, "unique": uniq, "dropped": False, "isTarget": False})

    return jsonify({
        "sid": sid,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": cols,
        "filename": f.filename,
    })

# ─── SAMPLE DATASET ───────────────────────────────────────────────────────────

@app.route("/api/sample/<name>", methods=["GET"])
def load_sample(name):
    sid = request.args.get("sid", str(uuid.uuid4()))

    if name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    elif name == "diabetes":
        from sklearn.datasets import load_diabetes
        d = load_diabetes(as_frame=True)
        df = d.frame
    elif name == "wine":
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        df = d.frame
    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        df = d.frame
    else:
        return jsonify({"error": "Unknown sample"}), 400

    sessions[sid] = {
        "df": df.copy(), "raw_df": df.copy(),
        "target": None, "feature_cols": list(df.columns),
        "cat_cols": [], "num_cols": [],
        "prep_config": {}, "model": None,
        "metrics": {}, "history": [],
        "model_type": "deep", "algo": "Random Forest",
        "layers": [], "hp": {},
        "model_path": None,
    }

    cols = []
    for c in df.columns:
        dt, nulls, uniq = col_dtype(df[c])
        cols.append({"name": c, "type": dt, "nulls": nulls, "unique": uniq, "dropped": False, "isTarget": False})

    return jsonify({"sid": sid, "rows": len(df), "cols": len(df.columns), "columns": cols, "filename": name})

# ─── DATASET PREVIEW ─────────────────────────────────────────────────────────

@app.route("/api/dataset/preview", methods=["GET"])
def dataset_preview():
    sid    = request.args.get("sid")
    page   = int(request.args.get("page", 0))
    limit  = int(request.args.get("limit", 50))
    fcol   = request.args.get("filterCol", "")
    fval   = request.args.get("filterVal", "")
    scol   = request.args.get("sortCol", "")
    sdir   = request.args.get("sortDir", "asc")

    if not sid_ok(sid):
        return jsonify({"error": "Session not found"}), 404

    df = sessions[sid]["df"].copy()

    if fcol and fval and fcol in df.columns:
        df = df[df[fcol].astype(str).str.contains(fval, case=False, na=False)]

    if scol and scol in df.columns:
        try:
            df = df.sort_values(scol, ascending=(sdir == "asc"))
        except Exception:
            pass

    total = len(df)
    page_df = df.iloc[page * limit : (page + 1) * limit]
    rows = []
    for _, row in page_df.iterrows():
        r = {}
        for k, v in row.items():
            r[k] = None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else
                                             float(round(v, 6)) if isinstance(v, (np.floating, float)) else str(v))
        rows.append(r)

    return jsonify({"rows": rows, "total": total, "page": page, "pages": max(1, -(-total // limit))})

# ─── COLUMN STATS ────────────────────────────────────────────────────────────

@app.route("/api/dataset/stats/<col>", methods=["GET"])
def dataset_stats(col):
    sid = request.args.get("sid")
    if not sid_ok(sid):
        return jsonify({"error": "Session not found"}), 404
    df = sessions[sid]["df"]
    if col not in df.columns:
        return jsonify({"error": "Column not found"}), 404
    dt, _, _ = col_dtype(df[col])
    return jsonify(col_stats(df[col], dt))

# ─── COLUMN OPERATIONS ───────────────────────────────────────────────────────

@app.route("/api/column/drop", methods=["POST"])
def column_drop():
    body = request.json
    sid, col = body.get("sid"), body.get("col")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    df = sessions[sid]["df"]
    if col in df.columns:
        sessions[sid]["df"] = df.drop(columns=[col])
    return jsonify({"ok": True, "rows": len(sessions[sid]["df"]), "cols": list(sessions[sid]["df"].columns)})

@app.route("/api/column/rename", methods=["POST"])
def column_rename():
    body = request.json
    sid, old, new = body.get("sid"), body.get("old"), body.get("new")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    sessions[sid]["df"] = sessions[sid]["df"].rename(columns={old: new})
    if sessions[sid].get("target") == old:
        sessions[sid]["target"] = new
    return jsonify({"ok": True})

@app.route("/api/column/target", methods=["POST"])
def column_target():
    body = request.json
    sid, col = body.get("sid"), body.get("col")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    sessions[sid]["target"] = col
    return jsonify({"ok": True, "target": col})

@app.route("/api/column/type", methods=["POST"])
def column_type():
    body = request.json
    sid, col, dtype = body.get("sid"), body.get("col"), body.get("dtype")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    df = sessions[sid]["df"]
    if dtype == "numeric":
        sessions[sid]["df"][col] = pd.to_numeric(df[col], errors="coerce")
    else:
        sessions[sid]["df"][col] = df[col].astype(str)
    return jsonify({"ok": True})

@app.route("/api/rows/delete", methods=["POST"])
def rows_delete():
    body = request.json
    sid, indices = body.get("sid"), body.get("indices", [])
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    df = sessions[sid]["df"]
    sessions[sid]["df"] = df.drop(index=df.index[indices]).reset_index(drop=True)
    return jsonify({"ok": True, "rows": len(sessions[sid]["df"])})

@app.route("/api/rows/dedup", methods=["POST"])
def rows_dedup():
    sid = request.json.get("sid")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    before = len(sessions[sid]["df"])
    sessions[sid]["df"] = sessions[sid]["df"].drop_duplicates().reset_index(drop=True)
    return jsonify({"ok": True, "removed": before - len(sessions[sid]["df"]), "rows": len(sessions[sid]["df"])})

@app.route("/api/dataset/reset", methods=["POST"])
def dataset_reset():
    sid = request.json.get("sid")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    sessions[sid]["df"] = sessions[sid]["raw_df"].copy()
    return jsonify({"ok": True, "rows": len(sessions[sid]["df"])})

# ─── FILL NULLS IN COLUMN ────────────────────────────────────────────────────

@app.route("/api/column/fillnull", methods=["POST"])
def fill_null():
    body = request.json
    sid, col, strategy = body.get("sid"), body.get("col"), body.get("strategy", "mean")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    df = sessions[sid]["df"]
    if col not in df.columns: return jsonify({"error": "Column not found"}), 404
    s = pd.to_numeric(df[col], errors="coerce")
    fill = s.mean() if strategy == "mean" else s.median() if strategy == "median" else 0
    sessions[sid]["df"][col] = s.fillna(fill).round(4)
    return jsonify({"ok": True, "filled": int(df[col].isna().sum())})

# ─── PREPROCESSING ───────────────────────────────────────────────────────────

@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    body = request.json
    sid  = body.get("sid")
    cfg  = body.get("config", {})
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404

    df     = sessions[sid]["df"].copy()
    target = sessions[sid].get("target")
    log    = []

    # ── Missing values ──────────────────────────────────────────────────────
    strategy = cfg.get("missingStrategy", "Mean Imputation")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target in cat_cols: cat_cols.remove(target)

    if strategy == "Drop Rows":
        before = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        log.append(f"Dropped {before - len(df)} rows with missing values")
    else:
        if num_cols:
            sk = "mean" if "Mean" in strategy else "median" if "Median" in strategy else "most_frequent" if "Mode" in strategy else "constant"
            kw = {} if sk != "constant" else {"fill_value": 0}
            if "KNN" in strategy and len(df) < 50000:
                imp = KNNImputer(n_neighbors=5)
            else:
                imp = SimpleImputer(strategy=sk if sk != "constant" else "constant", **kw)
            before_nulls = int(df[num_cols].isna().sum().sum())
            df[num_cols] = imp.fit_transform(df[num_cols])
            log.append(f"Imputed {before_nulls} numeric nulls ({strategy})")
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if len(df) else "unknown")
            log.append(f"Filled categorical nulls with mode")

    # ── Duplicates ─────────────────────────────────────────────────────────
    if cfg.get("removeDuplicates"):
        b = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        log.append(f"Removed {b - len(df)} duplicates")

    # ── Outlier removal ─────────────────────────────────────────────────────
    if cfg.get("outlierRemoval") and num_cols:
        b = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        for c in num_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            mask &= df[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask].reset_index(drop=True)
        log.append(f"Removed {b - len(df)} outlier rows (IQR)")

    # ── Categorical encoding ───────────────────────────────────────────────
    enc = cfg.get("encoding", "One-Hot")
    if enc == "One-Hot" and cat_cols:
        try:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
            log.append(f"One-hot encoded {len(cat_cols)} categorical columns")
        except Exception as e:
            log.append(f"One-hot encoding warning: {e}")
    elif enc == "Label Encoding" and cat_cols:
        le = LabelEncoder()
        for c in cat_cols:
            df[c] = le.fit_transform(df[c].astype(str))
        log.append(f"Label-encoded {len(cat_cols)} categorical columns")

    # ── Feature scaling ────────────────────────────────────────────────────
    scaling = cfg.get("scaling", "StandardScaler")
    feat_num = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if scaling != "None" and feat_num:
        sc = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(),
              "RobustScaler": RobustScaler(), "MaxAbsScaler": MaxAbsScaler()}.get(scaling, StandardScaler())
        df[feat_num] = sc.fit_transform(df[feat_num])
        log.append(f"Applied {scaling} to {len(feat_num)} numeric features")

    # ── Feature selection ──────────────────────────────────────────────────
    if cfg.get("featureSelection") and target and target in df.columns:
        feat_cols = [c for c in df.columns if c != target]
        X_ = df[feat_cols].select_dtypes(include=[np.number])
        y_ = df[target]
        k  = min(cfg.get("topK", 20), len(X_.columns))
        try:
            sel = SelectKBest(f_classif if y_.nunique() <= 20 else f_regression, k=k)
            sel.fit(X_, y_)
            best = [X_.columns[i] for i in sel.get_support(indices=True)]
            keep = [target] + best + [c for c in df.columns if c not in X_.columns and c != target]
            df   = df[[c for c in keep if c in df.columns]]
            log.append(f"Selected top {k} features via SelectKBest")
        except Exception as e:
            log.append(f"Feature selection skipped: {e}")

    # ── PCA ────────────────────────────────────────────────────────────────
    pca_obj = None
    if cfg.get("pca") and target and target in df.columns:
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        if feat_cols:
            try:
                var = float(cfg.get("pcaVariance", 0.95))
                pca_obj = PCA(n_components=var)
                X_pca = pca_obj.fit_transform(df[feat_cols])
                pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
                df = pd.concat([pca_df, df[[target]].reset_index(drop=True)], axis=1)
                log.append(f"PCA: reduced to {X_pca.shape[1]} components ({var*100:.0f}% variance)")
            except Exception as e:
                log.append(f"PCA skipped: {e}")

    sessions[sid]["df"]         = df
    sessions[sid]["prep_config"] = cfg
    sessions[sid]["prep_log"]   = log
    sessions[sid]["pca_obj"]    = pca_obj

    return jsonify({
        "ok": True, "log": log,
        "rows": len(df), "cols": len(df.columns),
        "columns": list(df.columns),
    })

# ─── TRAINING ────────────────────────────────────────────────────────────────

def build_keras_model(input_dim, output_dim, layers, out_act, weight_init, use_bias, problem):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed. Run: pip install tensorflow")
    init_map = {
        "Xavier/Glorot": "glorot_uniform", "He Normal": "he_normal",
        "He Uniform": "he_uniform", "LeCun Normal": "lecun_normal",
        "Orthogonal": "orthogonal", "Random Normal": "random_normal",
    }
    kinit = init_map.get(weight_init, "glorot_uniform")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for l in layers:
        model.add(tf.keras.layers.Dense(l["n"], activation=l["a"].lower(),
                                        kernel_initializer=kinit, use_bias=use_bias))
        if l.get("d", 0) > 0:
            model.add(tf.keras.layers.Dropout(l["d"]))
    model.add(tf.keras.layers.Dense(output_dim, activation=out_act.lower(),
                                    kernel_initializer=kinit, use_bias=use_bias))

    if problem in ("Classification", "Computer Vision", "NLP"):
        loss   = "sparse_categorical_crossentropy" if output_dim > 2 else "binary_crossentropy"
        metric = ["accuracy"]
    else:
        loss, metric = "mse", ["mae"]

    opt_map = {"Adam": tf.keras.optimizers.Adam, "AdamW": tf.keras.optimizers.AdamW,
               "SGD": tf.keras.optimizers.SGD, "RMSprop": tf.keras.optimizers.RMSprop,
               "Adagrad": tf.keras.optimizers.Adagrad, "Nadam": tf.keras.optimizers.Nadam}
    return model, loss, metric, opt_map

def train_in_thread(sid, config):
    try:
        sess    = sessions[sid]
        df      = sess["df"].copy()
        target  = sess.get("target")
        problem = config.get("problemType", "Classification")
        hp      = config.get("hp", {})
        layers  = config.get("layers", [])
        model_type = config.get("modelType", "classical")
        algo    = config.get("algo", "Random Forest")
        out_cfg = config.get("outCfg", {})

        if not target or target not in df.columns:
            push(sid, "error", {"message": "No target column selected"}); emit_done(sid); return

        push(sid, "status", {"message": "Preparing data..."})

        # ── Split X / y ────────────────────────────────────────────────────
        feat_cols = [c for c in df.columns if c != target]
        X = df[feat_cols].select_dtypes(include=[np.number])
        y = df[target]

        if X.empty:
            push(sid, "error", {"message": "No numeric feature columns after preprocessing. Apply preprocessing first."}); emit_done(sid); return

        is_clf   = problem in ("Classification", "Computer Vision", "NLP", "Recommendation", "Anomaly Detection")
        is_clust = problem == "Clustering"
        v_split  = float(hp.get("vSplit", 0.2))

        if is_clust:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=v_split, random_state=42,
                stratify=y if is_clf and y.nunique() < 100 else None
            )

        push(sid, "status", {"message": f"Training on {len(X_train):,} samples, validating on {len(X_test):,}..."})
        history = []

        # ─── CLASSICAL ML ─────────────────────────────────────────────────
        if model_type == "classical":
            n_est = int(hp.get("epochs", 100))

            model_map = {
                # Classifiers
                "Random Forest":        RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=42),
                "Gradient Boosting":    GradientBoostingClassifier(n_estimators=n_est, random_state=42),
                "Logistic Regression":  LogisticRegression(max_iter=1000, n_jobs=-1),
                "SVM":                  SVC(kernel="rbf", probability=True),
                "KNN":                  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                "Decision Tree":        DecisionTreeClassifier(random_state=42),
                "Naive Bayes":          GaussianNB(),
                "AdaBoost":             AdaBoostClassifier(n_estimators=n_est, random_state=42),
                # Regressors
                "Ridge Regression":     Ridge(alpha=float(hp.get("l2", 1.0))),
                "Lasso":                Lasso(alpha=float(hp.get("l2", 1.0)), max_iter=2000),
                "ElasticNet":           ElasticNet(max_iter=2000),
                "Linear Regression":    LinearRegression(n_jobs=-1),
                "SVR":                  SVR(kernel="rbf"),
                # Clustering
                "K-Means":              KMeans(n_clusters=int(y.nunique() or 3), n_init=10, random_state=42),
                "DBSCAN":               DBSCAN(eps=0.5, min_samples=5),
                "Isolation Forest":     IsolationForest(n_estimators=n_est, random_state=42),
            }

            if HAS_XGB:
                model_map["XGBoost"] = xgb.XGBClassifier(n_estimators=n_est, n_jobs=-1,
                                                          eval_metric="logloss", random_state=42,
                                                          verbosity=0) if is_clf else \
                                       xgb.XGBRegressor(n_estimators=n_est, n_jobs=-1, random_state=42, verbosity=0)

            # pick regressor variant if regression task
            reg_variants = {"Random Forest": RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_est, random_state=42),
                            "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
                            "SVM": SVR(kernel="rbf"),
                            "AdaBoost": AdaBoostRegressor(n_estimators=n_est, random_state=42)}
            if not is_clf and not is_clust and algo in reg_variants:
                mdl = reg_variants[algo]
            else:
                mdl = model_map.get(algo, RandomForestClassifier(n_estimators=50, n_jobs=-1))

            # Simulate per-step progress for GB / XGB (those support staged predict)
            if hasattr(mdl, "n_estimators") and not is_clust:
                total_steps = min(int(hp.get("epochs", 100)), 200)
                mdl.set_params(n_estimators=total_steps)
                push(sid, "status", {"message": f"Training {algo} ({total_steps} estimators)..."})

                # Fit with staged reporting
                if hasattr(mdl, "staged_predict"):
                    mdl.fit(X_train, y_train)
                    for i, y_pred_staged in enumerate(mdl.staged_predict(X_test)):
                        if (i + 1) % max(1, total_steps // 20) == 0 or i == total_steps - 1:
                            step = i + 1
                            if is_clf:
                                acc = accuracy_score(y_test, y_pred_staged) * 100
                                history.append({"epoch": step, "trainLoss": round(1-acc/100, 4),
                                                "valLoss": round(1-acc/100+0.01, 4), "valAcc": round(acc, 2)})
                            else:
                                mse = mean_squared_error(y_test, y_pred_staged)
                                history.append({"epoch": step, "trainLoss": round(mse, 4), "valLoss": round(mse*1.05, 4)})
                            push(sid, "progress", history[-1])
                else:
                    mdl.fit(X_train, y_train)
                    push(sid, "progress", {"epoch": total_steps, "trainLoss": 0, "valLoss": 0})
            else:
                push(sid, "status", {"message": f"Fitting {algo}..."})
                mdl.fit(X_train if not is_clust else X, y_train if not is_clust else y)
                push(sid, "progress", {"epoch": 1, "trainLoss": 0, "valLoss": 0})

            # ── Metrics ────────────────────────────────────────────────────
            metrics = {}
            if is_clust:
                labels = mdl.labels_ if hasattr(mdl, "labels_") else mdl.fit_predict(X)
                try:   metrics["silhouette"] = round(float(silhouette_score(X, labels)), 4)
                except: pass
                metrics["n_clusters"] = int(len(np.unique(labels)))
            elif is_clf:
                y_pred = mdl.predict(X_test)
                y_pred_train = mdl.predict(X_train)
                metrics["trainAcc"] = round(accuracy_score(y_train, y_pred_train) * 100, 2)
                metrics["valAcc"]   = round(accuracy_score(y_test, y_pred) * 100, 2)
                metrics["f1"]       = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["precision"]= round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["recall"]   = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                try:
                    metrics["classReport"] = classification_report(y_test, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test, y_pred)
                    metrics["confusionMatrix"] = cm.tolist()
                except: pass
                metrics["trainLoss"] = round(1 - metrics["trainAcc"] / 100, 4)
                metrics["valLoss"]   = round(1 - metrics["valAcc"] / 100, 4)
            else:
                y_pred = mdl.predict(X_test)
                y_pred_train = mdl.predict(X_train)
                metrics["trainLoss"] = round(float(mean_squared_error(y_train, y_pred_train)), 6)
                metrics["valLoss"]   = round(float(mean_squared_error(y_test, y_pred)), 6)
                metrics["r2"]        = round(float(r2_score(y_test, y_pred)), 4)
                metrics["mae"]       = round(float(mean_absolute_error(y_test, y_pred)), 6)
                metrics["rmse"]      = round(float(np.sqrt(metrics["valLoss"])), 6)

            # save
            model_path = os.path.join(MODEL_DIR, f"{sid}_model.pkl")
            with open(model_path, "wb") as fp:
                pickle.dump({"model": mdl, "feature_cols": list(X.columns),
                             "target": target, "algo": algo}, fp)
            sessions[sid]["model"]      = mdl
            sessions[sid]["metrics"]    = metrics
            sessions[sid]["history"]    = history
            sessions[sid]["model_path"] = model_path

            push(sid, "metrics", metrics)
            emit_done(sid)

        # ─── DEEP LEARNING (Keras) ─────────────────────────────────────────
        else:
            if not HAS_TF:
                push(sid, "error", {"message": "TensorFlow not installed. Run: pip install tensorflow"}); emit_done(sid); return

            # encode y
            y_train_enc, y_test_enc = y_train.copy(), y_test.copy()
            le = None
            if is_clf:
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train.astype(str))
                y_test_enc  = le.transform(y_test.astype(str))

            input_dim  = X_train.shape[1]
            output_dim = int(y_train_enc.max() + 1) if is_clf else 1
            epochs     = int(hp.get("epochs", 50))
            bs         = int(hp.get("bs", 32))
            lr         = float(hp.get("lr", 0.001))
            l2_val     = float(hp.get("l2", 0.0))

            # build model
            mdl, loss, metric, opt_map = build_keras_model(
                input_dim, output_dim, layers,
                out_cfg.get("activation", "Softmax"),
                out_cfg.get("weightInit", "Xavier/Glorot"),
                out_cfg.get("useBias", True), problem
            )

            reg = tf.keras.regularizers.l2(l2_val) if l2_val > 0 else None
            for layer in mdl.layers:
                if isinstance(layer, tf.keras.layers.Dense) and reg:
                    layer.kernel_regularizer = reg

            opt_cls = opt_map.get(hp.get("optimizer", "Adam"), tf.keras.optimizers.Adam)
            mdl.compile(optimizer=opt_cls(learning_rate=lr), loss=loss, metrics=metric)

            # callbacks
            cbs = []
            if hp.get("earlyStopping"):
                cbs.append(tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=int(hp.get("patience", 10)),
                    restore_best_weights=True, verbose=0))

            # LR scheduler
            sch = hp.get("scheduler", "None")
            if sch == "ReduceOnPlateau":
                cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0))
            elif sch == "CosineAnnealing":
                def cosine_decay(epoch):
                    return lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
                cbs.append(tf.keras.callbacks.LearningRateScheduler(cosine_decay, verbose=0))

            # SSE progress callback
            class SSECallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    entry = {"epoch": epoch + 1,
                             "trainLoss": round(float(logs.get("loss", 0)), 4),
                             "valLoss":   round(float(logs.get("val_loss", 0)), 4)}
                    if is_clf:
                        k = [k for k in logs if "accuracy" in k and "val" not in k]
                        vk= [k for k in logs if "val_accuracy" in k]
                        if k:  entry["trainAcc"] = round(float(logs[k[0]]) * 100, 2)
                        if vk: entry["valAcc"]   = round(float(logs[vk[0]]) * 100, 2)
                    history.append(entry)
                    push(sid, "progress", entry)

            cbs.append(SSECallback())

            y_fit = y_train_enc if is_clf else y_train.values.astype(np.float32)
            y_val = y_test_enc  if is_clf else y_test.values.astype(np.float32)

            mdl.fit(X_train.values.astype(np.float32), y_fit,
                    validation_data=(X_test.values.astype(np.float32), y_val),
                    epochs=epochs, batch_size=bs, callbacks=cbs, verbose=0)

            # ── Metrics ────────────────────────────────────────────────────
            metrics = {}
            y_pred_raw = mdl.predict(X_test.values.astype(np.float32), verbose=0)
            y_pred_train_raw = mdl.predict(X_train.values.astype(np.float32), verbose=0)

            if is_clf:
                y_pred       = np.argmax(y_pred_raw, axis=1) if output_dim > 2 else (y_pred_raw.flatten() > 0.5).astype(int)
                y_pred_train = np.argmax(y_pred_train_raw, axis=1) if output_dim > 2 else (y_pred_train_raw.flatten() > 0.5).astype(int)
                metrics["trainAcc"] = round(accuracy_score(y_train_enc, y_pred_train) * 100, 2)
                metrics["valAcc"]   = round(accuracy_score(y_test_enc,  y_pred) * 100, 2)
                metrics["f1"]       = round(float(f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)), 4)
                metrics["precision"]= round(float(precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)), 4)
                metrics["recall"]   = round(float(recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)), 4)
                try:
                    metrics["confusionMatrix"] = confusion_matrix(y_test_enc, y_pred).tolist()
                    metrics["classReport"] = classification_report(y_test_enc, y_pred, output_dict=True)
                except: pass
            else:
                y_pred = y_pred_raw.flatten()
                y_pred_train = y_pred_train_raw.flatten()
                metrics["r2"]        = round(float(r2_score(y_test, y_pred)), 4)
                metrics["mae"]       = round(float(mean_absolute_error(y_test, y_pred)), 6)
                metrics["rmse"]      = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6)

            metrics["trainLoss"] = round(float(history[-1]["trainLoss"]) if history else 0, 4)
            metrics["valLoss"]   = round(float(history[-1]["valLoss"])   if history else 0, 4)
            if history and "trainAcc" in history[-1]: metrics["trainAcc"] = history[-1]["trainAcc"]
            if history and "valAcc"   in history[-1]: metrics["valAcc"]   = history[-1]["valAcc"]

            # save model
            model_path = os.path.join(MODEL_DIR, f"{sid}_model.h5")
            try:
                mdl.save(model_path)
            except Exception:
                model_path = os.path.join(MODEL_DIR, f"{sid}_model.pkl")
                with open(model_path, "wb") as fp:
                    pickle.dump({"model": mdl, "feature_cols": list(X.columns),
                                 "target": target, "le": le}, fp)

            sessions[sid]["model"]      = mdl
            sessions[sid]["metrics"]    = metrics
            sessions[sid]["history"]    = history
            sessions[sid]["model_path"] = model_path

            push(sid, "metrics", metrics)
            emit_done(sid)

    except Exception as e:
        push(sid, "error", {"message": str(e), "trace": traceback.format_exc()})
        emit_done(sid)

@app.route("/api/train/stream", methods=["POST"])
def train_stream_start():
    body = request.json
    sid  = body.get("sid")
    if not sid_ok(sid):
        return jsonify({"error": "Session not found"}), 404

    q = queue.Queue()
    train_queues[sid] = q
    sessions[sid]["history"] = []

    t = threading.Thread(target=train_in_thread, args=(sid, body), daemon=True)
    t.start()

    def sse_gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"
        yield "data: {\"event\":\"done\",\"data\":{}}\n\n"

    return Response(sse_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─── MODEL EXPORT ─────────────────────────────────────────────────────────────

@app.route("/api/model/export", methods=["GET"])
def model_export():
    sid = request.args.get("sid")
    if not sid_ok(sid) or not sessions[sid].get("model_path"):
        return jsonify({"error": "No trained model"}), 404
    path = sessions[sid]["model_path"]
    if not os.path.exists(path):
        return jsonify({"error": "Model file missing"}), 404
    return send_file(path, as_attachment=True)

@app.route("/api/model/config", methods=["GET"])
def model_config():
    sid = request.args.get("sid")
    if not sid_ok(sid):
        return jsonify({"error": "No session"}), 404
    sess = sessions[sid]
    cfg = {
        "exportedAt": datetime.now().isoformat(),
        "target": sess.get("target"),
        "metrics": sess.get("metrics", {}),
        "historyLen": len(sess.get("history", [])),
        "prepLog": sess.get("prep_log", []),
    }
    return jsonify(cfg)

@app.route("/api/history/csv", methods=["GET"])
def history_csv():
    sid = request.args.get("sid")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    hist = sessions[sid].get("history", [])
    if not hist: return jsonify({"error": "No history"}), 404
    df = pd.DataFrame(hist)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="training_history.csv")

@app.route("/api/dataset/export", methods=["GET"])
def dataset_export():
    sid = request.args.get("sid")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    df = sessions[sid]["df"]
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="processed_dataset.csv")

# ─── HEALTH / INFO ────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "tensorflow": HAS_TF,
        "xgboost": HAS_XGB,
        "tf_version": tf.__version__ if HAS_TF else None,
        "sessions": len(sessions),
    })

@app.route("/api/session/info", methods=["GET"])
def session_info():
    sid = request.args.get("sid")
    if not sid_ok(sid): return jsonify({"error": "No session"}), 404
    sess = sessions[sid]
    df   = sess["df"]
    return jsonify({
        "rows": len(df), "cols": len(df.columns),
        "target": sess.get("target"),
        "columns": list(df.columns),
        "metrics": sess.get("metrics", {}),
        "hasPrepLog": bool(sess.get("prep_log")),
        "hasModel": sess.get("model") is not None,
    })

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ML Training Assistant — Backend")
    print("="*55)
    print(f"  TensorFlow : {'✓  ' + tf.__version__ if HAS_TF else '✗  not installed (pip install tensorflow)'}")
    print(f"  XGBoost    : {'✓' if HAS_XGB else '✗  not installed (pip install xgboost)'}")
    print(f"  Uploads    : {os.path.abspath(UPLOAD_DIR)}")
    print(f"  Models     : {os.path.abspath(MODEL_DIR)}")
    print("="*55)
    print("  Running at http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
