from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB, enforced by Werkzeug before the body is fully read


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size is 15MB."}), 413

# Load the saved model
model_path = "best_rf_model (1).pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# --- Graph-aware ensemble (GAN-augmented GBDT + GraphSAGE + HGT + anomaly
# detectors, stacked and calibrated) -- loaded in ADDITION to the
# RandomForest above, not a replacement. It was trained on this project's
# new relational schema, which has a different feature layout than the flat
# `features` array the endpoints below already accept, so it cannot
# reinterpret that legacy payload; it instead serves a new endpoint
# (`/predict/v2`) that scores transactions already present in its graph
# snapshot. See fraud_detection/fraud_detection/serving/ensemble_predictor.py
# for why GraphSAGE/HGT can't score a truly novel, never-seen transaction.
ensemble_predictor = None
try:
    from fraud_detection.serving.ensemble_predictor import FraudEnsemblePredictor

    ensemble_predictor = FraudEnsemblePredictor.load(
        os.getenv("FRAUD_MODEL_DIR", "../fraud_detection/artifacts/ensemble/latest")
    )
except Exception as e:
    print(f"Graph-aware ensemble not loaded ({e}); /predict/v2 will return 503 until it is trained.")

import anthropic

from fraud_detection.statement_analysis.parsers.csv_parser import StatementParseError
from fraud_detection.statement_analysis.statement_analyzer import analyze_statement

PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "")
MONO_SECRET_KEY = os.getenv("MONO_SECRET_KEY", "")
MONO_BASE_URL = "https://api.withmono.com/v2"

@app.route('/')
def home():
    return "Welcome to the ShieldPay API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/v2', methods=['POST'])
def predict_v2():
    """Graph-aware ensemble prediction (XGBoost + GraphSAGE + HGT + anomaly
    detectors, stacked and calibrated). Only scores transactions already
    present in the loaded graph snapshot -- see the module-level comment
    above on why, and ensemble_predictor.py for the full explanation.

    Expects: { "transaction_id": "txn_0000123" }
    Returns: { "transaction_id": ..., "fraud_probability": 0.87, "prediction": 1 }
    """
    if ensemble_predictor is None:
        return jsonify({"error": "Graph-aware ensemble model not loaded on this server"}), 503

    data = request.get_json()
    if not data or not data.get("transaction_id"):
        return jsonify({"error": "transaction_id is required"}), 400

    transaction_id = data["transaction_id"]
    proba = ensemble_predictor.predict_proba_by_transaction_id(transaction_id)
    if proba is None:
        return jsonify({"error": f"transaction_id '{transaction_id}' not found in the current graph snapshot"}), 404

    return jsonify({
        "transaction_id": transaction_id,
        "fraud_probability": proba,
        "prediction": int(proba >= 0.5),
    })

ALLOWED_STATEMENT_EXTENSIONS = {".csv", ".pdf"}

@app.route('/statement/analyze', methods=['POST'])
def statement_analyze():
    """Uploads a bank statement (CSV or PDF) and returns fresh, per-statement
    unsupervised fraud-pattern analysis -- NOT the trained graph ensemble
    above, which was trained on synthetic data with entity IDs a real
    statement doesn't have. See fraud_detection/fraud_detection/
    statement_analysis/ for the full pipeline (rule-based pattern detectors
    + IsolationForest/autoencoder anomaly scoring, both fit fresh on the
    uploaded statement).

    Expects: multipart/form-data with a `file` field (.csv or .pdf).
    Returns: { data_quality, summary, pattern_findings, flagged_transactions, ... }
    """
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_STATEMENT_EXTENSIONS:
        return jsonify({"error": "Only .csv and .pdf files are supported"}), 400

    file_bytes = uploaded.read()
    if ext == ".pdf" and not file_bytes.startswith(b"%PDF-"):
        return jsonify({"error": "File does not appear to be a valid PDF"}), 400

    try:
        report = analyze_statement(file_bytes, filename)
        return jsonify(report)
    except StatementParseError as e:
        return jsonify({"error": str(e)}), 400
    except NotImplementedError as e:
        return jsonify({"error": str(e)}), 501
    except anthropic.RateLimitError:
        return jsonify({"error": "PDF extraction is rate-limited right now. Please try again shortly."}), 503
    except (anthropic.APITimeoutError, anthropic.APIConnectionError):
        return jsonify({"error": "Could not reach the PDF extraction service. Please try again."}), 504
    except anthropic.AuthenticationError:
        return jsonify({"error": "PDF extraction is not configured on this server (invalid or missing API key)."}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error analyzing statement: {e}"}), 500

@app.route('/verify-account', methods=['POST'])
def verify_account():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    account_number = data.get("account_number", "").strip()
    bank_code = data.get("bank_code", "").strip()

    if not account_number or not bank_code:
        return jsonify({"error": "account_number and bank_code are required"}), 400

    if not PAYSTACK_SECRET_KEY:
        return jsonify({"error": "Paystack secret key not configured on server"}), 500

    try:
        response = requests.get(
            f"https://api.paystack.co/bank/resolve?account_number={account_number}&bank_code={bank_code}",
            headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
            timeout=10
        )
        result = response.json()
        if response.status_code == 200 and result.get("status"):
            return jsonify({
                "status": True,
                "account_name": result["data"]["account_name"],
                "account_number": result["data"]["account_number"]
            })
        else:
            return jsonify({"status": False, "error": result.get("message", "Could not resolve account")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Mono Connect API endpoints ──────────────────────────────────────────────

@app.route('/mono/exchange', methods=['POST'])
def mono_exchange():
    """Exchange the auth code from the Connect widget for a permanent account ID."""
    data = request.get_json()
    if not data or not data.get("code"):
        return jsonify({"error": "code is required"}), 400

    if not MONO_SECRET_KEY:
        return jsonify({"error": "Mono secret key not configured on server"}), 500

    try:
        response = requests.post(
            f"{MONO_BASE_URL}/accounts/auth",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "mono-sec-key": MONO_SECRET_KEY,
            },
            json={"code": data["code"]},
            timeout=30,
        )
        result = response.json()
        if response.status_code == 200:
            return jsonify({"id": result.get("id")})
        else:
            return jsonify({"error": result.get("message", "Failed to exchange token")}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mono/account/<account_id>', methods=['GET'])
def mono_account(account_id):
    """Fetch connected account details (balance, institution, meta)."""
    if not MONO_SECRET_KEY:
        return jsonify({"error": "Mono secret key not configured on server"}), 500

    try:
        response = requests.get(
            f"{MONO_BASE_URL}/accounts/{account_id}",
            headers={
                "Accept": "application/json",
                "mono-sec-key": MONO_SECRET_KEY,
            },
            timeout=30,
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mono/transactions/<account_id>', methods=['GET'])
def mono_transactions(account_id):
    """Fetch transactions for a connected account, optionally filtered by date range."""
    if not MONO_SECRET_KEY:
        return jsonify({"error": "Mono secret key not configured on server"}), 500

    start = request.args.get("start")
    end = request.args.get("end")
    page = request.args.get("page", "1")

    params = {"paginate": "false"}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if page:
        params["page"] = page

    try:
        response = requests.get(
            f"{MONO_BASE_URL}/accounts/{account_id}/transactions",
            headers={
                "Accept": "application/json",
                "mono-sec-key": MONO_SECRET_KEY,
            },
            params=params,
            timeout=60,
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mono/batch-predict', methods=['POST'])
def mono_batch_predict():
    """Run the fraud detection model across a batch of transactions.

    Expects: { "transactions": [ { "amount": 5000, "type": "debit", ... }, ... ] }
    Returns: { "results": [ { "index": 0, "prediction": 0 }, ... ], "fraud_count": 1 }
    """
    data = request.get_json()
    if not data or not data.get("transactions"):
        return jsonify({"error": "transactions array is required"}), 400

    transactions = data["transactions"]
    results = []
    fraud_count = 0

    for i, tx in enumerate(transactions):
        try:
            amount = float(tx.get("amount", 0))
            tx_type = 1 if str(tx.get("type", "")).lower() == "credit" else 0
            features = np.array([[amount, tx_type]])
            prediction = model.predict(features)
            pred_val = int(prediction[0])
            if pred_val == 1:
                fraud_count += 1
            results.append({"index": i, "prediction": pred_val})
        except Exception:
            results.append({"index": i, "prediction": 0, "error": "prediction failed"})

    return jsonify({"results": results, "fraud_count": fraud_count})


if __name__ == '__main__':
    app.run(debug=True)
