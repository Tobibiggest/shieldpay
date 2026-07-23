# 🌟ShieldPayAI — Graph-Native, Multi-Domain Fraud Detection Platform

A fraud detection system built around one idea: coordinated fraud is a **network** problem, not a per-row problem. Mule networks, layering chains, and smurfing rings are defined by relationships between accounts — shared devices, shared IP addresses, funds passed hop-to-hop through intermediaries — not by any single transaction's own features. A model that scores each transaction independently cannot see that structure no matter how well-tuned it is. This project makes the transaction graph a first-class model input, evaluates it against a strong tabular baseline under a benchmark specifically designed so the comparison is meaningful (fraud is feature-camouflaged, detectable only via graph structure), and generalizes the architecture to more than one fraud domain.

A full write-up of the methodology and results lives in [`research_paper/`](https://www.researchgate.net/publication/410744657_A_Generalized_Heterogeneous_Graph_Neural_Network_Architecture_for_Multi-Domain_Financial_Fraud_Detection_From_Payment_Fraud_to_Anti-Money-Laundering_Monitoring/).

## Results at a glance
<img width="1920" height="910" alt="Screenshot (645)" src="https://github.com/user-attachments/assets/de0b0138-e016-4e18-8edb-f0bce3380132" />
<img width="1920" height="872" alt="Screenshot (649)" src="https://github.com/user-attachments/assets/c239cf2e-c03a-4991-97a2-4e73fd2797cb" />
<img width="1920" height="897" alt="Screenshot (648)" src="https://github.com/user-attachments/assets/9685e983-99f0-41d1-a6ba-e5c6b0cd8340" />
<img width="1920" height="888" alt="Screenshot (650)" src="https://github.com/user-attachments/assets/6f3d970e-fbe1-4bc1-8ea4-940b8d9926c6" />
<img width="1920" height="905" alt="Screenshot (651)" src="https://github.com/user-attachments/assets/261a8f09-b2bc-4ae3-ac1b-8a4d28fa89ef" />

<img width="1920" height="903" alt="Screenshot (652)" src="https://github.com/user-attachments/assets/ddd357de-d4ac-49e6-8e4b-d3f351622f92" />

<img width="1920" height="909" alt="Screenshot (654)" src="https://github.com/user-attachments/assets/9e3c7eda-ff4d-4670-9d03-9a8e985678b0" />


**Payments** (synthetic benchmark, ~3% fraud, temporal split):

| Model | AUPRC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|
| XGBoost (tabular baseline) | 0.371 | 0.659 | 0.310 |
| GraphSAGE | 0.733 | 0.974 | 0.575 |
| GAT | 0.678 | 0.951 | 0.563 |
| R-GCN | 0.537 | 0.919 | 0.368 |
| Hypergraph Conv. | 0.670 | 0.891 | 0.626 |
| **Heterogeneous Graph Transformer (HGT)** | **0.797** | **0.983** | **0.701** |
| Calibrated ensemble | 0.721 | 0.951 | 0.718 |

**Anti-Money-Laundering** (independent domain, layering chains + smurfing, identical model code):

| Model | AUPRC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|
| XGBoost (tabular baseline) | 0.131 | 0.570 | 0.103 |
| GraphSAGE | 0.247 | 0.895 | 0.184 |
| **HGT** | **0.614** | **0.970** | **0.667** |

HGT is the strongest model in both domains — a 115% relative AUPRC improvement over XGBoost on payments, and 369% on AML — using **the same, unmodified model code** in both cases. Only the domain schema (which entities exist, how they relate) changes between them. See [`fraud_detection/fraud_detection/domains/README.md`](fraud_detection/fraud_detection/domains/README.md) for how that generalization works and how to add a new domain.

## What's in this repository

| Path | What it is |
|---|---|
| [`fraud_detection/`](fraud_detection/) | The active ML platform: dataset-agnostic schema/adapter layer, conditional WGAN-GP, GraphSAGE/GAT/HGT/R-GCN/hypergraph convolution, XGBoost baseline, autoencoder + IsolationForest anomaly detectors, a calibrated stacking ensemble, the generalized multi-domain graph schema, the AML domain, and an unsupervised bank-statement analysis module. |
| [`AI_model_server_Flask/`](AI_model_server_Flask/) | Flask API serving both the legacy tabular model and the new graph-ensemble/statement-analysis endpoints. |
| [`fraudAI_Frontend_React/`](fraudAI_Frontend_React/) | The main React app (dashboard, transactions, statement upload). |
| [`shieldpay-landing page/`](shieldpay-landing%20page/) | A separate marketing/landing site, not wired to the ML backend. |
| [`AI_model_Py_Scripts/`](AI_model_Py_Scripts/) | The original notebook-based GAN + Random Forest prototype, superseded by `fraud_detection/`. |
| [`research_paper/`](research_paper/) | Written summary of the architecture, methodology, and results. |
| [`SystemDesignDiagrams/`](SystemDesignDiagrams/) | Architecture diagrams referenced below. |

## System architecture

### Product flow

![System Design](SystemDesignDiagrams/SystemDesign.png)

Authentication (Google Sign-In → Firebase) assigns/selects a UPI ID, the user enters transaction details, the transaction process calls the fraud detection service, and the result gates transaction execution before it's recorded to transaction history.

![System Architecture and Workflow Overview](SystemDesignDiagrams/WorkFlowDiagram.png)

### ML pipeline

The "AI Fraud Detection" step above is backed by the [`fraud_detection/`](fraud_detection/) package:

```
Dataset adapter (schema.py / domain_schema.py)
        │  canonical dataframe: entities, features, timestamps, label
        ▼
Graph construction (data/graph/) ──► homogeneous / heterogeneous / hypergraph
        │                              transaction graphs (PyG HeteroData)
        ▼
Model layer (models/)
   ├─ XGBoost (tabular baseline)
   ├─ Conditional WGAN-GP (minority-class augmentation)
   ├─ GraphSAGE / GAT (homogeneous graph)
   ├─ HGT / R-GCN (heterogeneous graph)
   ├─ Hypergraph convolution (n-ary collusion)
   └─ Autoencoder / IsolationForest (unsupervised anomaly scoring)
        │
        ▼
Calibrated stacking ensemble (models/ensemble/)
        │
        ▼
Serving (AI_model_server_Flask/app.py: /predict/v2)
```

The same pipeline, unmodified, runs against either the payments schema or the AML schema (`fraud_detection/fraud_detection/domains/aml/`) — only the schema/adapter/generator at the top changes. A separate, unsupervised path (`statement_analysis/`) analyzes real, unlabeled bank statements without going through this trained pipeline at all, since a model trained on synthetic data has no honest basis to score real data it's never seen labels for.

## Getting started

### ML platform (`fraud_detection/`)

```bash
cd fraud_detection
pip install -r requirements.txt
pip install -e .
```

PyTorch Geometric's accelerated extensions (`torch_scatter`, `torch_sparse`, ...) need a separate, OS/CUDA-specific install step — see [`fraud_detection/docs/INSTALL_PYG.md`](fraud_detection/docs/INSTALL_PYG.md). Everything runs without them, just slower.

Generate data and train the payments comparison table:

```bash
python -m fraud_detection.cli generate-data --output data/synthetic_relational.csv
python -m fraud_detection.training.train_hgnn --data data/synthetic_relational.csv
```

Or the AML domain:

```python
from fraud_detection.domains.aml.aml_generator import generate_aml_transaction_dataset
generate_aml_transaction_dataset(output_csv="data/aml_transactions.csv")
```

```bash
python -m fraud_detection.training.train_aml --data data/aml_transactions.csv
```

Run the test suite (73 tests as of this writing):

```bash
pytest fraud_detection/tests/
```

See [`fraud_detection/docs/DATASETS.md`](fraud_detection/docs/DATASETS.md) for the full dataset catalogue (including the IEEE-CIS portability adapter) and [`fraud_detection/fraud_detection/domains/README.md`](fraud_detection/fraud_detection/domains/README.md) for the recipe to add a new fraud domain.

### Backend API (`AI_model_server_Flask/`)

```bash
cd AI_model_server_Flask
pip install -r ../fraud_detection/requirements.txt
python app.py
```

Copy `.env.example` (if present) or create `.env` with your own `PAYSTACK_SECRET_KEY`/`MONO_SECRET_KEY`/`ANTHROPIC_API_KEY` — never commit this file. Set `FRAUD_MODEL_DIR` to point at a trained ensemble bundle (produced by `fraud_detection.training.train_ensemble`) to enable `/predict/v2`.

Current endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Health check |
| `POST /predict` | Legacy tabular prediction (flat `features` array, the original Random Forest) |
| `POST /predict/v2` | Graph-ensemble prediction for a `transaction_id` already in the loaded graph snapshot |
| `POST /statement/analyze` | Upload a CSV/PDF bank statement for unsupervised fraud-pattern analysis |
| `POST /verify-account` | Paystack account-name resolution |
| `POST /mono/exchange`, `GET /mono/account/<id>`, `GET /mono/transactions/<id>`, `POST /mono/batch-predict` | Mono Open Banking integration |

### Frontend (`fraudAI_Frontend_React/`)

```bash
cd fraudAI_Frontend_React
npm install
npm run dev
```

Opens at `http://localhost:5173` by default (Vite). Needs its own `.env` with Firebase config — see the `VITE_FIREBASE_*` keys referenced in `src/components/logic/firebase.js`.
