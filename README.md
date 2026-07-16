# disaster-tweets-nlp

Production-ready ML pipeline for identifying disaster-related tweets and enabling faster emergency response


-------------------------------------------------------------------------------------------------------------------
                        disaster-tweets-nlp/
                        │
                        ├── README.md
                        ├── pyproject.toml
                        ├── requirements-dev.txt
                        ├── Makefile
                        ├── .gitignore
                        ├── .env.example
                        ├── LICENSE
                        │
                        ├── configs/
                        │   ├── baseline.yaml
                        │   ├── distilbert.yaml
                        │   └── roberta.yaml
                        │
                        ├── data/
                        │   ├── README.md
                        │   ├── raw/
                        │   │   └── .gitkeep
                        │   ├── interim/
                        │   │   └── .gitkeep
                        │   └── processed/
                        │       └── .gitkeep
                        │
                        ├── notebooks/
                        │   ├── 01_data_audit.ipynb
                        │   ├── 02_eda_and_sentiment.ipynb
                        │   ├── 03_classical_baselines.ipynb
                        │   ├── 04_keras_transformers.ipynb
                        │   └── 05_error_analysis.ipynb
                        │
                        ├── src/
                        │   └── disaster_tweets/
                        │       ├── __init__.py
                        │       ├── config.py
                        │       ├── data.py
                        │       ├── validation.py
                        │       ├── preprocessing.py
                        │       │
                        │       ├── features/
                        │       │   ├── lexical.py
                        │       │   ├── sentiment.py
                        │       │   └── metadata.py
                        │       │
                        │       ├── models/
                        │       │   ├── baseline.py
                        │       │   ├── keras_classifier.py
                        │       │   ├── evaluation.py
                        │       │   └── registry.py
                        │       │
                        │       ├── inference/
                        │       │   ├── predictor.py
                        │       │   └── schemas.py
                        │       │
                        │       └── visualization/
                        │           └── plots.py
                        │
                        ├── scripts/
                        │   ├── download_data.py
                        │   ├── audit_data.py
                        │   ├── train_baseline.py
                        │   ├── train_transformer.py
                        │   ├── evaluate_model.py
                        │   ├── generate_submission.py
                        │   └── predict.py
                        │
                        ├── app/
                        │   ├── streamlit_app.py
                        │   └── components.py
                        │
                        ├── artifacts/
                        │   └── README.md
                        │
                        ├── reports/
                        │   ├── figures/
                        │   ├── metrics/
                        │   ├── data_card.md
                        │   └── model_card.md
                        │
                        ├── tests/
                        │   ├── test_data_contract.py
                        │   ├── test_preprocessing.py
                        │   ├── test_sentiment.py
                        │   ├── test_inference.py
                        │   └── fixtures/
                        │
                        └── .github/
                            └── workflows/
                                └── ci.yml

--------------------------------------------------------------------------------------------------------------------
