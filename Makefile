.PHONY: install setup-all run-pipeline run-api run-ui build-db train-ner eval-ner train-intent eval-intent eval-all clean generate-report run-all-evals test clean-docs

install:
	pip install -r requirements.txt

setup-all:
	@echo "1. Cleaning and Auto-filling raw contracts via Gemini..."
	python scripts/clean_raw_docs.py --input data/raw --output data/processed
	@echo "2. Auto-annotating dataset for Intent & NER using Gemini..."
	python scripts/auto_annotate.py
	@echo "3. Training Intent Classifier..."
	python scripts/train_intent.py --model all
	@echo "4. Training NER Model..."
	python scripts/train_ner.py
	@echo "5. Building Vector Database for RAG..."
	python scripts/build_vector_store.py --input data/processed/
	@echo "Setup hoàn tất! Chạy 'make run-api' và 'make run-ui' để xem demo."

clean-docs:
	@echo "Cleaning and Auto-filling raw contracts via Gemini..."
	python scripts/clean_raw_docs.py --input data/raw --output data/processed

auto-annotate:
	python scripts/auto_annotate.py --mode generate

run-pipeline:
	python main.py --input data/processed/hop_dong_thue_nha.txt

run-api:
	uvicorn api.main:app --reload --port 8000

run-ui:
	streamlit run ui/app.py

build-db:
	python scripts/build_vector_store.py --input data/processed/

train-ner:
	python scripts/train_ner.py --epochs 30

train-srl:
	python scripts/train_srl.py

train-seg:
	python scripts/train_segmenter.py

train-all: train-ner train-srl train-seg train-intent
	@echo "All deep learning models updated."

eval-ner:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python scripts/evaluate_ner.py

eval-seg:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python scripts/evaluate_segmenter.py

train-intent:
	python scripts/train_intent.py --epochs 30

eval-intent:
	python scripts/evaluate_intent.py

eval-all: eval-ner eval-intent eval-seg
	@echo "========================================================="
	@echo "All evaluation reports (NER, Intent, Segmenter) have been updated in report/"
	@echo "========================================================="

clean:
	rm -rf output/*
	rm -rf report/*.txt
	find . -type d -name "__pycache__" -exec rm -rf {} +

generate-report:
	python scripts/generate_report.py

run-all-evals: train-intent train-ner eval-ner generate-report
	@echo "========================================================="
	@echo "Completed! The final report has been generated at report/FINAL_REPORT.md"
	@echo "========================================================="

test:
	pytest tests/ -v

test-seg:
	pytest tests/test_segmenter_regression.py -v

test-report:
	pytest tests/ -v > report/unit_test_results.txt