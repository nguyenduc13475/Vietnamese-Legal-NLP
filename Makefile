.PHONY: install check-models clean-docs auto-annotate run-pipeline run-api run-ui build-db train-ner train-srl train-seg eval-ner eval-srl eval-seg train-intent eval-intent clean generate-report

install:
	pip install -r requirements.txt

check-models:
	@if [ ! -d "models/ner" ]; then echo "Error: models/ner not found. Please download models from Google Drive link in README and extract to models/ folder."; exit 1; fi
	@echo "All model directories verified.

clean-docs:
	@echo "Cleaning and Auto-filling raw contracts via Gemini..."
	python scripts/clean_raw_docs.py --input data/raw --output data/processed

auto-annotate:
	python scripts/auto_annotate.py

run-pipeline:
	python main.py --input input/raw_contracts.txt --output_dir output

run-api:
	uvicorn api.main:app --reload --port 8000

run-ui:
	streamlit run ui/app.py

build-db:
	python scripts/build_vector_store.py --input data/processed/

train-ner:
	python scripts/train_ner.py --epochs 30

train-srl:
	python scripts/train_srl.py --epochs 25 --batch_size 16

train-seg:
	python scripts/train_segmenter.py

eval-ner:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python scripts/evaluate_ner.py

eval-srl:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" && python scripts/evaluate_srl.py

eval-seg:
	export PYTHONPATH="/env/python:" && export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1 && python scripts/evaluate_segmenter.py

train-intent:
	python scripts/train_intent.py --epochs 30

eval-intent:
	python scripts/evaluate_intent.py

clean:
	rm -rf output/*
	rm -rf report/*.txt
	find . -type d -name "__pycache__" -exec rm -rf {} +

generate-report:
	python scripts/generate_report.py