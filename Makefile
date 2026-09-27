.PHONY: install train train-olist download-olist score-olist test up down

install:
	python -m pip install -r requirements-dev.txt

train:
	python scripts/train_model.py

download-olist:
	PYTHONPATH=src python scripts/download_olist.py

train-olist:
	PYTHONPATH=src python scripts/train_olist_model.py

score-olist:
	PYTHONPATH=src python scripts/score_olist.py

test:
	pytest -q

up:
	docker compose up --build -d

down:
	docker compose down
