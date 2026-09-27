.PHONY: install train test up down
install:
	python -m pip install -r requirements-dev.txt
train:
	python scripts/train_model.py
test:
	pytest -q
up:
	docker compose up --build -d
down:
	docker compose down
