# Olist data policy

The project uses the **Brazilian E-Commerce Public Dataset by Olist** for local,
retrospective analysis. The canonical source is the [Olist Kaggle dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Kaggle metadata reports **CC BY-NC-SA 4.0**. The dataset is anonymised commercial
data supplied by Olist and covers historical orders from 2016–2018.

The raw archive, extracted CSVs, processed tables, model artifacts, and generated
reports are intentionally ignored by Git. Download them locally with:

```bash
PYTHONPATH=src python scripts/download_olist.py
```

The downloader records the source URL, license, and verified archive SHA-256 in
`data/raw/manifest.json`. Do not commit or redistribute the raw data. Do not
publicly host derived model artifacts or use the dataset commercially without
separate permission/legal review. The repository code remains a portfolio
example and does not claim professional ML experience or production performance.
