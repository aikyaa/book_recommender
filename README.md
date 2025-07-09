# Semantic Book Recommender with LLMs

## Files :
* `datasets.ipynb`: Cleared out books with missing fields and descriptions which are not very meaningful.
* `vector_search.ipynb`: Created a vector database to find similar books.
* `categorisation.ipynb`: Classified books into fiction and non-fiction(and some other useful categories) using zero-shot classification.
* `sentimental_analysis.ipynb`: Extracted the emotion scores from descriptions.
* `dashboard.py`: Created a web application that allows users to get recommendations, select a category and sort books by tone.

Created using Python 3.13.5.
Kaggle dataset: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata

`requirements.txt` contains all the project dependencies. Run:
```shell
pip install -r requirements.txt
```

