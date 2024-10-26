import pandas as pd
from sentence_transformers import SentenceTransformer


class SentenceEncoder:
    def __init__(self, embeddings_folder: str, model_cache_folder: str):
        self.model = SentenceTransformer(
            "sentence-transformers/sentence-t5-base", device="cuda", cache_folder=model_cache_folder
        )
        self.embeddings_folder = embeddings_folder

    def encode(self, items: pd.DataFrame):
        sentences = items.apply(lambda x: self.__construct_sentence(x), axis=1).tolist()
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        return embeddings

    def __construct_sentence(self, row):
        def is_valid_string(value):
            return not pd.isna(value) and len(value) > 0 and not value.isspace()

        def is_valid_list(value):
            return len(value) > 0

        def is_valid_float(value):
            return not pd.isna(value)

        # sentence format described here # https://openreview.net/forum?id=BJ0fQUU32w&noteId=VHFJP5cnpw
        pieces = []
        if is_valid_string(row["asin"]):
            pieces.append(f"Item ID: {row['asin']}")
        if is_valid_string(row["title"]):
            pieces.append(f"Title: {row['title']}")
        if is_valid_string(row["brand"]):
            pieces.append(f"Brand: {row['brand']}")
        if is_valid_list(row["categories"]):
            pieces.append(f"Categories: {row['categories']}")
        if is_valid_string(row["description"]):
            pieces.append(f"Description: {row['description']}")
        if is_valid_float(row["price"]):
            pieces.append(f"Price: {row['price']}")
        return ". ".join(pieces)
