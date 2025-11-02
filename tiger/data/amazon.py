import gzip
import itertools
import os
from typing import Generator

import pandas as pd


class AmazonDataset:
    def __init__(
        self,
        category: str,
        year: int,
        min_reviews_per_user: int,
        max_sequence_length: int,
        data_dir: str,
    ):
        self.category = category
        self.year = year

        self.raw_items_path = f"{data_dir}/raw/{self.year}/meta_{self.category}.json.gz"
        self.interim_items_path = f"{data_dir}/interim/{self.year}/meta_{self.category}.parquet"
        self.raw_reviews_path = f"{data_dir}/raw/{self.year}/reviews_{self.category}.json.gz"
        self.interim_reviews_path = f"{data_dir}/interim/{self.year}/reviews_{self.category}.parquet"
        self.interim_sequences_path = f"{data_dir}/interim/{self.year}/sequences_{self.category}.parquet"

        self.min_reviews_per_user = min_reviews_per_user

        self.items = self.get_items()
        self.reviews = self.get_reviews()
        self.sequences = self.get_sequences(k=max_sequence_length)

    def download(self):
        raise NotImplementedError()

    def get_items(self):
        if self.__is_processed(self.interim_items_path):
            return self.__load_cache_data(self.interim_items_path)

        items = self.__to_dataframe(self.raw_items_path)
        items = self.__process_items(items)
        return items

    def get_reviews(self):
        if self.__is_processed(self.interim_reviews_path):
            return self.__load_cache_data(self.interim_reviews_path)

        valid_item_ids = self.items["asin"].tolist()
        reviews = self.__to_dataframe(self.raw_reviews_path)
        reviews = self.__process_reviews(reviews, valid_item_ids)
        return reviews

    def get_sequences(self, k: int = 20):
        if self.__is_processed(self.interim_sequences_path):
            return self.__load_cache_data(self.interim_sequences_path)

        sequences = self.__process_sequences(self.reviews, k)
        return sequences

    def calculate_stats(self):
        num_users = self.reviews["reviewerID"].unique().shape[0]
        num_items = self.reviews["asin"].unique().shape[0]
        num_reviews = self.reviews.shape[0]
        sparsity = 1 - num_reviews / (num_users * num_items)
        stats = {
            "dataset": f"{self.category}",
            "num_users": num_users,
            "num_items": num_items,
            "num_reviews": num_reviews,
            "avg_reviews_per_user": self.reviews.groupby("reviewerID")["reviewerID"].count().mean(),
            "median_reviews_per_user": self.reviews.groupby("reviewerID")["reviewerID"].count().median(),
            "mean_reviews_per_item": self.reviews.groupby("asin")["asin"].count().mean(),
            "sparsity": sparsity,
        }
        return stats

    def __is_processed(self, path: str):
        return os.path.exists(path)

    def __process_items(self, items: pd.DataFrame):
        def select_columns(df: pd.DataFrame, cols: list = None):
            if cols is None:
                cols = ["asin", "title", "brand", "categories", "description", "price"]
            return df[cols]

        def flatten_categories(df: pd.DataFrame) -> pd.DataFrame:
            df.loc[:, "categories"] = df.apply(
                lambda x: list(itertools.chain.from_iterable(x["categories"])), axis=1
            )
            return df

        def remove_duplicate_categories(df: pd.DataFrame) -> pd.DataFrame:
            df.loc[:, "categories"] = df.apply(lambda x: list(dict.fromkeys(x["categories"])), axis=1)
            return df

        items = items.pipe(select_columns).pipe(flatten_categories).pipe(remove_duplicate_categories)
        self.__cache_data(items, self.interim_items_path)
        return items

    def __process_reviews(self, reviews: pd.DataFrame, valid_item_ids: list) -> pd.DataFrame:
        def filter_valid_items(df: pd.DataFrame, valid_item_ids):
            df = df[df["asin"].isin(valid_item_ids)]
            return df

        def count_reviews_per_user(df: pd.DataFrame) -> pd.DataFrame:
            df.loc[:, "num_reviews"] = df.groupby("reviewerID")["reviewerID"].transform("size")
            return df

        def filter_less_than_k_reviews(df: pd.DataFrame, k: int, col: str) -> pd.DataFrame:
            df = df[df[col] >= k]
            return df

        def select_columns(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
            if cols is None:
                cols = ["reviewerID", "asin", "unixReviewTime"]
            df = df[cols]
            return df

        reviews = (
            reviews.pipe(filter_valid_items, valid_item_ids=valid_item_ids)
            .pipe(count_reviews_per_user)
            .pipe(filter_less_than_k_reviews, k=self.min_reviews_per_user, col="num_reviews")
            .pipe(select_columns)
        )
        self.__cache_data(reviews, self.interim_reviews_path)
        return reviews

    def __process_sequences(self, reviews: pd.DataFrame, k: int) -> pd.DataFrame:
        def filter_last_k_items(df: pd.DataFrame, k: int) -> pd.DataFrame:
            df.loc[:, "rank"] = (
                df.sort_values(["reviewerID", "unixReviewTime"], ascending=[True, False])
                .groupby("reviewerID")["unixReviewTime"]
                .rank(method="first", ascending=False)
            )

            df = df[df["rank"] <= k]
            df = df.drop(columns=["rank"])
            return df

        def make_sequences(df: pd.DataFrame) -> pd.DataFrame:
            df = (
                df.sort_values(["reviewerID", "unixReviewTime"], ascending=[True, False])
                .groupby("reviewerID")["asin"]
                .apply(list)
                .reset_index()
            )
            return df

        sequences = reviews.pipe(filter_last_k_items, k=k).pipe(make_sequences)
        self.__cache_data(sequences, self.interim_sequences_path)
        return sequences

    def __load_cache_data(self, path: str):
        return pd.read_parquet(path, engine="pyarrow")

    def __cache_data(self, df: pd.DataFrame, path: str):
        df.to_parquet(path, engine="pyarrow")

    def __parse(self, path) -> Generator[dict, None, None]:
        g = gzip.open(path, "rb")
        for line in g:
            yield eval(line)

    def __to_dataframe(self, path) -> pd.DataFrame:
        i = 0
        dicts = {}
        for d in self.__parse(path):
            dicts[i] = d
            i += 1
        return pd.DataFrame.from_dict(dicts, orient="index")
