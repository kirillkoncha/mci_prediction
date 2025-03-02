import conllu
import fasttext
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances


class KMeansFeatures:
    def __init__(self, model_path: str = "./data/cc.en.300.bin"):
        self.model = fasttext.load_model(model_path)

        nltk.download("stopwords")
        from nltk.corpus import stopwords

        self.stop_words = set(stopwords.words("english"))

    def extract_vectors_for_sid(
        self,
        annotations: list[str],
        output_path: str,
        return_output: bool = True,
        save_output: bool = False,
    ):
        vectors = {"word": [], "vector": []}

        for text in annotations:
            text = conllu.parse(text)
            for sentence in text:
                for token in sentence:
                    if (
                        token["upos"] in ["NOUN", "VERB"]
                        and token["lemma"] not in vectors["word"]
                        and token["lemma"] != "x"
                        and token["lemma"] not in self.stop_words
                    ):
                        vectors["word"].append(token["lemma"].lower())
                        vectors["vector"].append(self.model[token["lemma"].lower()])

        vectors_df = pd.DataFrame.from_dict(vectors)

        if save_output:
            vectors_df.to_csv(output_path, index=False)

        if return_output:
            return vectors_df

    def get_clusters(
        self,
        vectors_df: pd.DataFrame,
        n_clusters: int,
        output_path: str,
        return_output: bool = True,
        save_output: bool = False,
    ):
        cluster_words = {"cluster": [], "words": []}

        X = np.vstack(vectors_df["vector"].values)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1000)
        vectors_df["cluster"] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        for i in range(n_clusters):
            cluster_df = vectors_df[
                vectors_df["cluster"] == i
            ]  # Get words in cluster i
            cluster_vectors = np.vstack(cluster_df["vector"].values)
            centroid = centroids[i].reshape(1, -1)

            # Compute cosine distances
            distances = cosine_distances(cluster_vectors, centroid).flatten()
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            # Find words within 3 standard deviations
            mask = distances <= (mean_dist + 3 * std_dist)
            filtered_words = cluster_df.loc[mask, "word"].tolist()

            # Store results
            cluster_words["cluster"].append(i)
            cluster_words["words"].append(filtered_words)

        cluster_words = pd.DataFrame.from_dict(cluster_words)

        if save_output:
            cluster_words.to_csv(output_path, index=False)

        if return_output:
            return cluster_words
