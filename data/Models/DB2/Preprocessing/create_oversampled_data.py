#!/usr/bin/env python3
import os
import csv
import logging
import numpy as np
import time

from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import ADASYN
from joblib import Parallel, delayed
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)

OFFSET = 512
# Adjust these paths and parameters to your needs:
ORIGINAL_TRAIN_PATH = f"../datasets/{OFFSET}_Split/split_dataset_5_US/test.csv"
OVERSAMPLED_TRAIN_PATH = f"../datasets/{OFFSET}_Split/split_dataset_5_US/final_test.csv"

N_JOBS = os.cpu_count() or 1
print(f"💻 Using {N_JOBS} CPU cores for processing.\n")

def load_csv_data(csv_path):
    """
    Loads the CSV, returning:
      header, texts, labels
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))
    header = data[0]
    rows = data[1:]

    # Check format
    if len(rows[0]) == 2:
        # Single-sequence: [text, label]
        texts = [r[0] for r in rows]
        labels = [int(r[1]) for r in rows]
    elif len(rows[0]) == 3:
        # Pair-sequence: [text1, text2, label]
        texts = [f"{r[0]} {r[1]}" for r in rows]
        labels = [int(r[2]) for r in rows]
    else:
        raise ValueError("Data format not supported. Must have 2 or 3 columns.")

    return header, texts, labels

def chunk_list(data_list, chunk_size):
    """Split a list into multiple chunks of size `chunk_size`."""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i : i + chunk_size]

def transform_chunk(vectorizer, chunk):
    """Parallelizable function to transform a chunk of text into numeric features."""
    return vectorizer.transform(chunk)

def inverse_transform_chunk(vectorizer, subX):
    """Parallelizable function to inverse-transform a chunk of numeric features back to tokens."""
    return vectorizer.inverse_transform(subX)

def main():
    total_start_time = time.time()
    logging.info(f"Loading original dataset: {ORIGINAL_TRAIN_PATH}")
    header, texts, labels = load_csv_data(ORIGINAL_TRAIN_PATH)
    labels = np.array(labels)

    # 1) FIT a CountVectorizer
    logging.info("Fitting CountVectorizer on entire dataset...")
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1,1))
    # Fit on all texts (single-threaded)
    vectorizer.fit(texts)
    logging.info("CountVectorizer fitted.")

    # 2) TRANSFORM in parallel
    chunk_size = max(len(texts) // (N_JOBS * 2), 1)  # heuristic chunk size
    text_chunks = list(chunk_list(texts, chunk_size))
    logging.info(f"Transforming {len(texts)} texts in {len(text_chunks)} chunks with n_jobs={N_JOBS}...")

    # Transform each chunk in parallel → returns a list of sparse matrices
    transformed_parts = Parallel(n_jobs=N_JOBS)(
        delayed(transform_chunk)(vectorizer, chunk) for chunk in text_chunks
    )
    # Combine all parts into one big sparse matrix
    X = sp.vstack(transformed_parts)
    logging.info(f"Final shape of X: {X.shape}")

    # 3) Apply ADASYN
    logging.info("Applying ADASYN oversampling...")
    logging.info(f"Class distribution before ADASYN: {Counter(labels)}")
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X, labels)
    logging.info(f"Class distribution after ADASYN: {Counter(y_res)}")

    # 4) INVERSE-TRANSFORM in parallel
    # We convert the oversampled numeric features back to text tokens.
    # Chunk X_res to parallelize inverse_transform
    row_count = X_res.shape[0]
    chunk_size_res = max(row_count // (N_JOBS * 2), 1)
    row_splits = range(0, row_count, chunk_size_res)

    def get_submatrix(matrix, start, end):
        return matrix[start:end]

    sub_matrices = []
    for start_idx in row_splits:
        end_idx = min(start_idx + chunk_size_res, row_count)
        subX = get_submatrix(X_res, start_idx, end_idx)
        sub_matrices.append(subX)

    logging.info(f"Inverse-transforming {row_count} rows in {len(sub_matrices)} chunks with n_jobs={N_JOBS}...")
    inverted_chunks = Parallel(n_jobs=N_JOBS)(
        delayed(inverse_transform_chunk)(vectorizer, subX) for subX in sub_matrices
    )

    # Combine all inverted tokens
    # Each chunk is a list of lists-of-tokens
    X_res_tokens = list(chain.from_iterable(inverted_chunks))
    logging.info("Flattened all inverted chunks.")


    # 5) Rebuild oversampled texts by joining tokens
    oversampled_texts = ["".join(tokens) for tokens in X_res_tokens]
    oversampled_labels = y_res.tolist()

    # 6) Write out oversampled CSV
    os.makedirs(os.path.dirname(OVERSAMPLED_TRAIN_PATH), exist_ok=True)
    logging.info(f"Saving oversampled dataset to {OVERSAMPLED_TRAIN_PATH}")
    with open(OVERSAMPLED_TRAIN_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # If original dataset had 2 columns or 3 columns
        if len(header) == 2:
            # single-sequence format
            writer.writerow(header)  # same header as original
            for text, lab in zip(oversampled_texts, oversampled_labels):
                writer.writerow([text, lab])
        elif len(header) == 3:
            # pair format => you might need to parse back out text1 vs text2
            # for now, we'll just save them as combined text
            writer.writerow(header)
            for text, lab in zip(oversampled_texts, oversampled_labels):
                writer.writerow([text, "", lab])
        else:
            # fallback
            writer.writerow(["text", "label"])
            for text, lab in zip(oversampled_texts, oversampled_labels):
                writer.writerow([text, lab])

    logging.info("Done. Oversampled dataset saved.")
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    minutes, seconds = divmod(total_elapsed_time, 60)
    print(f"🔚 Total script execution time: {int(minutes)}m {seconds:.2f}s")
    
    
if __name__ == "__main__":
    main()
