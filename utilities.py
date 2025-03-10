import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pandas as pd
import torch
import pickle
import random
from nltk.tokenize import word_tokenize
import string



def encode(texts, model, intent, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    context_vectors = model.encode("".join(contexts))
    intent_vectors = model.encode(intent)

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
            np.asarray(intent_vectors),
        ],
        axis=-1,
    )


# ===================================================


def cosine_sim(data_vectors, query_vectors) -> list:
    """returns list of tuples with similarity score and
    script index in initial dataframe"""

    data_emb = sparse.csr_matrix(data_vectors)
    query_emb = sparse.csr_matrix(query_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    ind = np.argwhere(similarity)
    match = sorted(zip(similarity, ind.tolist()), reverse=True)

    return match


# ===================================================


def top_candidates(score_lst_sorted, intent, initial_data, top=1):
    """this functions receives results of the cousine similarity ranking and
    returns top items' scores and their indices"""
    intent_idx = initial_data.index[initial_data["INTENT_TAG"] == intent]
    filtered_candiates = [item for item in score_lst_sorted if item[1][0] in intent_idx]
    scores = [item[0] for item in filtered_candiates]
    candidates_indexes = [item[1][0] for item in filtered_candiates]
    return scores[0:top], candidates_indexes[0:top]


# ===================================================


def candidates_reranking(
    top_candidates_idx_lst, conversational_history, utterance, initial_df, pipeline
):
    """this function applies trained bert classifier to identified candidates and
    returns their updated rank"""
    reranked_idx = {}

    for idx in top_candidates_idx_lst:

        combined_text = (
            " ".join(conversational_history)
            + " [SEP] "
            + utterance
            + " [SEP] "
            + initial_df.iloc[idx]["ANSWER"]
        )

        prediction = pipeline(combined_text)
        if prediction[0]["label"] == "LABEL_0":
            reranked_idx[idx] = prediction[0]["score"]
  
    return reranked_idx


# ===================================================


def intent_classification(question, answer, tag_model):
    greetings = ["hi", "hello", "greeting", "greetings", "hii", "helo", "hellow", "how are you?", "howdy", "hey", "heya", "heyo", "hiya", "hiyah", "hola", "howdy-do", "howdy-doody", "shalom", "what's up", "what's happening", "what's going on", "what's new", "what's the news"]
    tokens = word_tokenize(answer.lower())
    for token in tokens:
        if token in greetings:
            return "Conventional-opening"
        else:
            intent = tag_model.predict_tag(question)
            return intent


# ===================================================

