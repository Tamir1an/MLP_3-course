"""Общие функции для обучения и использования векторных моделей."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from gensim.models import KeyedVectors


def train_glove(
    sentences: Sequence[Sequence[str]],
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20,
    learning_rate: float = 0.05,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 42,
) -> tuple[KeyedVectors, list[float]]:
    """Обучает компактную модель GloVe методом AdaGrad на своём корпусе."""
    frequencies = Counter(token for sentence in sentences for token in sentence)
    vocabulary = sorted(
        (word for word, count in frequencies.items() if count >= min_count),
        key=lambda word: (-frequencies[word], word),
    )
    word_to_id = {word: index for index, word in enumerate(vocabulary)}

    cooccurrence: defaultdict[tuple[int, int], float] = defaultdict(float)
    for sentence in sentences:
        ids = [word_to_id[word] for word in sentence if word in word_to_id]
        for center_position, center_id in enumerate(ids):
            left = max(0, center_position - window)
            right = min(len(ids), center_position + window + 1)
            for context_position in range(left, right):
                if center_position == context_position:
                    continue
                distance = abs(center_position - context_position)
                cooccurrence[(center_id, ids[context_position])] += 1.0 / distance

    pairs = np.array(list(cooccurrence.keys()), dtype=np.int32)
    values = np.array(list(cooccurrence.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = 0.5 / vector_size
    word_vectors = rng.normal(0, scale, (len(vocabulary), vector_size))
    context_vectors = rng.normal(0, scale, (len(vocabulary), vector_size))
    word_bias = np.zeros(len(vocabulary), dtype=np.float64)
    context_bias = np.zeros(len(vocabulary), dtype=np.float64)

    grad_sq_word = np.ones_like(word_vectors)
    grad_sq_context = np.ones_like(context_vectors)
    grad_sq_word_bias = np.ones_like(word_bias)
    grad_sq_context_bias = np.ones_like(context_bias)
    weights = np.where(values < x_max, (values / x_max) ** alpha, 1.0)
    log_values = np.log(values)
    loss_history: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        for pair_index in rng.permutation(len(pairs)):
            word_id, context_id = pairs[pair_index]
            weight = weights[pair_index]
            difference = (
                np.dot(word_vectors[word_id], context_vectors[context_id])
                + word_bias[word_id]
                + context_bias[context_id]
                - log_values[pair_index]
            )
            weighted_gradient = weight * difference
            total_loss += 0.5 * weight * difference * difference

            word_vector_copy = word_vectors[word_id].copy()
            word_gradient = weighted_gradient * context_vectors[context_id]
            context_gradient = weighted_gradient * word_vector_copy

            word_vectors[word_id] -= (
                learning_rate * word_gradient / np.sqrt(grad_sq_word[word_id])
            )
            context_vectors[context_id] -= (
                learning_rate * context_gradient / np.sqrt(grad_sq_context[context_id])
            )
            word_bias[word_id] -= (
                learning_rate * weighted_gradient / np.sqrt(grad_sq_word_bias[word_id])
            )
            context_bias[context_id] -= (
                learning_rate * weighted_gradient / np.sqrt(grad_sq_context_bias[context_id])
            )

            grad_sq_word[word_id] += word_gradient * word_gradient
            grad_sq_context[context_id] += context_gradient * context_gradient
            grad_sq_word_bias[word_id] += weighted_gradient * weighted_gradient
            grad_sq_context_bias[context_id] += weighted_gradient * weighted_gradient

        loss_history.append(total_loss / len(pairs))

    vectors = word_vectors + context_vectors
    # На небольшом корпусе GloVe получает сильное общее направление, которое
    # завышает косинусную близость любых слов. Центрирование и удаление первой
    # главной компоненты сохраняют различия и делают сравнение осмысленным.
    vectors -= vectors.mean(axis=0, keepdims=True)
    _, _, principal_components = np.linalg.svd(vectors, full_matrices=False)
    dominant_component = principal_components[0]
    vectors -= np.outer(vectors @ dominant_component, dominant_component)
    vectors = vectors.astype(np.float32)
    keyed_vectors = KeyedVectors(vector_size=vector_size)
    keyed_vectors.add_vectors(vocabulary, vectors)
    keyed_vectors.fill_norms(force=True)
    return keyed_vectors, loss_history


def average_document_vectors(
    sentences: Sequence[Sequence[str]],
    vectors: KeyedVectors,
) -> np.ndarray:
    """Усредняет векторы известных слов для каждого документа."""
    output = np.zeros((len(sentences), vectors.vector_size), dtype=np.float32)
    for row_index, sentence in enumerate(sentences):
        document_vectors = []
        for token in sentence:
            try:
                document_vectors.append(vectors.get_vector(token))
            except KeyError:
                continue
        if document_vectors:
            output[row_index] = np.mean(document_vectors, axis=0)
    return output


def vocabulary_token_coverage(
    sentences: Sequence[Sequence[str]],
    vectors: KeyedVectors,
) -> float:
    """Возвращает долю токенов корпуса, для которых доступен вектор."""
    total = 0
    covered = 0
    for sentence in sentences:
        for token in sentence:
            total += 1
            try:
                vectors.get_vector(token)
                covered += 1
            except KeyError:
                pass
    return covered / total if total else 0.0


def cosine_similarity(vectors: KeyedVectors, first: str, second: str) -> float:
    """Косинусная близость двух слов."""
    return float(vectors.similarity(first, second))


def all_words_available(vectors: KeyedVectors, words: Iterable[str]) -> bool:
    """Проверяет, доступны ли все слова для операций над векторами."""
    for word in words:
        try:
            vectors.get_vector(word)
        except KeyError:
            return False
    return True
