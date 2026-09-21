"""
Проектная работа по NLP — вариант 4.
Анализ тональности отзывов о товарах маркетплейса, недели 1–6.

Запуск из VS Code: python main.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import platform
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import FastText, Word2Vec
import joblib
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import save_npz
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from wordcloud import WordCloud

from preprocessing import RUSSIAN_STOPWORDS, preprocess_text
from vector_models import (
    all_words_available,
    average_document_vectors,
    cosine_similarity,
    train_glove,
    vocabulary_token_coverage,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "wildberries_reviews_1000.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
SPACY_MODEL = "ru_core_news_sm"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["axes.titlesize"] = 14


def simple_words(text: str) -> list[str]:
    """Выделяет слова для исходной статистики."""
    return re.findall(r"(?iu)\b[а-яёa-z]+(?:-[а-яёa-z]+)?\b", str(text))


def save_table(table: pd.DataFrame, filename: str) -> None:
    path = RESULTS_DIR / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Сохранена таблица: {path.name}")


def load_corpus() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден корпус {DATA_PATH.name}. Положите CSV рядом с main.py."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"review_text", "rating", "sentiment", "category_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {sorted(missing)}")
    return df


def week_1_overview(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ ОСМОТР КОРПУСА")
    print("=" * 72)

    df["char_count"] = df["review_text"].str.len()
    df["word_count_raw"] = df["review_text"].map(lambda x: len(simple_words(x)))
    raw_words = [word.lower() for text in df["review_text"] for word in simple_words(text)]
    summary = pd.DataFrame(
        {
            "метрика": [
                "Количество документов", "Всего слов до очистки",
                "Уникальных слов до очистки", "Пустых отзывов",
                "Полных дубликатов", "Среднее слов в отзыве",
                "Медиана слов в отзыве",
            ],
            "значение": [
                len(df), len(raw_words), len(set(raw_words)),
                int(df["review_text"].isna().sum() + df["review_text"].str.strip().eq("").sum()),
                int(df["review_text"].duplicated().sum()),
                round(df["word_count_raw"].mean(), 2),
                float(df["word_count_raw"].median()),
            ],
        }
    )
    print(summary.to_string(index=False))
    save_table(summary, "week1_corpus_summary.csv")

    rating_table = (
        df.groupby(["rating", "sentiment"], observed=True)
        .size().rename("count").reset_index()
    )
    category_table = (
        df["category_label"].value_counts()
        .rename_axis("category").reset_index(name="count")
    )
    save_table(rating_table, "week1_rating_distribution.csv")
    save_table(category_table, "week1_category_distribution.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.histplot(df["word_count_raw"], bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Длина отзывов", xlabel="Слов в отзыве", ylabel="Количество")
    sns.countplot(data=df, x="rating", hue="rating", legend=False, ax=axes[1], palette="viridis")
    axes[1].set(title="Распределение оценок", xlabel="Оценка", ylabel="Количество")
    sns.barplot(
        data=category_table, y="category", x="count", hue="category",
        legend=False, ax=axes[2], palette="crest",
    )
    axes[2].set(title="Категории товаров", xlabel="Количество", ylabel="")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week1_corpus_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: week1_corpus_overview.png")
    return df


def week_2_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 2. ОЧИСТКА И ПРЕДОБРАБОТКА")
    print("=" * 72)

    demo = [
        "ОченьХороший товар!!!РекомендуюВсем",
        "Купила2штуки,однаНеРаботает...",
        "НЕ понравилось:доставка3дня,но товар норм",
    ]
    demo_table = pd.DataFrame(
        {"до": demo, "после": [preprocess_text(text)["clean_text"] for text in demo]}
    )
    print("\nПримеры обработки слитного текста:")
    print(demo_table.to_string(index=False))
    save_table(demo_table, "week2_glued_text_examples.csv")

    processed = df["review_text"].map(preprocess_text).apply(pd.Series)
    df = pd.concat([df, processed], axis=1)
    df["token_count_clean"] = df["tokens"].str.len()
    before_vocab = Counter(
        word.lower() for text in df["review_text"] for word in simple_words(text)
    )
    after_vocab = Counter(token for tokens in df["tokens"] for token in tokens)
    before_after = pd.DataFrame(
        {
            "метрика": [
                "Всего слов/токенов", "Размер словаря",
                "Среднее на документ", "Медиана на документ",
            ],
            "до": [
                sum(before_vocab.values()), len(before_vocab),
                df["word_count_raw"].mean(), df["word_count_raw"].median(),
            ],
            "после": [
                sum(after_vocab.values()), len(after_vocab),
                df["token_count_clean"].mean(), df["token_count_clean"].median(),
            ],
        }
    ).round(2)
    print("\nСравнение до и после очистки:")
    print(before_after.to_string(index=False))
    save_table(before_after, "week2_before_after.csv")
    save_table(
        df[["review_text", "clean_text"]].sample(10, random_state=42),
        "week2_review_examples.csv",
    )
    return df


def counter_table(counter: Counter, names: dict[str, str], column_name: str) -> pd.DataFrame:
    total = sum(counter.values())
    return pd.DataFrame(
        [
            {
                column_name: names.get(code, code), "код": code,
                "количество": count, "доля_%": round(count / total * 100, 2),
            }
            for code, count in counter.most_common()
        ]
    )


def week_3_morphology(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 3. ЛЕММАТИЗАЦИЯ, МОРФОАНАЛИЗ И СТЕММИНГ")
    print("=" * 72)

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "Не найдена модель ru_core_news_sm. Выполните: "
            "pip install -r requirements_weeks_1_6.txt"
        ) from exc

    lemma_lists: list[list[str]] = []
    pos_counter: Counter = Counter()
    case_counter: Counter = Counter()
    number_counter: Counter = Counter()
    for doc in nlp.pipe(df["normalized_text"], batch_size=64):
        lemmas = []
        for token in doc:
            lower = token.text.lower()
            if not token.is_alpha:
                continue
            pos_counter[token.pos_] += 1
            case_counter.update(token.morph.get("Case"))
            number_counter.update(token.morph.get("Number"))
            if lower not in RUSSIAN_STOPWORDS:
                lemmas.append(token.lemma_.lower().strip() or lower)
        lemma_lists.append(lemmas)
    df["lemmas"] = lemma_lists
    df["lemma_text"] = df["lemmas"].str.join(" ")

    pos_names = {
        "NOUN": "существительное", "VERB": "глагол", "ADJ": "прилагательное",
        "ADV": "наречие", "PRON": "местоимение", "ADP": "предлог",
        "CCONJ": "сочинительный союз", "SCONJ": "подчинительный союз",
        "PART": "частица", "DET": "определитель", "NUM": "числительное",
        "PROPN": "имя собственное", "AUX": "вспомогательный глагол",
        "INTJ": "междометие", "X": "другое",
    }
    case_names = {
        "Nom": "именительный", "Gen": "родительный", "Dat": "дательный",
        "Acc": "винительный", "Ins": "творительный", "Loc": "предложный",
    }
    number_names = {"Sing": "единственное", "Plur": "множественное"}
    pos_table = counter_table(pos_counter, pos_names, "часть_речи")
    case_table = counter_table(case_counter, case_names, "падеж")
    number_table = counter_table(number_counter, number_names, "число")
    save_table(pos_table, "week3_parts_of_speech.csv")
    save_table(case_table, "week3_cases.csv")
    save_table(number_table, "week3_numbers.csv")

    word_freq = Counter(token for tokens in df["tokens"] for token in tokens)
    lemma_freq = Counter(lemma for lemmas in df["lemmas"] for lemma in lemmas)
    top_words = pd.DataFrame(word_freq.most_common(50), columns=["слово", "частота_слова"])
    top_lemmas = pd.DataFrame(lemma_freq.most_common(50), columns=["лемма", "частота_леммы"])
    save_table(pd.concat([top_words, top_lemmas], axis=1), "week3_top50_words_and_lemmas.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(data=top_words.head(20), y="слово", x="частота_слова", ax=axes[0], color="#4C72B0")
    axes[0].set(title="Топ-20 словоформ", xlabel="Частота", ylabel="")
    sns.barplot(data=top_lemmas.head(20), y="лемма", x="частота_леммы", ax=axes[1], color="#55A868")
    axes[1].set(title="Топ-20 лемм", xlabel="Частота", ylabel="")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week3_top_words_lemmas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: week3_top_words_lemmas.png")

    stemmer = SnowballStemmer("russian")
    all_tokens = [token for tokens in df["tokens"] for token in tokens]
    all_lemmas = [lemma for lemmas in df["lemmas"] for lemma in lemmas]
    all_stems = [stemmer.stem(token) for token in all_tokens]
    compare_tokens = [word for word, _ in word_freq.most_common(25)]
    compare_doc = nlp(" ".join(compare_tokens))
    examples = pd.DataFrame(
        {
            "словоформа": compare_tokens,
            "лемма_spaCy": [token.lemma_.lower() for token in compare_doc],
            "стем_NLTK": [stemmer.stem(word) for word in compare_tokens],
        }
    )
    save_table(examples, "week3_lemmatization_vs_stemming_examples.csv")

    method_summary = pd.DataFrame(
        {
            "подход": ["Словоформы после очистки", "Леммы spaCy", "Стемы Snowball"],
            "уникальных_единиц": [len(set(all_tokens)), len(set(all_lemmas)), len(set(all_stems))],
            "интерпретируемость": ["высокая, формы раздроблены", "высокая", "средняя/низкая"],
            "учитывает_контекст": ["нет", "да", "нет"],
            "сохраняет_словарную_форму": ["не всегда", "да", "нет"],
        }
    )
    method_summary["доля_от_словаря_слов_%"] = (
        method_summary["уникальных_единиц"]
        / method_summary.loc[0, "уникальных_единиц"] * 100
    ).round(2)
    save_table(method_summary, "week3_method_comparison.csv")

    lemma_reduction = 100 * (1 - len(set(all_lemmas)) / len(set(all_tokens)))
    stem_reduction = 100 * (1 - len(set(all_stems)) / len(set(all_tokens)))
    top_pos, top_case, top_number = pos_table.iloc[0], case_table.iloc[0], number_table.iloc[0]
    conclusions = f"""ИТОГОВЫЕ ВЫВОДЫ ЗА НЕДЕЛИ 1–3

1. В корпусе {len(df)} уникальных непустых отзывов и {int(df['word_count_raw'].sum())} слов до очистки.
2. После очистки осталось {sum(word_freq.values())} содержательных токенов.
3. Самая частая часть речи — {top_pos['часть_речи']} ({top_pos['доля_%']}%).
4. Самый частый падеж — {top_case['падеж']} ({top_case['доля_%']}% размеченных форм).
5. Преобладающее число — {top_number['число']} ({top_number['доля_%']}%).
6. Лемматизация сократила словарь на {lemma_reduction:.1f}%, стемминг — на {stem_reduction:.1f}%.
7. Для проекта выбрана лемматизация spaCy: она возвращает словарные формы, учитывает контекст и лучше подходит для объяснения результатов.
8. SnowballStemmer быстрее, но создаёт усечённые основы, которые сложнее интерпретировать.
9. Отрицания «не», «нет», «ни» сохранены, поскольку они меняют тональность фразы.
10. Оценки несбалансированы, а разметка по звёздам может содержать шум. На этапе классификации понадобятся стратификация и macro-F1.
""".strip() + "\n"
    (RESULTS_DIR / "conclusions_weeks_1_3.txt").write_text(conclusions, encoding="utf-8")
    print("\n" + conclusions)
    return df


def find_cyrillic_font() -> str | None:
    """Находит шрифт с кириллицей для облака слов."""
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def week_4_bow_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """Строит и сравнивает разреженные матрицы BOW и TF-IDF."""
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 4. BAG-OF-WORDS И TF-IDF")
    print("=" * 72)

    # Одинаковые параметры обеспечивают одинаковый словарь и честное сравнение.
    vectorizer_params = {
        "lowercase": False,
        "token_pattern": r"(?u)\b[а-яё]+(?:-[а-яё]+)?\b",
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 5000,
        "ngram_range": (1, 1),
    }
    documents = df["lemma_text"].fillna("")
    bow_vectorizer = CountVectorizer(**vectorizer_params)
    tfidf_vectorizer = TfidfVectorizer(
        **vectorizer_params,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    bow_matrix = bow_vectorizer.fit_transform(documents)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    bow_terms = bow_vectorizer.get_feature_names_out()
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    if not np.array_equal(bow_terms, tfidf_terms):
        raise RuntimeError("Словари BOW и TF-IDF не совпали при одинаковых параметрах.")

    save_npz(RESULTS_DIR / "week4_bow_matrix.npz", bow_matrix)
    save_npz(RESULTS_DIR / "week4_tfidf_matrix.npz", tfidf_matrix)
    vocabulary = pd.DataFrame(
        {"column_index": np.arange(len(bow_terms)), "term": bow_terms}
    )
    save_table(vocabulary, "week4_vocabulary.csv")

    bow_scores = np.asarray(bow_matrix.sum(axis=0)).ravel()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    document_frequency = np.asarray((bow_matrix > 0).sum(axis=0)).ravel()
    scores = pd.DataFrame(
        {
            "term": bow_terms,
            "bow_count": bow_scores.astype(int),
            "tfidf_mean": tfidf_scores,
            "document_frequency": document_frequency.astype(int),
        }
    )
    scores["bow_rank"] = scores["bow_count"].rank(method="min", ascending=False).astype(int)
    scores["tfidf_rank"] = scores["tfidf_mean"].rank(method="min", ascending=False).astype(int)
    scores["rank_change_tfidf_vs_bow"] = scores["bow_rank"] - scores["tfidf_rank"]

    top_n = 30
    top_bow_set = set(scores.nsmallest(top_n, "bow_rank")["term"])
    top_tfidf_set = set(scores.nsmallest(top_n, "tfidf_rank")["term"])
    scores["top_30_group"] = np.select(
        [
            scores["term"].isin(top_bow_set & top_tfidf_set),
            scores["term"].isin(top_bow_set - top_tfidf_set),
            scores["term"].isin(top_tfidf_set - top_bow_set),
        ],
        ["оба метода", "только BOW", "только TF-IDF"],
        default="вне топ-30",
    )
    scores["best_rank"] = scores[["bow_rank", "tfidf_rank"]].min(axis=1)
    comparison = (
        scores[(scores["bow_rank"] <= 50) | (scores["tfidf_rank"] <= 50)]
        .sort_values(["best_rank", "term"])
        .drop(columns="best_rank")
    )
    save_table(comparison, "week4_bow_vs_tfidf_top_words.csv")

    matrix_summary = pd.DataFrame(
        {
            "method": ["BOW", "TF-IDF"],
            "documents": [bow_matrix.shape[0], tfidf_matrix.shape[0]],
            "features": [bow_matrix.shape[1], tfidf_matrix.shape[1]],
            "nonzero_values": [bow_matrix.nnz, tfidf_matrix.nnz],
            "sparsity_%": [
                100 * (1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])),
                100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])),
            ],
            "average_nonzero_per_document": [
                bow_matrix.nnz / bow_matrix.shape[0],
                tfidf_matrix.nnz / tfidf_matrix.shape[0],
            ],
            "weight_meaning": ["число вхождений", "нормированный вес TF-IDF"],
        }
    )
    matrix_summary[["sparsity_%", "average_nonzero_per_document"]] = matrix_summary[
        ["sparsity_%", "average_nonzero_per_document"]
    ].round(2)
    save_table(matrix_summary, "week4_matrix_summary.csv")
    print("\nХарактеристики матриц:")
    print(matrix_summary.to_string(index=False))

    top_bow = scores.nlargest(20, "bow_count").sort_values("bow_count")
    top_tfidf = scores.nlargest(20, "tfidf_mean").sort_values("tfidf_mean")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    axes[0].barh(top_bow["term"], top_bow["bow_count"], color="#4C72B0")
    axes[0].set(title="BOW: топ-20 по суммарной частоте", xlabel="Количество вхождений", ylabel="")
    axes[1].barh(top_tfidf["term"], top_tfidf["tfidf_mean"], color="#DD8452")
    axes[1].set(title="TF-IDF: топ-20 по среднему весу", xlabel="Средний TF-IDF", ylabel="")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week4_bow_vs_tfidf_top_words.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: week4_bow_vs_tfidf_top_words.png")

    font_path = find_cyrillic_font()
    bow_cloud = WordCloud(
        width=1400, height=800, background_color="white", colormap="Blues",
        max_words=120, random_state=42, font_path=font_path,
    ).generate_from_frequencies(dict(zip(bow_terms, bow_scores)))
    tfidf_cloud = WordCloud(
        width=1400, height=800, background_color="white", colormap="Oranges",
        max_words=120, random_state=42, font_path=font_path,
    ).generate_from_frequencies(dict(zip(tfidf_terms, tfidf_scores)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(bow_cloud, interpolation="bilinear")
    axes[0].set_title("Облако слов BOW")
    axes[0].axis("off")
    axes[1].imshow(tfidf_cloud, interpolation="bilinear")
    axes[1].set_title("Облако слов TF-IDF")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week4_bow_tfidf_wordclouds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Сохранены облака слов: week4_bow_tfidf_wordclouds.png")

    overlap = sorted(top_bow_set & top_tfidf_set)
    bow_only = (
        scores[scores["term"].isin(top_bow_set - top_tfidf_set)]
        .sort_values("bow_rank")["term"].tolist()
    )
    tfidf_only = (
        scores[scores["term"].isin(top_tfidf_set - top_bow_set)]
        .sort_values("tfidf_rank")["term"].tolist()
    )
    conclusions = f"""СРАВНИТЕЛЬНЫЙ АНАЛИЗ BOW И TF-IDF

1. Обе матрицы имеют размер {bow_matrix.shape[0]} × {bow_matrix.shape[1]} и разреженность {matrix_summary.loc[0, 'sparsity_%']}%.
2. BOW хранит абсолютное число вхождений. Поэтому выше поднимаются слова, которые часто повторяются во всём корпусе: {', '.join(scores.nsmallest(10, 'bow_rank')['term'])}.
3. TF-IDF снижает вес слов, встречающихся во многих документах, и выделяет более специфичные признаки: {', '.join(scores.nsmallest(10, 'tfidf_rank')['term'])}.
4. В топ-30 двух методов совпали {len(overlap)} слов из 30.
5. Только в топ-30 BOW вошли: {', '.join(bow_only) if bow_only else 'нет уникальных слов'}.
6. Только в топ-30 TF-IDF вошли: {', '.join(tfidf_only) if tfidf_only else 'нет уникальных слов'}.
7. Для объяснения общей лексики корпуса удобнее BOW. Для будущей классификации тональности предпочтительнее TF-IDF, поскольку он уменьшает влияние повсеместно встречающихся слов и усиливает различительные признаки.
8. Отрицания сохранены в признаках, так как слова «не» и «нет» напрямую влияют на тональность отзыва.
""".strip() + "\n"
    (RESULTS_DIR / "conclusions_week4_bow_vs_tfidf.txt").write_text(
        conclusions, encoding="utf-8"
    )
    print("\n" + conclusions)
    return scores


def week_5_vector_models(df: pd.DataFrame) -> dict[str, object]:
    """Обучает Word2Vec, GloVe и fastText и сравнивает их семантику."""
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 5. WORD2VEC, GLOVE И FASTTEXT")
    print("=" * 72)

    sentences = [list(tokens) for tokens in df["lemmas"] if len(tokens) > 0]
    vector_size = 50
    common_params = {
        "vector_size": vector_size,
        "window": 5,
        "min_count": 2,
        "workers": 1,
        "seed": 42,
        "epochs": 60,
    }

    word2vec = Word2Vec(sentences=sentences, sg=1, negative=10, **common_params)
    fasttext = FastText(
        sentences=sentences,
        sg=1,
        negative=10,
        min_n=3,
        max_n=5,
        bucket=20_000,
        **common_params,
    )
    glove_vectors, glove_loss = train_glove(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=2,
        epochs=20,
        seed=42,
    )

    word2vec.save(str(MODELS_DIR / "word2vec_marketplace.model"))
    fasttext.save(str(MODELS_DIR / "fasttext_marketplace.model"))
    glove_vectors.save(str(MODELS_DIR / "glove_marketplace.kv"))
    loss_table = pd.DataFrame(
        {"epoch": np.arange(1, len(glove_loss) + 1), "mean_weighted_loss": glove_loss}
    )
    save_table(loss_table, "week5_glove_training_loss.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(loss_table["epoch"], loss_table["mean_weighted_loss"], marker="o", color="#8172B2")
    ax.set(title="Обучение GloVe на собственном корпусе", xlabel="Эпоха", ylabel="Средняя взвешенная ошибка")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week5_glove_training_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    vectors_by_model = {
        "Word2Vec": word2vec.wv,
        "GloVe": glove_vectors,
        "fastText": fasttext.wv,
    }
    corpus_vocabulary = set(token for sentence in sentences for token in sentence)
    comparison_rows = []
    for model_name, vectors in vectors_by_model.items():
        known_words = sum(word in vectors.key_to_index for word in corpus_vocabulary)
        comparison_rows.append(
            {
                "model": model_name,
                "vector_size": vectors.vector_size,
                "vocabulary_size": len(vectors.index_to_key),
                "known_unique_words": known_words,
                "unique_word_coverage_%": round(known_words / len(corpus_vocabulary) * 100, 2),
                "token_coverage_%": round(vocabulary_token_coverage(sentences, vectors) * 100, 2),
                "supports_oov": "да" if model_name == "fastText" else "нет",
                "training_source": "собственный корпус",
            }
        )
    model_comparison = pd.DataFrame(comparison_rows)

    probe_candidates = [
        "качество", "размер", "доставка", "цена", "цвет",
        "запах", "упаковка", "продавец", "крем", "покупка",
    ]
    probe_words = [
        word for word in probe_candidates
        if all(all_words_available(vectors, [word]) for vectors in vectors_by_model.values())
    ]
    nearest_rows = []
    for model_name, vectors in vectors_by_model.items():
        for query in probe_words:
            for rank, (neighbor, similarity) in enumerate(vectors.most_similar(query, topn=5), start=1):
                nearest_rows.append(
                    {
                        "model": model_name,
                        "query": query,
                        "rank": rank,
                        "neighbor": neighbor,
                        "cosine_similarity": round(float(similarity), 4),
                    }
                )
    nearest_table = pd.DataFrame(nearest_rows)
    save_table(nearest_table, "week5_nearest_words.csv")

    analogy_tasks = [
        ("хороший - плохой + качество", ["хороший", "качество"], ["плохой"]),
        ("большой - маленький + размер", ["большой", "размер"], ["маленький"]),
        ("дорогой - дешёвый + цена", ["дорогой", "цена"], ["дешёвый"]),
    ]
    analogy_rows = []
    for model_name, vectors in vectors_by_model.items():
        for expression, positive, negative in analogy_tasks:
            if not all_words_available(vectors, positive + negative):
                continue
            for rank, (word, similarity) in enumerate(
                vectors.most_similar(positive=positive, negative=negative, topn=5), start=1
            ):
                analogy_rows.append(
                    {
                        "model": model_name,
                        "expression": expression,
                        "rank": rank,
                        "result": word,
                        "cosine_similarity": round(float(similarity), 4),
                    }
                )
    analogy_table = pd.DataFrame(analogy_rows)
    save_table(analogy_table, "week5_vector_arithmetic.csv")

    related_pairs = [
        ("хороший", "отличный"), ("красивый", "приятный"),
        ("плохой", "ужасный"), ("цена", "стоимость"),
        ("коробка", "упаковка"), ("крем", "кожа"),
        ("размер", "подойти"), ("волос", "шампунь"),
    ]
    unrelated_pairs = [
        ("крем", "доставка"), ("размер", "запах"),
        ("цвет", "цена"), ("волос", "коробка"),
        ("шампунь", "продавец"),
    ]
    similarity_rows = []
    quality_rows = []
    for model_name, vectors in vectors_by_model.items():
        model_scores: dict[str, list[float]] = {"related": [], "unrelated": []}
        for relation, pairs in (("related", related_pairs), ("unrelated", unrelated_pairs)):
            for first, second in pairs:
                if not all_words_available(vectors, [first, second]):
                    continue
                score = cosine_similarity(vectors, first, second)
                model_scores[relation].append(score)
                similarity_rows.append(
                    {
                        "model": model_name,
                        "pair_type": relation,
                        "first_word": first,
                        "second_word": second,
                        "cosine_similarity": round(score, 4),
                    }
                )
        related_mean = float(np.mean(model_scores["related"]))
        unrelated_mean = float(np.mean(model_scores["unrelated"]))
        quality_rows.append(
            {
                "model": model_name,
                "related_pairs_mean": round(related_mean, 4),
                "unrelated_pairs_mean": round(unrelated_mean, 4),
                "semantic_separation": round(related_mean - unrelated_mean, 4),
            }
        )
    similarity_table = pd.DataFrame(similarity_rows)
    quality_table = pd.DataFrame(quality_rows)
    save_table(similarity_table, "week5_similarity_pairs.csv")
    save_table(quality_table, "week5_semantic_quality.csv")
    model_comparison = model_comparison.merge(quality_table, on="model", how="left")
    save_table(model_comparison, "week5_model_comparison.csv")

    best_semantic = quality_table.sort_values("semantic_separation", ascending=False).iloc[0]
    conclusions = f"""ВЫВОДЫ ПО НЕДЕЛЕ 5

1. Word2Vec, GloVe и fastText обучены на одном лемматизированном корпусе из {len(sentences)} отзывов.
2. Размерность всех векторов равна {vector_size}, минимальная частота слова — 2.
3. Все модели покрывают частотную лексику корпуса; fastText дополнительно строит векторы для незнакомых слов через символьные n-граммы.
4. По разнице близости связанных и несвязанных контрольных пар лучший результат показал {best_semantic['model']} ({best_semantic['semantic_separation']:.4f}).
5. Таблица ближайших слов показывает локальные семантические связи, а арифметика векторов демонстрирует перенос направлений между понятиями.
6. Корпус сравнительно небольшой, поэтому отдельные соседи и аналогии могут быть нестабильными. Качество дополнительно проверяется через классификацию тональности на неделе 6.
""".strip() + "\n"
    (RESULTS_DIR / "conclusions_week5_vectors.txt").write_text(conclusions, encoding="utf-8")
    print("\n" + conclusions)
    return vectors_by_model


def lemmatize_new_reviews(texts: list[str]) -> list[list[str]]:
    """Применяет тот же пайплайн к новым отзывам для рабочего прототипа."""
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    normalized_texts = [preprocess_text(text)["normalized_text"] for text in texts]
    output = []
    for doc in nlp.pipe(normalized_texts, batch_size=32):
        output.append(
            [
                (token.lemma_.lower().strip() or token.text.lower())
                for token in doc
                if token.is_alpha and token.text.lower() not in RUSSIAN_STOPWORDS
            ]
        )
    return output


def week_6_sentiment_prototype(
    df: pd.DataFrame,
    vectors_by_model: dict[str, object],
) -> None:
    """Обучает прототип классификации тональности на усреднённых векторах."""
    print("\n" + "=" * 72)
    print("НЕДЕЛЯ 6. КЛАССИФИКАЦИЯ ТОНАЛЬНОСТИ НА ОСНОВЕ ВЕКТОРОВ")
    print("=" * 72)

    sentences = [list(tokens) for tokens in df["lemmas"]]
    labels = df["sentiment"].to_numpy()
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    metric_rows = []
    trained_classifiers = {}
    document_vectors_by_model = {}
    for model_name, vectors in vectors_by_model.items():
        document_vectors = average_document_vectors(sentences, vectors)
        document_vectors_by_model[model_name] = document_vectors
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        classifier.fit(document_vectors[train_indices], labels[train_indices])
        predictions = classifier.predict(document_vectors[test_indices])
        trained_classifiers[model_name] = classifier
        metric_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(labels[test_indices], predictions),
                "balanced_accuracy": balanced_accuracy_score(labels[test_indices], predictions),
                "macro_f1": f1_score(labels[test_indices], predictions, average="macro"),
                "weighted_f1": f1_score(labels[test_indices], predictions, average="weighted"),
                "zero_vector_documents": int(np.all(document_vectors == 0, axis=1).sum()),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    numeric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    metrics[numeric_columns] = metrics[numeric_columns].round(4)
    save_table(metrics, "week6_classifier_comparison.csv")
    print("\nСравнение классификаторов:")
    print(metrics.to_string(index=False))

    best_model_name = str(metrics.iloc[0]["model"])
    best_classifier = trained_classifiers[best_model_name]
    best_vectors = vectors_by_model[best_model_name]
    best_document_vectors = document_vectors_by_model[best_model_name]
    best_predictions = best_classifier.predict(best_document_vectors[test_indices])
    class_order = ["negative", "neutral", "positive"]

    report = pd.DataFrame(
        classification_report(
            labels[test_indices],
            best_predictions,
            labels=class_order,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="class_or_average")
    save_table(report, "week6_best_model_classification_report.csv")

    matrix = confusion_matrix(labels[test_indices], best_predictions, labels=class_order)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
        ax=ax,
    )
    ax.set(
        title=f"Матрица ошибок: {best_model_name}",
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week6_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    normalized_vectors = normalize(best_document_vectors)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(normalized_vectors)
    pca_table = pd.DataFrame(
        {
            "pca_1": coordinates[:, 0],
            "pca_2": coordinates[:, 1],
            "sentiment": labels,
            "rating": df["rating"].to_numpy(),
        }
    )
    save_table(pca_table, "week6_document_vectors_pca.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"negative": "#C44E52", "neutral": "#8172B2", "positive": "#55A868"}
    sns.scatterplot(
        data=pca_table,
        x="pca_1",
        y="pca_2",
        hue="sentiment",
        hue_order=class_order,
        palette=palette,
        alpha=0.65,
        s=45,
        ax=ax,
    )
    ax.set(
        title=f"PCA-визуализация векторов отзывов ({best_model_name})",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% дисперсии)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% дисперсии)",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "week6_document_vectors_pca.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    sample_reviews = [
        "Отличный товар, качество очень хорошее, рекомендую",
        "Ужасное качество, товар сломался в первый день",
        "Обычный товар, есть плюсы и минусы",
        "Размер подошёл, материал приятный и удобный",
        "Не работает, упаковка повреждена, деньги потрачены зря",
    ]
    sample_lemmas = lemmatize_new_reviews(sample_reviews)
    sample_vectors = average_document_vectors(sample_lemmas, best_vectors)
    sample_predictions = best_classifier.predict(sample_vectors)
    sample_probabilities = best_classifier.predict_proba(sample_vectors)
    sample_table = pd.DataFrame(
        {
            "review": sample_reviews,
            "lemmas": [" ".join(tokens) for tokens in sample_lemmas],
            "predicted_sentiment": sample_predictions,
            "confidence": sample_probabilities.max(axis=1).round(4),
        }
    )
    save_table(sample_table, "week6_prototype_predictions.csv")

    joblib.dump(
        {"model_name": best_model_name, "classifier": best_classifier},
        MODELS_DIR / "sentiment_classifier.joblib",
    )
    np.save(RESULTS_DIR / "week6_best_document_vectors.npy", best_document_vectors)
    metadata = {
        "best_embedding_model": best_model_name,
        "classes": best_classifier.classes_.tolist(),
        "vector_size": int(best_document_vectors.shape[1]),
        "test_size": int(len(test_indices)),
        "random_state": 42,
        "metric_used_for_selection": "macro_f1",
        "embedding_files": {
            "Word2Vec": "word2vec_marketplace.model",
            "GloVe": "glove_marketplace.kv",
            "fastText": "fasttext_marketplace.model",
        },
    }
    (MODELS_DIR / "sentiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_metrics = metrics.iloc[0]
    conclusions = f"""ВЫВОДЫ ПО НЕДЕЛЕ 6

1. Реализована трёхклассовая классификация тональности: negative, neutral и positive.
2. Каждый отзыв представлен средним вектором его лемм. Для всех моделей использовано одно стратифицированное разбиение 80/20.
3. Лучшей моделью по macro-F1 стала {best_model_name}: accuracy={best_metrics['accuracy']:.4f}, balanced_accuracy={best_metrics['balanced_accuracy']:.4f}, macro-F1={best_metrics['macro_f1']:.4f}.
4. Метрика macro-F1 выбрана из-за дисбаланса классов: она одинаково учитывает положительные, нейтральные и отрицательные отзывы.
5. PCA показывает двумерную структуру векторов документов, но не является доказательством полной разделимости классов.
6. Рабочий прототип принимает новый текст, повторяет очистку и лемматизацию, строит вектор и возвращает тональность с вероятностью.
7. Ограничение: тональность автоматически получена из оценки в звёздах, поэтому разметка содержит шум. Для улучшения нужны больший корпус и ручная проверка части меток.
""".strip() + "\n"
    (RESULTS_DIR / "conclusions_week6_sentiment.txt").write_text(conclusions, encoding="utf-8")
    print("\n" + conclusions)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    print("Проект: анализ тональности отзывов маркетплейса — недели 1–6")
    print(f"Python: {platform.python_version()}")
    print(f"spaCy: {spacy.__version__}")
    print(f"Данные: {DATA_PATH}")
    print("Источник: Hplss/wb-review-dataset, лицензия CC BY-NC-SA 4.0")

    df = week_1_overview(load_corpus())
    df = week_2_preprocessing(df)
    df = week_3_morphology(df)
    week_4_bow_tfidf(df)
    vectors_by_model = week_5_vector_models(df)
    week_6_sentiment_prototype(df, vectors_by_model)
    output_columns = [
        "review_id", "product_id", "category_label", "rating", "sentiment",
        "date", "review_text", "clean_text", "lemma_text",
    ]
    save_table(df[output_columns], "processed_reviews.csv")
    print(f"\nГотово. Все результаты находятся в: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
