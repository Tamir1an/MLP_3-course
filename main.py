"""
Проектная работа по NLP — вариант 4.
Анализ тональности отзывов о товарах маркетплейса, недели 1–3.

Запуск из VS Code: python main.py
Неделя 4 (Bag-of-Words и TF-IDF) в этот файл не входит.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import platform
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from nltk.stem.snowball import SnowballStemmer

from preprocessing import RUSSIAN_STOPWORDS, preprocess_text


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "wildberries_reviews_1000.csv"
RESULTS_DIR = BASE_DIR / "results"
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
            "pip install -r requirements_weeks_1_3.txt"
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

Неделя 4 (Bag-of-Words и TF-IDF) пока не выполнялась.
"""
    (RESULTS_DIR / "conclusions_weeks_1_3.txt").write_text(conclusions, encoding="utf-8")
    print("\n" + conclusions)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Проект: анализ тональности отзывов маркетплейса — недели 1–3")
    print(f"Python: {platform.python_version()}")
    print(f"spaCy: {spacy.__version__}")
    print(f"Данные: {DATA_PATH}")
    print("Источник: Hplss/wb-review-dataset, лицензия CC BY-NC-SA 4.0")

    df = week_1_overview(load_corpus())
    df = week_2_preprocessing(df)
    df = week_3_morphology(df)
    output_columns = [
        "review_id", "product_id", "category_label", "rating", "sentiment",
        "date", "review_text", "clean_text", "lemma_text",
    ]
    save_table(df[output_columns], "processed_reviews.csv")
    print(f"\nГотово. Все результаты находятся в: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

