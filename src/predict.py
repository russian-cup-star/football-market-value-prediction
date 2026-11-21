"""
Функция предсказания рыночной стоимости футболиста.

Загружает обученную модель, scaler и список признаков,
принимает параметры игрока и возвращает стоимость в евро.
"""
import pickle
import numpy as np
import pandas as pd
import os

# Пути к артефактам (относительно корня репозитория)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Загружаем артефакты один раз при импорте
with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
    SCALER = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb') as f:
    FEATURE_NAMES = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'num_features.pkl'), 'rb') as f:
    NUM_FEATURES = pickle.load(f)


def predict_market_value(
    age: float,
    position: str,
    sub_position: str,
    country_of_citizenship: str,
    league_label: str,
    goals: int = 0,
    assists: int = 0,
    appearances: int = 0,
    minutes_played: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    height_in_cm: float = 180.0,
) -> float:
    """
    Возвращает предсказанную рыночную стоимость игрока в евро.

    Параметры:
        age: возраст (лет)
        position: широкая позиция ('Attack', 'Midfield', 'Defender', 'Goalkeeper')
        sub_position: детальная позиция ('Centre-Forward', 'Right Winger', ...)
        country_of_citizenship: гражданство ('France', 'Brazil', ...)
        league_label: лига в формате 'Страна — Лига' ('Spain — laliga', ...)
        goals, assists, appearances, minutes_played, yellow_cards, red_cards: статистика
        height_in_cm: рост (см)

    Возвращает:
        float: предсказанная стоимость в евро.
    """
    # 1. Собираем словарь признаков
    row = {
        'age': age,
        'height_in_cm': height_in_cm,
        'goals': goals,
        'assists': assists,
        'appearances': appearances,
        'minutes_played': minutes_played,
        'yellow_cards': yellow_cards,
        'red_cards': red_cards,
        'position': position,
        'sub_position': sub_position,
        'country_of_citizenship': country_of_citizenship,
        'league_label': league_label,
    }

    # 2. One-hot для категориальных (как в этапе 3)
    cat_features = ['position', 'sub_position',
                    'country_of_citizenship', 'league_label']
    df_row = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df_row[cat_features], drop_first=True, dtype=int)

    # 3. Собираем матрицу признаков в том же порядке, что при обучении
    X_row = pd.concat([
        df_row[NUM_FEATURES].reset_index(drop=True),
        df_encoded.reset_index(drop=True)
    ], axis=1)

    # 4. Добавляем отсутствующие one-hot колонки (если категория не встретилась)
    for col in FEATURE_NAMES:
        if col not in X_row.columns:
            X_row[col] = 0
    X_row = X_row[FEATURE_NAMES]  # порядок колонок как при обучении

    # 5. Масштабируем числовые
    num_idx = [X_row.columns.get_loc(c) for c in NUM_FEATURES]
    X_row.iloc[:, num_idx] = SCALER.transform(X_row.iloc[:, num_idx])

    # 6. Предсказываем log1p(цену) → переводим в евро
    pred_log = MODEL.predict(X_row)[0]
    pred_eur = float(np.expm1(pred_log))
    return pred_eur


if __name__ == '__main__':
    price = predict_market_value(
        age=25,
        position='Attack',
        sub_position='Centre-Forward',
        country_of_citizenship='France',
        league_label='England — premier-league',
        goals=20,
        assists=5,
        appearances=30,
        minutes_played=2500,
        height_in_cm=185,
    )
    print(f"Пример предсказания: {price:,.0f} €")
