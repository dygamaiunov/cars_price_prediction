import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('improved_ridge_regression_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

@st.cache_data
def load_train():
    # читаем локальный файл
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    return df_train, df_test

df_train, df_test = load_train()

# --- Визуализации ---
st.subheader("📈 Визуализации")


num_cols = df_train.select_dtypes(include=[np.number]).columns

# pairplot
pp = sns.pairplot(df_train[num_cols], y_vars=["selling_price"])
pp.fig.suptitle("Взаимосвязь признаков с ценой", y=1.02)
st.pyplot(pp.fig)

# correlation
pearson_corr = df_train[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", ax=ax)

# заголовок внутри фигуры
ax.set_title("Корреляция числовых признаков", pad=16)

st.pyplot(fig)

# распределение лог таргета
TARGET_COL = "selling_price"

y_train = df_train[TARGET_COL]
y_train_ln = np.log(y_train)  # или np.log1p, если вдруг есть нули

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_train_ln, bins=50)
ax.set_title("Распределение ln(selling_price)")
ax.set_xlabel("ln(selling_price)")
ax.set_ylabel("Частота")

st.pyplot(fig)


# --- Основной интерфейс ---
st.title("🎯 Предсказание стоимости машины")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Предсказание
    test_sample = df.drop(columns=[TARGET_COL])
    y_test_ = df[TARGET_COL]

    prediction = model.predict(test_sample)
    
    # Визуализация результатов
    st.metric("R-квадрат", r2_score(y_test_, prediction))
  

    # --- Веса модели---
    st.subheader("📊 Результаты")

    coefs_df = pd.DataFrame({
        'Признак': test_sample.columns,
        'Вес': np.abs(model.named_steps["reg"].regressor_.coef_)
    }).sort_values('Вес', ascending = False)

    st.subheader("Веса модели:")
    st.dataframe(coefs_df.head(20), use_container_width=True)