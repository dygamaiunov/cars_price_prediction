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



def torque_power_extraction(df):
    # регулярки написаны с помощью ChatGPT
    # извлекаем значение torque (крутящий момент) и нормализуем, т.к. есть записи в европейском стандарте (Nm) и в азиатском (kgm). Приводим к европейскому умножением на 9.8

    # регулярка извлекает числовое значение перед kgm
    df['asian_torque'] = df['torque'].str.extract(r'(?i)(?=.*\bkgm?\b)\b(150(?:\.0+)?|[0-9]?[0-9](?:\.\d+)?|1[0-4][0-9](?:\.\d+)?)\b', expand = False).astype(float)*9.8

    # регулярка извлекает числовое значение перед Nm
    df['european_torque'] = df['torque'].str.extract(r'(?i)(\d+(?:\.\d+)?)\s*Nm(?:\s*@|\s+at)\s*', expand = False).astype(float)
    df['torque_normalized'] = df['european_torque'].fillna(df['asian_torque'])

    # извлекаем rpm (количество оборотов в минуту, на которых достигается указанный torque)
    # важно, что если указано одно значение rpm вместо диапазона, то оно уйдёт в max_rpm
    df['rpm'] = df['torque'].str.extract(r'(\d{1,4}(?:[.,]\d{1,3})?(?:\s*[-–]\s*\d{1,4}(?:[.,]\d{1,3})?)?)(?=[^\d]*rpm)')
    out = df['rpm'].str.replace(',', '').str.extract(r'(?P<min>\d[\d,]*)(?:-(?P<max>\d[\d,]*))?').replace(',', '')

    out['max'] = out['max'].fillna(out['min'])
    df['max_rpm'] = out['max']
    df['max_rpm'] = pd.to_numeric(df['max_rpm'], errors='coerce')

    # финально считаем максимальную  мощность двигателя по формуле P (Вт) = Torque * RPM * 0.10472
    df['torque_power'] = df['torque_normalized'] * df['max_rpm'] * 0.10472

    df.drop(['european_torque', 'asian_torque', 'torque', 'rpm', 'torque_normalized', 'max_rpm'], axis = 1, inplace = True)

    return df


def columns_preprocessing(df):
    df['mileage'] = df['mileage'].str.extract(r'(?i)(\d+(?:\.\d+)?)\s*kmpl', expand=False).astype(float)
    df['engine'] = df['engine'].str.extract(r'(?i)(\d+(?:\.\d+)?)\s*CC', expand=False).astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(?i)(\d+(?:\.\d+)?)\s*bhp', expand=False).astype(float)

    return df


def name_preprocessing(df):
    # регулярки написаны с помощью ChatGPT
    # достаём bs_emission из любой позиции строки
    bs_pattern = r'(BS[ -]?(?:VI|IV|V|I{1,3}|[1-6]))'
    df['bs_emission'] = df['name'].str.extract(bs_pattern, expand=False).str.replace(' ', '').fillna('not stated')

    # убираем bs_emission из name, чтобы не мешался при разборе
    name_clean = (
        df['name']
        .str.replace(bs_pattern, '', regex=True)   # вырезаем "BS IV" и т.п.
        .str.replace(r'\s+', ' ', regex=True)      # схлопываем лишние пробелы
        .str.strip()
    )

    # регулярка для извлечения brand, model (модель машины), variant (вариант модели)
    pattern = r'''^(?P<brand>[A-Za-z]+)\s+(?P<model>[A-Za-z0-9]+(?:\s+[A-Za-z0-9]{1,2})?)(?:\s+(?P<variant>.*))?$'''

    # применяем к очищенному name и добавляем в df
    cols = name_clean.str.extract(pattern)
    df = pd.concat([df, cols], axis=1)
    df.drop(columns = ['name'], inplace = True)

    return df


def type_casting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce').astype('Int64')
    df['seats']  = pd.to_numeric(df['seats'], errors='coerce').astype('Int64')
    return df



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

    tabular = model.named_steps["tabular"]
        feature_names = tabular.get_feature_names_out()

        
        final_reg = model.named_steps["reg"]
        coefs = final_reg.regressor_.coef_

        # cобираем табличку
        coefs_df = pd.DataFrame({
            'Признак': feature_names,
            'Вес': np.abs(coefs)
        }).sort_values('Вес', ascending=False)

    st.subheader("Веса модели:")
    st.dataframe(coefs_df.head(20), use_container_width=True)
