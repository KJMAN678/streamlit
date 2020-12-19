import streamlit as st
import pandas as pd
import numpy as np
import math

import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

st.title('エイムズ州のHousePriceバリデーション')
st.header('データセット:kaggleのHousePriceコンペ')

# サイドバー
# ファイルアップロード
train = pd.read_csv("train.csv").sort_values("SalePrice").reset_index().reset_index()

# 特徴量設定
columns_list = list(train.columns[(train.isnull().sum()==0).values])
columns_list.pop(0)
columns_list.pop(0)
columns_list.pop(-1)

columns_list_corr = sorted(columns_list)
columns_list_corr.append("SalePrice")

axis_1 = st.sidebar.selectbox('特徴量1',(sorted(columns_list)))
axis_2 = st.sidebar.selectbox('特徴量2',(sorted(columns_list)))
axis_3 = st.sidebar.selectbox('特徴量3',(sorted(columns_list)))
axis_4 = st.sidebar.selectbox('特徴量4',(sorted(columns_list)))
axis_5 = st.sidebar.selectbox('特徴量5',(sorted(columns_list)))
axis_6 = st.sidebar.selectbox('特徴量6',(sorted(columns_list)))

# 欠損値除外
y = train[train.columns[(train.isnull().sum()==0).values]]["SalePrice"]
X = train[train.columns[(train.isnull().sum()==0).values]].drop("SalePrice", axis=1)

# ラベルエンコーディング
le = LabelEncoder()
for i in list(X.select_dtypes(exclude='number').columns):
    X[i] = le.fit_transform(X[i])

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)


# メイン画面
st.header('読込んだデータセット')

# アップロードファイルをメイン画面にデータ表示
st.write(train[columns_list_corr])

st.header('SalePriceとの相関係数')
st.write(train[columns_list_corr].corr()[-1:])

# 選ばれたx軸の値からグラフ化

# 選択した特徴量で学習
model = lgb.LGBMRegressor()
model.fit(X_train[[axis_1, axis_2, axis_3, axis_4, axis_5, axis_6]], y_train)
y_pred = model.predict(X_test.sort_index()[[axis_1, axis_2, axis_3, axis_4, axis_5, axis_6]])

st.header('結果')

result = pd.DataFrame([[r2_score(y_test.sort_index(), y_pred)],
    [math.ceil(np.sqrt(mean_squared_error(y_test.sort_index(), y_pred)))]], index=["R2スコア", "RMSE"], columns=["結果"])

st.write(result)

chart_data = pd.DataFrame({"実績":y_test.sort_values(),"予測":y_pred})

st.line_chart(chart_data)