#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import plotly.graph_objs as go
from datetime import timedelta

# Baca file Excel
df = pd.read_excel('UNIT 2 MARET SD JUNI BEBAN DIATAS 180 MW.xlsx')
df['TIME'] = pd.to_datetime(df['TIME'])

# Ganti nama kolom
column_target = 'REALISASI MW UNIT 2'
df.rename(columns={'TIME': 'ds', column_target: 'y'}, inplace=True)

# Feature Engineering
df['hour'] = df['ds'].dt.hour
df['day'] = df['ds'].dt.day
df['month'] = df['ds'].dt.month
df['weekday'] = df['ds'].dt.weekday

# Buat fitur dan label
X = df[['hour', 'day', 'month', 'weekday']]
y = df['y']

# Split data untuk training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Buat model XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)

# Prediksi dengan XGBoost
y_pred_xgb = model_xgb.predict(X_test)
df_result = pd.DataFrame({'ds': df['ds'].iloc[-len(y_test):], 'y_actual': y_test, 'y_pred_xgb': y_pred_xgb})

# Buat data untuk prediksi 7 hari ke depan dengan interval 30 menit
future_dates = [df['ds'].max() + timedelta(minutes=30 * i) for i in range(1, 7 * 48 + 1)]
df_future = pd.DataFrame({'ds': future_dates})
df_future['hour'] = df_future['ds'].dt.hour
df_future['day'] = df_future['ds'].dt.day
df_future['month'] = df_future['ds'].dt.month
df_future['weekday'] = df_future['ds'].dt.weekday

# Prediksi
future_predictions = model_xgb.predict(df_future[['hour', 'day', 'month', 'weekday']])
df_future['y_pred_xgb'] = future_predictions

# Streamlit App
st.title("Prediksi Beban Listrik dengan XGBoost")

# Plot hasil prediksi
data_actual = go.Scatter(x=df_result['ds'], y=df_result['y_actual'], mode='lines', name='Data Aktual', line=dict(color='blue'))
data_predicted = go.Scatter(x=df_result['ds'], y=df_result['y_pred_xgb'], mode='lines', name='Prediksi XGBoost', line=dict(color='red'))
data_future = go.Scatter(x=df_future['ds'], y=df_future['y_pred_xgb'], mode='lines', name='Prediksi 7 Hari ke Depan', line=dict(color='green'))

fig = go.Figure([data_actual, data_predicted, data_future])
fig.update_layout(title='Hasil Prediksi', xaxis_title='Tanggal', yaxis_title='Realisasi MW')

st.plotly_chart(fig)

# Tampilkan tabel hasil prediksi
st.subheader("Hasil Prediksi 7 Hari ke Depan")
st.dataframe(df_future[['ds', 'y_pred_xgb']])

