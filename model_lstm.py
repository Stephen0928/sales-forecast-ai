import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from auto_restock import calculate_restock_quantity  # 匯入補貨計算函式

# 讀取資料並整理
df = pd.read_csv("Sales Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')
daily_sales = df['Total Amount'].resample('D').sum()

# 正規化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_sales.values.reshape(-1, 1))

# 建立序列資料（用過去30天預測第31天）
X, y = [], []
window_size = 30
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# 模型架構
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 訓練
model.fit(X, y, epochs=20, batch_size=16)

# 預測未來7天
last_sequence = scaled_data[-window_size:]
predictions = []

for _ in range(7):
    seq_input = last_sequence.reshape((1, window_size, 1))
    pred = model.predict(seq_input, verbose=0)
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0])

# 還原預測數值
predicted_sales = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 顯示結果
for i, val in enumerate(predicted_sales, 1):
    print(f"Day {i}: 預測銷售額 = {val[0]:.2f}")

# 把 numpy array 轉成 list，方便補貨函式使用
predicted_sales_list = predicted_sales.flatten().tolist()

# 設定現有庫存（你可以改成真實庫存數字）
current_inventory = 5000

# 計算建議補貨數量
restock_qty = calculate_restock_quantity(predicted_sales_list, current_inventory)

print(f"建議補貨量為: {restock_qty} 件")