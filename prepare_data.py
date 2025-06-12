import pandas as pd

# 讀取資料
df = pd.read_csv("Sales Dataset.csv")

# 將日期轉換成 datetime 格式
df['Date'] = pd.to_datetime(df['Date'])

# 以日期為索引，並依照日期排序
df = df.sort_values('Date').set_index('Date')

# 依照日期彙總每日的總銷售額
daily_sales = df['Total Amount'].resample('D').sum()

# 顯示結果
print(daily_sales.head())
