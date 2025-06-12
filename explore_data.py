import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("Sales Dataset.csv")

# 顯示前幾筆資料
print(df.head())

# 查看欄位資訊
print("\n欄位資料型別：")
print(df.dtypes)

# 檢查是否有缺漏值
print("\n缺漏值統計：")
print(df.isnull().sum())
