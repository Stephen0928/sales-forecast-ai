# 假設你有一個 list，裡面放著未來7天預測銷售額
predicted_sales = [1189.00, 1198.68, 1202.55, 1203.06, 1201.88, 1199.94, 1198.10]

# 假設目前庫存量
current_inventory = 5000

# 設定安全庫存 (safety stock)，避免缺貨
safety_stock = 1000

# 計算一週預計銷售量
total_forecast = sum(predicted_sales)

# 計算應補貨量 = 預計銷售量 + 安全庫存 - 目前庫存
restock_quantity = total_forecast + safety_stock - current_inventory

# 補貨量不能小於0
restock_quantity = max(0, int(restock_quantity))

print(f"建議補貨量為: {restock_quantity} 件")
