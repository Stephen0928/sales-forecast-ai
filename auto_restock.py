def calculate_restock_quantity(predicted_sales, current_inventory):
    total_predicted = sum(predicted_sales)
    restock_qty = max(0, int(total_predicted - current_inventory))
    return restock_qty