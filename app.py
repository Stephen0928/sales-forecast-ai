from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from auto_restock import calculate_restock_quantity
import io

app = Flask(__name__)

def predict_sales(product_name, current_inventory=5000):
    df = pd.read_csv("Sales Dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Product Category'] == product_name]

    if df_filtered.empty:
        return None, None, []

    df_filtered = df_filtered.sort_values('Date').set_index('Date')
    daily_sales = df_filtered['Total Amount'].resample('D').sum()
    recent_history = daily_sales[-30:].tolist()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_sales.values.reshape(-1, 1))

    window_size = 30
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    last_sequence = scaled_data[-window_size:]
    predictions = []
    for _ in range(7):
        seq_input = last_sequence.reshape((1, window_size, 1))
        pred = model.predict(seq_input, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred[0, 0])

    predicted_sales = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    restock_qty = calculate_restock_quantity(predicted_sales, current_inventory)

    return predicted_sales.tolist(), restock_qty, recent_history

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_vals = []
    restock = None
    product_name = None
    historical_vals = []

    df = pd.read_csv("Sales Dataset.csv")
    product_categories = sorted(df['Product Category'].dropna().unique().tolist())

    if request.method == "POST":
        product_name = request.form.get("product_name")
        inventory_str = request.form.get("current_inventory")
        try:
            current_inventory = int(inventory_str)
        except:
            current_inventory = 5000

        predicted_sales, restock_qty, recent_history = predict_sales(product_name, current_inventory)
        if predicted_sales is None:
            prediction = "找不到該商品的銷售資料"
            restock = "-"
        else:
            prediction = [f"Day {i + 1}: {val:.2f}" for i, val in enumerate(predicted_sales)]
            prediction_vals = predicted_sales
            historical_vals = recent_history
            restock = restock_qty

    return render_template("index.html",
                           prediction=prediction,
                           prediction_vals=prediction_vals,
                           restock=restock,
                           product_name=product_name,
                           product_categories=product_categories,
                           historical_vals=historical_vals)

@app.route("/download_csv", methods=["POST"])
def download_csv():
    product_name = request.form.get("product_name")
    current_inventory = int(request.form.get("current_inventory") or 5000)
    predicted_sales, restock_qty, _ = predict_sales(product_name, current_inventory)

    df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(len(predicted_sales))],
        "Predicted Sales": predicted_sales
    })
    df.loc[len(df)] = ["建議補貨量", restock_qty]

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name=f"{product_name}_forecast.csv")

if __name__ == "__main__":
    app.run(debug=True)
