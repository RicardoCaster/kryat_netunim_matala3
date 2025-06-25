from flask import Flask, render_template, request
import pandas as pd
import pickle
from assets_data_prep import prepare_data

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Main route: display the form
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Retrieve the data from the HTML form
        input_data = {
            'property_type': request.form.get("property_type"),
            'neighborhood': request.form.get("neighborhood"),
            'address': request.form.get("address"),
            'room_num': float(request.form.get("room_num" or 0)),
            'floor': float(request.form.get("floor" or 0)),
            'area': float(request.form.get("area" or 0)),
            'garden_area': float(request.form.get("garden_area" or 0)),
            'days_to_enter': float(request.form.get("days_to_enter" or 0)),
            'num_of_payments': float(request.form.get("num_of_payments" or 0)),
            'monthly_arnona': float(request.form.get("monthly_arnona" or 0)),
            'building_tax': float(request.form.get("building_tax" or 0)),
            'total_floors': float(request.form.get("total_floors" or 0)),
            'description': request.form.get("description"),
            'has_parking': 1 if request.form.get("has_parking") == "on" else 0,
            'has_storage': 1 if request.form.get("has_storage") == "on" else 0,
            'elevator': 1 if request.form.get("elevator") == "on" else 0,
            'ac': 1 if request.form.get("ac") == "on" else 0,
            'handicap': 1 if request.form.get("handicap") == "on" else 0,
            'has_bars': 1 if request.form.get("has_bars") == "on" else 0,
            'has_safe_room': 1 if request.form.get("has_safe_room") == "on" else 0,
            'has_balcony': 1 if request.form.get("has_balcony") == "on" else 0,
            'is_furnished': 1 if request.form.get("is_furnished") == "on" else 0,
            'is_renovated': 1 if request.form.get("is_renovated") == "on" else 0,
            'num_of_images': float(request.form.get("num_of_images", 0)),
            'distance_from_center': float(request.form.get("distance_from_center", 0))
        }

        # Convert in dataframe
        df_input = pd.DataFrame([input_data])

        # Prepare the data with  prepare_data (test mode)
        df_prepared = prepare_data(df_input, mode="test")
        
        with open("columns.pkl", "rb") as f:
            expected_columns = pickle.load(f)

        for col in expected_columns:
            if col not in df_prepared.columns:
                df_prepared[col] = 0  # add the missing column with zeros

        df_prepared = df_prepared[expected_columns] # put the columns back in the correct order
        
        df_prepared = scaler.transform(df_prepared)
        
        # Predict the price
        predicted_price = model.predict(df_prepared)[0]

        # Display the result in the HTML
        return render_template("index.html", prediction=round(predicted_price, 2))

    # If GET, just display the page
    return render_template("index.html", prediction=None)

# start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
