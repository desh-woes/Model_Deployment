import joblib
import pandas as pd
from flask import Flask, request, render_template

# Create flask app
flask_app = Flask(__name__)
model = joblib.load("./random_forest.joblib")

# List of features
list_of_features = ["Age", "Value", "Wage", "Special", "Weight", "Crossing",
                    "Finishing", "HeadingAccuracy", "SprintSpeed", "Reactions", "Stamina",
                    "Aggression", "Composure", "Release Clause"]
feature_dtype_dict = {
    "Value": "float64",
    "Age": "int64",
    "Release Clause": "float64",
    "Reactions": "float64",
    "Wage": "float64",
    "Special": "int64",
    "HeadingAccuracy": "float64",
    "Composure": "float64",
    "Aggression": "float64",
    "Crossing": "float64",
    "Stamina": "float64",
    "Weight": "int64",
    "Finishing": "float64",
    "SprintSpeed": "float64",
}


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    form_input_values = [float(x) for x in request.form.values()]
    feature_value_dict = {}
    for i in range(0, len(form_input_values)):
        feature_value_dict[list_of_features[i]] = form_input_values[i]

    df = pd.DataFrame(feature_value_dict, index=[0])
    for key in feature_dtype_dict:
        df[key].astype(feature_dtype_dict[key])

    prediction = model.predict(df.loc[:, df.columns != "Overall"].values)
    return render_template("index.html", prediction_text="{}".format(int(prediction[0])), code_text=feature_value_dict)


if __name__ == "__main__":
    flask_app.run(debug=True)
