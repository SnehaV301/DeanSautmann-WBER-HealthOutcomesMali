from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import joblib

app = Flask(__name__, template_folder="templates")

# Load the model
best_xgb_loaded = joblib.load("xgboost_model.pkl")


@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("////////// RECEIVED REQUEST //////////")

        feature_names = [
            'TrtOrder', 'LitFA', 'vaccine_card_available', 'health_program_',
            'health_worker_provides_', 'ORT_recipe', 'Months_breastfeeding_correct',
            'tag_HH', 'tag_C', 'tag_M'
        ]

        features = []
        for name in feature_names:
            value = request.form.get(name)
            print(f"Extracting {name}: {value}")
            if value is None:
                raise ValueError(f"Missing form field: {name}")
            features.append(float(value))

        print("////////// FEATURES EXTRACTED //////////")
        print(f"Features: {features}")

        # Convert features to a NumPy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        print(f"Input data shape: {input_data.shape}, values: {input_data}")

        # Predict probability and class
        prediction_proba = best_xgb_loaded.predict_proba(input_data)[0][1]  # Probability of class 1
        prediction = int(prediction_proba > 0.5)

        print("////////// PREDICTION MADE //////////")
        result = {"prediction": prediction, "probability": float(prediction_proba)}
        print(f"Prediction result: {result}")

        return jsonify(result)

    except Exception as e:
        print("////////// ERROR OCCURRED //////////")
        print(f"Error details: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


#..........................................................
# from flask import Flask, request, jsonify, render_template
# import xgboost as xgb
# import numpy as np
#
# app = Flask(__name__, template_folder="templates")
#
# # Load XGBoost model
# model = xgb.Booster()
# model.load_model("xgb_model.json")
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         print("////////////////00000000000000000000////////////////////////")
#         feature_names = [
#             'TrtOrder', 'LitFA', 'vaccine_card_available', 'health_program_',
#             'health_worker_provides_', 'ORT_recipe', 'Months_breastfeeding_correct',
#             'tag_HH', 'tag_C', 'tag_M'
#         ]
#         features = []
#         for name in feature_names:
#             value = request.form.get(name)
#             print(f"Extracting {name}: {value}")
#             if value is None:
#                 raise ValueError(f"Missing form field: {name}")
#             features.append(float(value))
#
#         print("//////////////1111111//////////////////////////")
#         print(f"Features extracted: {features}")
#
#         input_data = np.array(features).reshape(1, -1)
#         print(f"Input data shape: {input_data.shape}, values: {input_data}")
#
#         dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
#         print("/////////////22222///////////////////////////")
#
#         prediction_proba = model.predict(dmatrix)[0]
#         prediction = int(prediction_proba > 0.5)
#         print("/////////////333333///////////////////////////")
#         result = {"prediction": prediction, "probability": float(prediction_proba)}
#         print(f"Prediction result: {result}")
#
#         return jsonify(result)
#
#     except Exception as e:
#         print("///////////eeeeeeeeeeeeeeeee////////////////")
#         print(f"Error details: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True)