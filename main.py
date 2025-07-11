from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
app = Flask(__name__)


model = XGBClassifier()
model.load_model("tuned_xgb.model") 


def preprocess_raw_data(data):
    df_raw = pd.DataFrame(data)
    df = df_raw.copy()

    # Drop unnecessary columns
    if "customerID" in df.columns:
        df = df.drop(columns="customerID")

    # Ordinal Encoding
    contract_order = ["Month-to-month", "One year", "Two year"]
    df["Contract"] = OrdinalEncoder(categories=[contract_order]).fit_transform(df[["Contract"]])

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype(float)

    # Label Encoding
    skip = ["Contract", "TotalCharges"]
    for col in df.columns:
        if col not in skip:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)  

    # Smartcare Features
    df["KQI_Score"] = (
        0.25 * df["TechSupport"] +
        0.25 * df["OnlineSecurity"] +
        0.2 * (df["tenure"] / df["tenure"].max()) +
        0.3 * (1 - df["MonthlyCharges"] / df["MonthlyCharges"].max())
    )
    df["Simulated_Latency"] = 100 - (df["tenure"] * 1.5)
    df["Simulated_PacketLoss"] = (df["MonthlyCharges"] / df["MonthlyCharges"].max()) * 5
    df["SQM_Score"] = (
        0.6 * df["KQI_Score"] +
        0.2 * (1 - df["Simulated_Latency"] / 100) +
        0.2 * (1 - df["Simulated_PacketLoss"] / 5)
    )

    # Scaling
    feature_cols = df.columns.drop("Churn") if "Churn" in df.columns else df.columns
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df[feature_cols]
#++++++++++++++++++++++++++++++++++++++Endpoint++++++++++++++++++++++++++++++++
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json['data']
        prediction = model.predict([input_data])[0]
        return jsonify({'prediction': int(prediction)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.json["data"]
    df_raw = pd.DataFrame(data)
    df_processed = preprocess_raw_data(data)

    df_raw_cleaned = df_raw.loc[df_processed.index].reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)

    preds = model.predict(df_processed)

    enriched = []
    for i, row in df_raw_cleaned.iterrows():
        enriched.append({
            **row.to_dict(),
            "prediction": int(preds[i]),
            "churn": "Yes" if preds[i] == 1 else "No",
            "KQI_Score": round(df_processed.loc[i, "KQI_Score"], 3),
            "SQM_Score": round(df_processed.loc[i, "SQM_Score"], 3)
        })


    return jsonify({ "predictions": enriched })




#run Flask
if __name__ == '__main__':
    app.run(debug=True)