import base64
import io
import json
import pickle
from pathlib import Path

import matplotlib
import pandas as pd
from django.conf import settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google import genai


class CreditScorePredictor:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.initialized = False
        return cls.instance

    def __init__(self):
        if self.initialized:
            return

        model_dir = Path(settings.BASE_DIR) / "credit_score_app" / "models"

        with open(model_dir / "xgb_credit_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(model_dir / "feature_order.json", "r") as f:
            self.feature_names = json.load(f)

        with open(model_dir / "categorical_mappings.json", "r") as f:
            self.categorical_mappings = json.load(f)

        self.initialized = True

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])[self.feature_names]

        pred_encoded = self.model.predict(input_df)[0]
        pred_proba = self.model.predict_proba(input_df)[0]

        pred_category = self.label_encoder.inverse_transform([pred_encoded])[0]

        score_weights = {"good": 850, "standard": 650, "poor": 300}
        credit_score = 0
        for cls, prob in zip(self.label_encoder.classes_, pred_proba):
            credit_score += prob * score_weights[cls]
        credit_score = int(round(credit_score))

        prob_dict = {}
        for cls, prob in zip(self.label_encoder.classes_, pred_proba):
            prob_dict[f"prob_{cls}"] = float(prob)

        return {
            "predicted_category": pred_category,
            "credit_score": credit_score,
            "confidence": float(pred_proba[pred_encoded]) * 100,
            "prob_good": prob_dict["prob_good"],
            "prob_poor": prob_dict["prob_poor"],
            "prob_standard": prob_dict["prob_standard"],
            "pred_encoded": int(pred_encoded),
        }

    def generate_ai_explanation(self, input_data, prediction_result):
        try:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)

            credit_mix_map = {0: "Bad", 1: "Good", 2: "Standard", 3: "Unknown"}
            payment_min_map = {0: "No", 1: "Unknown", 2: "Yes"}

            prompt = f"""
                Analyze this credit score prediction and provide a clear explanation.

                Financial Profile:
                - Outstanding Debt: ${input_data["outstanding_debt"]:.2f}
                - Interest Rate: {input_data["interest_rate"]:.1f}%
                - Days Late on Payments: {input_data["delay_from_due_date"]} days
                - Credit History: {input_data["credit_history_months"]} months
                - Credit Mix: {credit_mix_map[input_data["credit_mix"]]}
                - Credit Limit Change: {input_data["changed_credit_limit"]:.1f}%
                - Credit Inquiries: {input_data["num_credit_inquiries"]}
                - Pays Minimum Amount: {payment_min_map[input_data["payment_of_min_amount"]]}
                - Bank Accounts: {input_data["num_bank_accounts"]}
                - Monthly Balance: ${input_data["monthly_balance"]:.2f}
                - Annual Income: ${input_data["annual_income"]:.2f}
                - Monthly EMI: ${input_data["total_emi_per_month"]:.2f}

                Prediction: {prediction_result["predicted_category"].upper()}
                Score: {prediction_result["credit_score"]}/850

                Respond strictly in this format:

                Why this score
                1.
                2.
                3.

                Strengths
                1.
                2.
                3.

                Weaknesses
                1.
                2.
                3.

                How to improve
                1.
                2.
                3.

                Keep under 200 words.
                Plain English only.
                No markdown symbols like #, *, **.
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[prompt],
            )

            text = response.text.strip()
            text = text.replace("#", "").replace("*", "")

            return text

        except Exception:
            return "Unable to generate AI explanation at this time."

    def generate_shap_plot(self, input_data, pred_encoded):
        try:
            import shap

            input_df = pd.DataFrame([input_data])[self.feature_names]
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(input_df)

            plt.close("all")
            fig, ax = plt.subplots(figsize=(10, 6))

            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0, :, pred_encoded],
                    base_values=explainer.expected_value[pred_encoded],
                    data=input_df.iloc[0],
                    feature_names=self.feature_names,
                ),
                max_display=12,
                show=False,
            )

            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close("all")

            return f"data:image/png;base64,{image_base64}"
        except Exception:
            plt.close("all")
            return None

    def get_shap_breakdown(self, input_data):
        import shap

        input_df = pd.DataFrame([input_data])[self.feature_names]
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(input_df)

        breakdown = []
        for i, feature in enumerate(self.feature_names):
            breakdown.append(
                {
                    "feature": feature.replace("_", " ").title(),
                    "good": float(shap_values[0, i, 0]),
                    "poor": float(shap_values[0, i, 1]),
                    "standard": float(shap_values[0, i, 2]),
                }
            )

        return breakdown

    def analyze_factors(self, input_data):
        positive_factors = []
        negative_factors = []

        if input_data["credit_history_months"] > 120:
            positive_factors.append("Long credit history")
        if input_data["delay_from_due_date"] < 10:
            positive_factors.append("Timely payments")
        if input_data["payment_of_min_amount"] == 2:
            positive_factors.append("Pays minimum on time")
        if input_data["outstanding_debt"] < 1000:
            positive_factors.append("Low outstanding debt")
        if input_data["num_credit_inquiries"] < 5:
            positive_factors.append("Few credit inquiries")

        if input_data["outstanding_debt"] > 2000:
            negative_factors.append("High outstanding debt")
        if input_data["delay_from_due_date"] > 20:
            negative_factors.append("Frequent payment delays")
        if input_data["num_credit_inquiries"] > 10:
            negative_factors.append("Too many credit inquiries")
        if input_data["payment_of_min_amount"] == 0:
            negative_factors.append("Not paying minimum amount")
        if input_data["interest_rate"] > 25:
            negative_factors.append("High interest rates")

        return {
            "positive": positive_factors if positive_factors else ["No strong positive factors identified"],
            "negative": negative_factors if negative_factors else ["No major concerns identified"],
        }


def prediction():
    return CreditScorePredictor()
