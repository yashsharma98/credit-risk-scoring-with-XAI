from django.contrib.auth.models import User
from django.db import models


class CreditScore(models.Model):
    CATEGORY_CHOICES = [
        ("good", "Good"),
        ("standard", "Standard"),
        ("poor", "Poor"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")

    # Input features
    outstanding_debt = models.FloatField()
    interest_rate = models.FloatField()
    delay_from_due_date = models.IntegerField()
    credit_history_months = models.IntegerField()
    credit_mix = models.IntegerField()  # 0=Bad, 1=Good, 2=Standard, 3=Unknown
    changed_credit_limit = models.FloatField()
    num_credit_inquiries = models.IntegerField()
    payment_of_min_amount = models.IntegerField()  # 0=No, 1=Unknown, 2=Yes
    num_bank_accounts = models.IntegerField()
    monthly_balance = models.FloatField()
    annual_income = models.FloatField()
    total_emi_per_month = models.FloatField()

    # Outputs
    predicted_category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)
    credit_score = models.IntegerField()
    prob_good = models.FloatField()
    prob_poor = models.FloatField()
    prob_standard = models.FloatField()

    # AI Explanation
    ai_explanation = models.TextField(blank=True, null=True)

    # SHAP visualization (store image path)
    shap_plot = models.ImageField(upload_to="shap_plots/", blank=True, null=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Credit Scores"

    def __str__(self):
        return f"{self.user.username} - {self.predicted_category}"

    @property
    def credit_mix_label(self):
        mapping = {0: "Bad", 1: "Good", 2: "Standard", 3: "Unknown"}
        return mapping.get(self.credit_mix, "Unknown")

    @property
    def payment_min_label(self):
        mapping = {0: "No", 1: "Unknown", 2: "Yes"}
        return mapping.get(self.payment_of_min_amount, "Unknown")

    @property
    def prob_good_pct(self):
        return round(self.prob_good * 100, 1)

    @property
    def prob_standard_pct(self):
        return round(self.prob_standard * 100, 1)

    @property
    def prob_poor_pct(self):
        return round(self.prob_poor * 100, 1)
