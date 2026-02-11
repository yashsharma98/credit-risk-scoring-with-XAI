from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    login,
    logout,
)
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.views.decorators.cache import cache_control

from .forms import registration_form
from .ml_utils import predictor
from .models import CreditScore


def landingpgView(request):
    return render(request, "credit_score_app/landingpg.html")


def signupView(request):
    if request.method == "POST":
        form = registration_form(request.POST)
        if form.is_valid():
            if not form.user_exit():
                form.save()
                messages.success(request, "Account created successfully! Please login.")
                return redirect("login")
            else:
                messages.error(request, "Username already exists!")
                return redirect("signup")
    else:
        form = registration_form()

    return render(request, "credit_score_app/signup.html", {"form": form})


def loginView(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        if email and password:
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user=user)
                return redirect("home")
            else:
                messages.error(request, "Invalid email or password")
        else:
            messages.error(request, "Please provide both email and password")

    return render(request, "credit_score_app/login.html")


@cache_control(no_cache=True, must_revalidate=True)
def logoutView(request):
    logout(request)
    request.session.flush()

    next_page = request.GET.get("next")
    if next_page == "signup":
        return redirect("signup")
    else:
        return redirect("login")


@login_required
def homeView(request):
    if not request.user.is_authenticated:
        return redirect("login")

    context = {}

    if request.method == "POST":
        try:
            input_data = {
                "outstanding_debt": float(request.POST.get("outstanding_debt")),
                "interest_rate": float(request.POST.get("interest_rate")),
                "delay_from_due_date": int(request.POST.get("delay_from_due_date")),
                "credit_history_months": int(request.POST.get("credit_history_months")),
                "credit_mix": int(request.POST.get("credit_mix")),
                "changed_credit_limit": float(request.POST.get("changed_credit_limit")),
                "num_credit_inquiries": int(request.POST.get("num_credit_inquiries")),
                "payment_of_min_amount": int(request.POST.get("payment_of_min_amount")),
                "num_bank_accounts": int(request.POST.get("num_bank_accounts")),
                "monthly_balance": float(request.POST.get("monthly_balance")),
                "annual_income": float(request.POST.get("annual_income")),
                "total_emi_per_month": float(request.POST.get("total_emi_per_month")),
            }

            prediction_result = predictor.predict(input_data)

            prediction_result["prob_good_pct"] = prediction_result["prob_good"] * 100
            prediction_result["prob_poor_pct"] = prediction_result["prob_poor"] * 100
            prediction_result["prob_standard_pct"] = prediction_result["prob_standard"] * 100

            shap_plot = predictor.generate_shap_plot(input_data, prediction_result["pred_encoded"])

            shap_breakdown = predictor.get_shap_breakdown(input_data)

            factors = predictor.analyze_factors(input_data)

            ai_explanation = predictor.generate_ai_explanation(input_data, prediction_result)

            CreditScore.objects.create(
                user=request.user,
                **input_data,
                predicted_category=prediction_result["predicted_category"],
                credit_score=prediction_result["credit_score"],
                prob_good=prediction_result["prob_good"],
                prob_poor=prediction_result["prob_poor"],
                prob_standard=prediction_result["prob_standard"],
                ai_explanation=ai_explanation,
            )

            context = {
                "prediction_result": prediction_result,
                "shap_plot": shap_plot,
                "shap_breakdown": shap_breakdown,
                "ai_explanation": ai_explanation,
                "positive_factors": factors["positive"],
                "negative_factors": factors["negative"],
                "input_data": input_data,
            }

        except Exception as e:
            messages.error(request, f"Error making prediction: {str(e)}")
            import traceback

            print(f"Prediction error: {traceback.format_exc()}")

    return render(request, "credit_score_app/home.html", context)


@login_required
def historyView(request):
    predictions = CreditScore.objects.filter(user=request.user).order_by("-created_at")

    results = []
    for pred in predictions:
        input_data = {
            "outstanding_debt": pred.outstanding_debt,
            "interest_rate": pred.interest_rate,
            "delay_from_due_date": pred.delay_from_due_date,
            "credit_history_months": pred.credit_history_months,
            "credit_mix": pred.credit_mix,
            "changed_credit_limit": pred.changed_credit_limit,
            "num_credit_inquiries": pred.num_credit_inquiries,
            "payment_of_min_amount": pred.payment_of_min_amount,
            "num_bank_accounts": pred.num_bank_accounts,
            "monthly_balance": pred.monthly_balance,
            "annual_income": pred.annual_income,
            "total_emi_per_month": pred.total_emi_per_month,
        }

        factors = predictor.analyze_factors(input_data)

        results.append({"prediction": pred, "positive_factors": factors["positive"], "negative_factors": factors["negative"]})

    context = {"results": results, "total_count": predictions.count()}

    return render(request, "credit_score_app/history.html", context)
