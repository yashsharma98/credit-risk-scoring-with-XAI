from django.urls import path

from .views import historyView, homeView, landingpgView, loginView, logoutView, signupView

urlpatterns = [
    path("", landingpgView, name="landingpg"),
    path("login/", loginView, name="login"),
    path("signup/", signupView, name="signup"),
    path("logout/", logoutView, name="logout"),
    path("home/", homeView, name="home"),
    path("history/", historyView, name="history"),
]
