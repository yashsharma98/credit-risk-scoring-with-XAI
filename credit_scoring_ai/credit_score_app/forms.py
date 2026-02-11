from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class registration_form(UserCreationForm):
    fname = forms.CharField(max_length=200, required=True)
    lname = forms.CharField(max_length=200, required=True)
    email = forms.EmailField(max_length=200, required=True)

    class Meta:
        model = User
        fields = ["fname", "lname", "email", "password1", "password2"]

    def user_exit(self):
        email = self.cleaned_data.get("email")
        if email and User.objects.filter(username=email).exists():
            return True
        return False

    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["fname"]
        user.last_name = self.cleaned_data["lname"]
        user.email = self.cleaned_data["email"]

        if commit:
            user.save()
        return user
