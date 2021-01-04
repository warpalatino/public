
from django.contrib import admin
from django.urls import include, path
from ProTWO_app import views

urlpatterns = [
    path('', views.index, name='index'),
    path('my_model_form/', views.my_model_form, name='my_model_form'),
    path('users/', views.users, name='users'),
]

