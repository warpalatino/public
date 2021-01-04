from django.urls import path
from basic_app import views

# piece of news - set up the app name
app_name = 'basic_app'

urlpatterns=[
    path('relative/',views.relative,name='relative'),
    path('other/',views.other,name='other'),
]
