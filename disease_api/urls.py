
from django.urls import path, include
from .views import (
    DiseaseApiView,
)

urlpatterns = [
    path('prediction', DiseaseApiView.as_view()),
]