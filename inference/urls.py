from django.urls import path
from .views import InferenceView

urlpatterns = [
    path('infer/', InferenceView.as_view(), name='infer'),
]
