from django.urls import path

from .views import (
    SequenceListAPIView,
    SequencePredictAPIView,
    UserPredictionLatestDetailAPIView,
    UserPredictionListAPIView,
)

urlpatterns = [
    path('diagnostics/', SequencePredictAPIView.as_view(), name='diagnostic-create'),
    path('diagnostics/list/', SequenceListAPIView.as_view(), name='diagnostic-list'),
    path('reports/', UserPredictionListAPIView.as_view(), name='report-list'),
    path('reports/<int:sequence_id>/latest/', UserPredictionLatestDetailAPIView.as_view(), name='report-latest'),
]
