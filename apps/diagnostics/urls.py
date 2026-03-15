from django.urls import path

from .views import SequenceListAPIView, SequencePredictAPIView

urlpatterns = [
    path('diagnostics/', SequencePredictAPIView.as_view(), name='diagnostic-create'),
    path('diagnostics/list/', SequenceListAPIView.as_view(), name='diagnostic-list'),
]
