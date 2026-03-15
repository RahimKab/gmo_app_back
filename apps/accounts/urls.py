from django.urls import path

from .views import HomeAPIView, LoginAPIView, MeAPIView, RegisterAPIView

urlpatterns = [
    path('home/', HomeAPIView.as_view(), name='home'),
    path('auth/register/', RegisterAPIView.as_view(), name='register'),
    path('auth/login/', LoginAPIView.as_view(), name='login'),
    path('auth/me/', MeAPIView.as_view(), name='me'),
]
