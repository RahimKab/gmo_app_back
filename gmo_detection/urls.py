"""
URL configuration for gmo_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import django
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.db import connection
from django.http import JsonResponse
from django.urls import include, path


def health_check(request):
    """Retourne le statut operationnel de l'API et de la base de donnees."""
    db_ok = False
    try:
        connection.ensure_connection()
        db_ok = True
    except Exception: 
        pass

    status = 'ok' if db_ok else 'degraded'
    code = 200 if db_ok else 503

    return JsonResponse(
        {
            'status': status,
            'django': django.get_version(),
            'database': 'connected' if db_ok else 'unreachable',
        },
        status=code,
    )


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/health/', health_check, name='health-check'),
    path('api/', include('apps.accounts.urls')),
    path('api/', include('apps.diagnostics.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
