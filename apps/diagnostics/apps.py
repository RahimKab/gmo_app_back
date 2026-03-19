from django.apps import AppConfig
from django.conf import settings


class DiagnosticsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.diagnostics'

    def ready(self):
        # Avoid duplicate load in the parent autoreload process during development.
        if settings.DEBUG:
            import os
            import sys

            if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
                return

        from .services.inference import get_inference_service

        get_inference_service().warm_up()
