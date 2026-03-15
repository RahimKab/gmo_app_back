from django.conf import settings
from django.db import models


class Sequence(models.Model):
    SOURCE_RAW = 'raw'
    SOURCE_FASTA = 'fasta'
    SOURCE_FNA = 'fna'
    SOURCE_CHOICES = [
        (SOURCE_RAW, 'Raw Sequence'),
        (SOURCE_FASTA, 'FASTA File'),
        (SOURCE_FNA, 'FNA File'),
    ]

    sequence_id = models.AutoField(primary_key=True)
    seq_id = models.CharField(max_length=100, unique=True)
    seq_nom = models.CharField(max_length=255)
    seq_nom_courant = models.CharField(max_length=255, blank=True)
    sequence = models.TextField()
    sequence_cleaned = models.TextField(blank=True)
    seq_lien = models.URLField(max_length=500, blank=True)
    type_entree = models.CharField(max_length=10, choices=SOURCE_CHOICES)
    seq_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='sequences'
    )

    class Meta:
        ordering = ['-seq_date']

    def __str__(self):
        return f'{self.seq_id} - {self.seq_nom} ({self.seq_date:%Y-%m-%d %H:%M})'
    
class Prediction(models.Model):
    prediction_id = models.AutoField(primary_key=True)
    pred_libelle = models.CharField(max_length=100)
    pred_prob = models.FloatField()
    pred_confiance = models.FloatField()
    pred_date = models.DateTimeField(auto_now_add=True)
    sequence = models.ForeignKey( Sequence, on_delete=models.CASCADE, 
        related_name='predictions'
    )

    class Meta:
        ordering = ['-pred_date']

    def __str__(self):
        return f"{self.pred_libelle} ({self.pred_prob})"
