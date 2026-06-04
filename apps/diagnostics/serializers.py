from rest_framework import serializers

from .models import Prediction, Report, Sequence


class SequenceCreateSerializer(serializers.Serializer):
    seq_nom = serializers.CharField(required=False, allow_blank=True)
    seq_nom_courant = serializers.CharField(required=False, allow_blank=True)
    sequence = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField(required=False)

    def validate(self, attrs):
        has_sequence = bool((attrs.get('sequence') or '').strip())
        has_file = bool(attrs.get('file'))

        if not has_sequence and not has_file:
            raise serializers.ValidationError('Choisissez une seule source: sequence texte ou fichier.')

        if has_sequence and has_file:
            raise serializers.ValidationError('Vous devez choisir soit la sequence texte, soit le fichier, pas les deux.')

        return attrs


class SequenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sequence
        fields = [
            'sequence_id',
            'seq_id',
            'seq_nom',
            'seq_nom_courant',
            'sequence',
            'sequence_cleaned',
            'seq_lien',
            'type_entree',
            'seq_date',
            'user',
        ]
        read_only_fields = fields


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['prediction_id', 'pred_libelle', 'pred_prob', 'pred_confiance', 'pred_date']
        read_only_fields = fields


class AttentionSegmentSerializer(serializers.Serializer):
    start = serializers.IntegerField()
    end = serializers.IntegerField()
    sequence = serializers.CharField()
    score = serializers.FloatField()


class ProbabilitySerializer(serializers.Serializer):
    gmo = serializers.FloatField()
    no_gmo = serializers.FloatField()


class SequencePredictionResponseSerializer(serializers.Serializer):
    sequence = SequenceSerializer()
    prediction = PredictionSerializer()
    predicted_label = serializers.CharField()
    probabilities = ProbabilitySerializer()
    attention = AttentionSegmentSerializer(many=True)
    chunks_processed = serializers.IntegerField(default=1)


class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = [
            'report_id',
            'predicted_label',
            'gmo_probability',
            'no_gmo_probability',
            'chunks_processed',
            'pdf_path',
            'created_at',
        ]
        read_only_fields = fields


class UserPredictionSummarySerializer(serializers.Serializer):
    sequence = SequenceSerializer()
    prediction = PredictionSerializer()
    report = ReportSerializer()


class UserPredictionDetailSerializer(serializers.Serializer):
    sequence = SequenceSerializer()
    prediction = PredictionSerializer()
    predicted_label = serializers.CharField()
    probabilities = ProbabilitySerializer()
    attention = AttentionSegmentSerializer(many=True)
    chunks_processed = serializers.IntegerField(default=1)
    report = ReportSerializer()
