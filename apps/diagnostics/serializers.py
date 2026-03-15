from rest_framework import serializers

from .models import Sequence


class SequenceCreateSerializer(serializers.ModelSerializer):
    sequence = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField(required=False)

    def validate(self, attrs):
        if not attrs.get('sequence') and not attrs.get('file'):
            raise serializers.ValidationError('Inserer une sequence ou un fichier.')
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
