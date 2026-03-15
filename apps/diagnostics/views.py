from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Sequence
from .serializers import SequenceCreateSerializer, SequenceSerializer
from .services.inference import InferenceService
from .services.preprocess import clean_sequence, extract_sequence_from_text


class SequencePredictAPIView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request):
        serializer = SequenceCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        sequence = serializer.validated_data.get('sequence', '')
        uploaded_file = serializer.validated_data.get('file')

        source_type = Sequence.SOURCE_RAW
        input_filename = ''

        if uploaded_file:
            input_filename = uploaded_file.name
            source_type = Sequence.SOURCE_FASTA if input_filename.endswith('.fasta') else Sequence.SOURCE_FNA
            sequence = uploaded_file.read().decode('utf-8', errors='ignore')

        extracted_sequence = extract_sequence_from_text(sequence)
        cleaned_sequence = clean_sequence(extracted_sequence)

        if not cleaned_sequence:
            return Response(
                {'detail': 'Sequence is empty after cleaning.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        prediction = InferenceService().predict(cleaned_sequence)

        record = Sequence.objects.create(
            user=request.user,
            source_type=source_type,
            input_filename=input_filename,
            raw_sequence=sequence,
            cleaned_sequence=cleaned_sequence,
            prediction_label=prediction['label'],
            prediction_score=prediction['score'],
        )

        return Response(SequenceSerializer(record).data, status=status.HTTP_201_CREATED)


class SequenceListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = Sequence.objects.filter(user=request.user)
        serializer = SequenceSerializer(queryset, many=True)
        return Response(serializer.data)
