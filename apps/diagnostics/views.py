from uuid import uuid4

from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Prediction, Report, Sequence
from .serializers import (
    SequenceCreateSerializer,
    SequencePredictionResponseSerializer,
    SequenceSerializer,
    UserPredictionDetailSerializer,
    UserPredictionSummarySerializer,
)
from .services.inference import get_inference_service
from .services.preprocess import clean_sequence, extract_sequence_from_text
from .services.report_pdf import build_report_pdf_bytes, save_pdf_report_file


class SequencePredictAPIView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request):
        serializer = SequenceCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        sequence = serializer.validated_data.get('sequence', '')
        seq_nom = serializer.validated_data.get('seq_nom', '').strip()
        seq_nom_courant = serializer.validated_data.get('seq_nom_courant', '').strip()
        uploaded_file = serializer.validated_data.get('file')

        source_type = Sequence.SOURCE_RAW
        input_filename = ''
        sequence_name = 'Manual sequence'

        if uploaded_file:
            input_filename = uploaded_file.name
            lower_name = input_filename.lower()
            source_type = Sequence.SOURCE_FASTA if lower_name.endswith('.fasta') else Sequence.SOURCE_FNA if lower_name.endswith('.fna') else Sequence.SOURCE_RAW
            sequence_name = input_filename.rsplit('.', 1)[0]
            sequence = uploaded_file.read().decode('utf-8', errors='ignore')

        if seq_nom:
            sequence_name = seq_nom

        current_name = seq_nom_courant or input_filename or sequence_name

        extracted_sequence = extract_sequence_from_text(sequence)
        cleaned_sequence = clean_sequence(extracted_sequence)

        if not cleaned_sequence:
            return Response(
                {'detail': 'La sequence est inutilisable.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        prediction = get_inference_service().predict(cleaned_sequence)

        record = Sequence.objects.create(
            seq_id=f'SEQ-{uuid4().hex[:10].upper()}',
            seq_nom=sequence_name,
            seq_nom_courant=current_name,
            sequence=sequence,
            sequence_cleaned=cleaned_sequence,
            seq_lien='',
            type_entree=source_type,
            user=request.user,
        )

        prediction_record = Prediction.objects.create(
            sequence=record,
            pred_libelle=prediction['label'],
            pred_prob=prediction['probabilities']['gmo'],
            pred_confiance=prediction['score'],
        )

        report_payload = {
            'sequence_name': record.seq_nom_courant or record.seq_nom,
            'seq_id': record.seq_id,
            'predicted_label': prediction['label'],
            'score': prediction['score'],
            'gmo_probability': prediction['probabilities']['gmo'],
            'no_gmo_probability': prediction['probabilities']['no_gmo'],
            'chunks_processed': prediction.get('chunks_processed', 1),
            'attention': prediction['attention'],
            'sequence_cleaned': cleaned_sequence,
        }
        pdf_bytes = build_report_pdf_bytes(report_payload)
        file_name = f"report-{record.seq_id}-{prediction_record.prediction_id}.pdf"
        pdf_url = save_pdf_report_file(file_name, pdf_bytes)
        absolute_pdf_url = request.build_absolute_uri(pdf_url)

        Report.objects.create(
            prediction=prediction_record,
            predicted_label=prediction['label'],
            gmo_probability=prediction['probabilities']['gmo'],
            no_gmo_probability=prediction['probabilities']['no_gmo'],
            attention=prediction['attention'],
            chunks_processed=prediction.get('chunks_processed', 1),
            pdf_path=absolute_pdf_url,
        )

        record.seq_lien = absolute_pdf_url
        record.save(update_fields=['seq_lien'])

        response_payload = {
            'sequence': record,
            'prediction': prediction_record,
            'predicted_label': prediction['label'],
            'probabilities': prediction['probabilities'],
            'attention': prediction['attention'],
            'chunks_processed': prediction.get('chunks_processed', 1),
        }

        return Response(
            SequencePredictionResponseSerializer(response_payload).data,
            status=status.HTTP_201_CREATED,
        )


class SequenceListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = Sequence.objects.filter(user=request.user)
        serializer = SequenceSerializer(queryset, many=True)
        return Response(serializer.data)


class UserPredictionListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        predictions = (
            Prediction.objects
            .filter(sequence__user=request.user)
            .select_related('sequence', 'report')
            .order_by('-pred_date')
        )

        payload = [
            {
                'sequence': item.sequence,
                'prediction': item,
                'report': item.report,
            }
            for item in predictions
            if hasattr(item, 'report')
        ]
        serializer = UserPredictionSummarySerializer(payload, many=True)
        return Response(serializer.data)


class UserPredictionLatestDetailAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, sequence_id: int):
        sequence = get_object_or_404(Sequence, sequence_id=sequence_id, user=request.user)
        latest_prediction = (
            Prediction.objects
            .filter(sequence=sequence)
            .select_related('report')
            .order_by('-pred_date')
            .first()
        )

        if latest_prediction is None or not hasattr(latest_prediction, 'report'):
            return Response({'detail': 'No report found for this sequence.'}, status=status.HTTP_404_NOT_FOUND)

        report = latest_prediction.report
        response_payload = {
            'sequence': sequence,
            'prediction': latest_prediction,
            'predicted_label': report.predicted_label,
            'probabilities': {
                'gmo': report.gmo_probability,
                'no_gmo': report.no_gmo_probability,
            },
            'attention': report.attention,
            'chunks_processed': report.chunks_processed,
            'report': report,
        }
        serializer = UserPredictionDetailSerializer(response_payload)
        return Response(serializer.data)
