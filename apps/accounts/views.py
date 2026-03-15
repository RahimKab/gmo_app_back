from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import User
from .serializers import RegisterSerializer, UserSerializer


class HomeAPIView(APIView):
	permission_classes = [AllowAny]

	def get(self, request):
		return Response(
			{
				'message': 'GMO Detection API is running',
				'date': '2026-03-14',
			}
		)


class RegisterAPIView(APIView):
	permission_classes = [AllowAny]

	def post(self, request):
		serializer = RegisterSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		user = serializer.save()
		token, _ = Token.objects.get_or_create(user=user)
		print('ici')
		return Response(
			{
				'token': token.key,
				'user': UserSerializer(user).data,
			},
			status=status.HTTP_201_CREATED,
		)


class LoginAPIView(APIView):
	permission_classes = [AllowAny]

	def post(self, request):
		email = request.data.get('email', '').strip()
		password = request.data.get('password', '')

		if not email or not password:
			return Response(
				{'detail': 'Email et mot de passe sont requis.'},
				status=status.HTTP_400_BAD_REQUEST,
			)

		try:
			user_obj = User.objects.get(email=email)
		except User.DoesNotExist:
			return Response(
				{'detail': 'Identifiants invalides.'},
				status=status.HTTP_400_BAD_REQUEST,
			)

		user = authenticate(request, username=user_obj.username, password=password)
		if not user:
			return Response(
				{'detail': 'Identifiants invalides.'},
				status=status.HTTP_400_BAD_REQUEST,
			)

		token, _ = Token.objects.get_or_create(user=user)
		return Response(
			{
				'token': token.key,
				'user': UserSerializer(user).data,
			}
		)


class MeAPIView(APIView):
	permission_classes = [IsAuthenticated]

	def get(self, request):
		return Response(UserSerializer(request.user).data)
