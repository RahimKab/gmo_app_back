from rest_framework import serializers

from .models import Role, User


class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = ['role_id', 'role_code', 'role_nom']


class UserSerializer(serializers.ModelSerializer):
    role = RoleSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['user_id', 'nom', 'prenom', 'username', 'email', 'user_actif', 'role']


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    role_id = serializers.PrimaryKeyRelatedField(
        queryset=Role.objects.all(), source='role', required=False, allow_null=True
    )

    class Meta:
        model = User
        fields = ['nom', 'prenom', 'username', 'email', 'password', 'role_id']
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user
    