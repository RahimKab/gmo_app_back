from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class Role(models.Model):
    role_id = models.AutoField(primary_key=True)
    role_code = models.CharField(max_length=6, unique=True)
    role_nom = models.CharField(max_length=50, unique=True)


    def __str__(self):
        return self.role_nom


class UserManager(BaseUserManager):
    def create_user(self, username, email, nom, prenom, password=None, **extra_fields):
        if not username:
            raise ValueError('Nom d\'utilisateur requis')
        if not email:
            raise ValueError('Email requis')
        if not nom:
            raise ValueError('Nom requis')
        if not prenom:
            raise ValueError('Prénom requis')   
        if len(password) < 8:
            raise ValueError('Le mot de passe doit contenir au moins 8 caractères')
        
        email = self.normalize_email(email)
        user = self.model(username=username, email=email, nom=nom, prenom=prenom, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, nom, prenom, password=None, **extra_fields):
        extra_fields.setdefault('user_actif', True)
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(username, email, nom, prenom, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    user_id = models.AutoField(primary_key=True)
    nom = models.CharField(max_length=100)
    prenom = models.CharField(max_length=100)
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    user_actif = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    role = models.ForeignKey(Role, on_delete=models.SET_NULL, null=True, blank=True, 
        related_name='users'
    )

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email', 'nom', 'prenom']

    objects = UserManager()

    @property
    def is_active(self):
        return self.user_actif

    @is_active.setter
    def is_active(self, value):
        self.user_actif = value

    def __str__(self):
        return self.username