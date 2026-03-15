from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import Role, User


@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ['role_id', 'role_code', 'role_nom']


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['user_id', 'username', 'email', 'nom', 'prenom', 'user_actif', 'role']
    list_filter = ['is_staff', 'is_superuser', 'user_actif']
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Informations', {'fields': ('nom', 'prenom', 'email', 'role', 'user_actif')}),
        ('Permissions', {'fields': ('is_staff', 'is_superuser', 'groups', 'user_permissions')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'nom', 'prenom', 'role', 'password1', 'password2'),
        }),
    )