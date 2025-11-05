from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout, name='logout'),
    path('upload-document/', views.upload_document, name='upload_document'),
    path('delete-document/<int:document_id>/', views.delete_document, name='delete_document'),
]