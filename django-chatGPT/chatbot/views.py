from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import auth
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.files.storage import default_storage
from .models import Chat, Document

import requests
import json
import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Ollama host (adjust if needed)
OLLAMA_HOST = 'http://localhost:11434'

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))


def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_text_from_docx(file_path):
    """Extract text from DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text


def extract_text_from_txt(file_path):
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def process_document(document_id, user_id):
    """Process document and store in vector database."""
    try:
        document = Document.objects.get(id=document_id, user_id=user_id)
        file_path = document.file.path
        
        # Extract text based on file type
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            return False
        
        if not text.strip():
            return False
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Create or get collection for this user
        collection_name = f"user_{user_id}_docs"
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            collection = chroma_client.create_collection(name=collection_name)
        
        # Generate embeddings and store in ChromaDB
        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"doc_{document_id}_chunk_{idx}"],
                metadatas=[{"document_id": document_id, "document_title": document.title}]
            )
        
        # Mark document as processed
        document.is_processed = True
        document.save()
        
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


def search_documents(user_id, query, top_k=3):
    """Search for relevant document chunks using vector similarity."""
    try:
        collection_name = f"user_{user_id}_docs"
        collection = chroma_client.get_collection(name=collection_name)
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search for similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if results and results['documents']:
            return results['documents'][0]  # Return list of relevant chunks
        return []
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def ask_ollama(message, context_chunks=None):
    """
    Sends a message to Ollama with optional document context.
    """
    # Build the prompt with context if available
    if context_chunks:
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        system_prompt = f"""You are a helpful assistant. Use the following context from the user's documents to answer their question. If the answer cannot be found in the context, say so.

{context_text}"""
    else:
        system_prompt = "You are a helpful assistant."
    
    payload = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.5,
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "Unexpected response format from Ollama.")
    except Exception as e:
        print("Ollama API error:", e)
        return "Sorry, I couldn't process your request."


def chatbot(request):
    """
    Simple chatbot with document Q&A support.
    """
    chats = Chat.objects.filter(user=request.user).order_by('created_at')
    documents = Document.objects.filter(user=request.user).order_by('-uploaded_at')

    if request.method == 'POST':
        message = request.POST.get('message', '').strip()
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        # Check if user has uploaded documents
        has_documents = documents.filter(is_processed=True).exists()
        
        if has_documents:
            # Search for relevant context in documents
            context_chunks = search_documents(request.user.id, message)
            response = ask_ollama(message, context_chunks)
        else:
            # Regular chat without document context
            response = ask_ollama(message)

        # Save chat in DB
        Chat.objects.create(
            user=request.user,
            message=message,
            response=response,
            created_at=timezone.now()
        )

        return JsonResponse({'message': message, 'response': response})

    return render(request, 'chatbot.html', {
        'chats': chats,
        'documents': documents
    })


def upload_document(request):
    """Handle document upload."""
    if request.method == 'POST' and request.FILES.get('document'):
        file = request.FILES['document']
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.name)[1].lower()
        
        if file_ext not in allowed_extensions:
            return JsonResponse({
                'error': 'Invalid file type. Only PDF, DOCX, and TXT files are allowed.'
            }, status=400)
        
        # Save document
        document = Document.objects.create(
            user=request.user,
            title=file.name,
            file=file
        )
        
        # Process document in background (or use Celery for production)
        success = process_document(document.id, request.user.id)
        
        if success:
            return JsonResponse({
                'success': True,
                'message': 'Document uploaded and processed successfully!',
                'document_id': document.id
            })
        else:
            document.delete()
            return JsonResponse({
                'error': 'Failed to process document.'
            }, status=400)
    
    return JsonResponse({'error': 'No file provided'}, status=400)


def delete_document(request, document_id):
    """Delete a document and its embeddings."""
    try:
        document = Document.objects.get(id=document_id, user=request.user)
        
        # Delete from vector database
        collection_name = f"user_{request.user.id}_docs"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            # Delete all chunks related to this document
            collection.delete(
                where={"document_id": document_id}
            )
        except Exception as e:
            print(f"Error deleting from ChromaDB: {e}")
        
        # Delete file and database entry
        if document.file:
            document.file.delete()
        document.delete()
        
        return JsonResponse({'success': True, 'message': 'Document deleted successfully'})
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user:
            auth.login(request, user)
            return redirect('chatbot')
        return render(request, 'login.html', {'error_message': 'Invalid username or password'})
    return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            return render(request, 'register.html', {'error_message': 'Passwords do not match'})

        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
            auth.login(request, user)
            return redirect('chatbot')
        except Exception as e:
            return render(request, 'register.html', {'error_message': f'Error creating account: {e}'})

    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')
