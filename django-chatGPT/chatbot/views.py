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
from dotenv import load_dotenv

# ================================
# Load environment variables
# ================================
load_dotenv()

# Use environment variable if provided
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))


# ================================
# Document Text Extraction
# ================================
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text


def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ================================
# Document Processing
# ================================
def process_document(document_id, user_id):
    """Process document and store in vector database."""
    try:
        document = Document.objects.get(id=document_id, user_id=user_id)
        file_path = document.file.path

        # Extract text
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        else:
            return False

        if not text.strip():
            return False

        chunks = chunk_text(text)
        collection_name = f"user_{user_id}_docs"

        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            collection = chroma_client.create_collection(name=collection_name)

        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"doc_{document_id}_chunk_{idx}"],
                metadatas=[{
                    "document_id": document_id,
                    "document_title": document.title
                }]
            )

        document.is_processed = True
        document.save()
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


def search_documents(user_id, query, top_k=3):
    """Search for relevant chunks in user documents."""
    try:
        collection_name = f"user_{user_id}_docs"
        collection = chroma_client.get_collection(name=collection_name)
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        if results and results["documents"]:
            return results["documents"][0]
        return []
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


# ================================
# LLaMA / Ollama Chat Function
# ================================
def ask_ollama(message, context_chunks=None):
    """
    Sends a message to Ollama or open LLaMA API with optional document context.
    Automatically adjusts format depending on the endpoint.
    """
    # Build context if available
    if context_chunks:
        context_text = "\n\n".join(
            [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )
        prompt = f"""You are a helpful assistant. Use the following context to answer:

{context_text}

User question: {message}
"""
    else:
        prompt = message

    # Try Ollama-style JSON first
    payload_ollama = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False
    }

    # Try Open-LLaMA style JSON
    payload_open = {
        "model": "llama3.2:1b",
        "prompt": prompt
    }

    try:
        # First try Ollama format
        response = requests.post(f"{OLLAMA_HOST}/", json=payload_ollama, timeout=60)
        if response.status_code == 422:
            # Retry with open-API format
            response = requests.post(f"{OLLAMA_HOST}/", json=payload_open, timeout=60)

        response.raise_for_status()
        data = response.json()

        # Handle both API styles
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        elif "response" in data:
            return data["response"]
        elif "text" in data:
            return data["text"]
        else:
            return json.dumps(data)

    except Exception as e:
        print("Ollama API error:", e)
        return "Sorry, I couldn't process your request."


# ================================
# Chatbot Views
# ================================
def chatbot(request):
    """Main chatbot page."""
    chats = Chat.objects.filter(user=request.user).order_by("created_at")
    documents = Document.objects.filter(user=request.user).order_by("-uploaded_at")

    if request.method == "POST":
        message = request.POST.get("message", "").strip()
        if not message:
            return JsonResponse({"error": "Empty message"}, status=400)

        has_documents = documents.filter(is_processed=True).exists()

        if has_documents:
            context_chunks = search_documents(request.user.id, message)
            response = ask_ollama(message, context_chunks)
        else:
            response = ask_ollama(message)

        Chat.objects.create(
            user=request.user,
            message=message,
            response=response,
            created_at=timezone.now()
        )

        return JsonResponse({"message": message, "response": response})

    return render(request, "chatbot.html", {"chats": chats, "documents": documents})


# ================================
# Document Upload
# ================================
def upload_document(request):
    if request.method == "POST" and request.FILES.get("document"):
        file = request.FILES["document"]
        allowed_extensions = [".pdf", ".docx", ".txt"]
        file_ext = os.path.splitext(file.name)[1].lower()

        if file_ext not in allowed_extensions:
            return JsonResponse({
                "error": "Invalid file type. Only PDF, DOCX, and TXT files are allowed."
            }, status=400)

        document = Document.objects.create(
            user=request.user,
            title=file.name,
            file=file
        )

        success = process_document(document.id, request.user.id)
        if success:
            return JsonResponse({
                "success": True,
                "message": "Document uploaded and processed successfully!",
                "document_id": document.id
            })
        else:
            document.delete()
            return JsonResponse({"error": "Failed to process document."}, status=400)

    return JsonResponse({"error": "No file provided"}, status=400)


# ================================
# Document Delete
# ================================
def delete_document(request, document_id):
    try:
        document = Document.objects.get(id=document_id, user=request.user)
        collection_name = f"user_{request.user.id}_docs"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            collection.delete(where={"document_id": document_id})
        except Exception as e:
            print(f"Error deleting from ChromaDB: {e}")

        if document.file:
            document.file.delete()
        document.delete()

        return JsonResponse({"success": True, "message": "Document deleted successfully"})
    except Document.DoesNotExist:
        return JsonResponse({"error": "Document not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


# ================================
# Auth Views
# ================================
def login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = auth.authenticate(request, username=username, password=password)
        if user:
            auth.login(request, user)
            return redirect("chatbot")
        return render(request, "login.html", {"error_message": "Invalid username or password"})
    return render(request, "login.html")


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password1 = request.POST["password1"]
        password2 = request.POST["password2"]

        if password1 != password2:
            return render(request, "register.html", {"error_message": "Passwords do not match"})

        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
            auth.login(request, user)
            return redirect("chatbot")
        except Exception as e:
            return render(request, "register.html", {"error_message": f"Error creating account: {e}"})

    return render(request, "register.html")


def logout(request):
    auth.logout(request)
    return redirect("login")
