import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running successfully", "status_code": 200}

def test_upload_files():
    files = [
        ("files", ("test.txt", b"Sample text content", "text/plain")),
        ("files", ("test.pdf", b"%PDF-1.4 Sample PDF content", "application/pdf"))
    ]
    response = client.post("/upload-files", files=files)
    assert response.status_code == 200
    assert response.json()["message"] == "Success"

def test_get_user_prompt():
    response = client.post("/ask", json={"user_prompt": "What is AI?"})
    assert response.status_code == 200
    assert response.json()["message"] == "Success"
