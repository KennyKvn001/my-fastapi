import requests
import json


def test_api():
    base_url = "http://localhost:8000"

    # Test 1: Health Check
    print("\nTesting Health Check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print("Health Check Response:", response.json())
    except Exception as e:
        print("Health Check Error:", str(e))

    # Test 2: Get Categories
    print("\nTesting Categories endpoint...")
    try:
        response = requests.get(f"{base_url}/categories")
        print("Categories Response:", response.json())
    except Exception as e:
        print("Categories Error:", str(e))

    # Test 3: Prediction
    print("\nTesting Prediction endpoint...")
    test_data = {
        "age": 30,
        "gender": "Male",
        "degree": "Bachelor's",
        "job_title": "UX Researcher",
        "experience": 5,
    }

    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )
        print("Prediction Response:", response.json())
    except Exception as e:
        print("Prediction Error:", str(e))


if __name__ == "__main__":
    test_api()
