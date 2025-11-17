"""
Test script for Model Serving API
Run this after starting the API to validate functionality
"""
import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"🔹 {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    if response.ok:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def test_health_check():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.ok


def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print_response("Model Info", response)
    return response.ok


def test_single_prediction():
    """Test single prediction with actual feature names from the model"""
    # Real example with subset of 170 features
    # NOTE: In production, you must provide ALL 170 features
    # This is a minimal example for testing - adjust values as needed
    payload = {
        "features": {
            # Core demographic features
            "NAME_CONTRACT_TYPE": 0,
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "FLAG_OWN_REALTY": 1,
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 450000,
            "AMT_ANNUITY": 24750,
            "NAME_TYPE_SUITE": 0,
            "NAME_INCOME_TYPE": 2,
            "NAME_EDUCATION_TYPE": 0,
            "NAME_FAMILY_STATUS": 2,
            "NAME_HOUSING_TYPE": 1,
            "REGION_POPULATION_RELATIVE": 0.018801,
            "DAYS_BIRTH": -15750,
            "DAYS_EMPLOYED": -1500,
            "DAYS_REGISTRATION": -4500,
            "DAYS_ID_PUBLISH": -3000,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 0,
            "FLAG_EMAIL": 1,
            "OCCUPATION_TYPE": 5,
            "CNT_FAM_MEMBERS": 2,
            "REGION_RATING_CLIENT": 2,
            "WEEKDAY_APPR_PROCESS_START": 3,
            "HOUR_APPR_PROCESS_START": 10,
            "REG_REGION_NOT_LIVE_REGION": 0,
            "REG_REGION_NOT_WORK_REGION": 0,
            "LIVE_REGION_NOT_WORK_REGION": 0,
            "REG_CITY_NOT_LIVE_CITY": 0,
            "REG_CITY_NOT_WORK_CITY": 0,
            "LIVE_CITY_NOT_WORK_CITY": 0,
            "ORGANIZATION_TYPE": 15,
            # External sources (most important features)
            "EXT_SOURCE_2": 0.605876,
            "EXT_SOURCE_3": 0.729567,
            "EXT_SOURCE_MEAN": 0.667722,
            "EXT_SOURCE_MAX": 0.729567,
            "EXT_SOURCE_MIN": 0.605876,
            "EXT_SOURCE_STD": 0.0618455,
            # Engineered features from top 20
            "CREDIT_ANNUITY_YEARS": 18.18,
            "ANNUITY_CREDIT_RATIO": 0.055,
            "ANNUITY_INCOME_RATIO": 0.1375,
            "CREDIT_INCOME_RATIO": 2.5,
            "INCOME_PER_FAMILY_MEMBER": 90000,
            "CREDIT_PER_PERSON": 225000,
            # KNN feature (highly important)
            "KNN_TARGET_MEAN_500": 0.082,
            # Add more features as 0 (will be handled by model)
            # In production, load actual feature list and provide all values
        },
        "application_id": "TEST_001",
        "explain": False
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_response("Single Prediction", response)
    return response.ok


def test_prediction_with_shap():
    """Test prediction with SHAP explanation"""
    payload = {
        "features": {
            "EXT_SOURCE_1": 0.3,
            "EXT_SOURCE_2": 0.4,
            "EXT_SOURCE_3": 0.5,
            "AMT_CREDIT": 600000,
            "AMT_INCOME_TOTAL": 150000,
            "AMT_ANNUITY": 30000,
            "AMT_GOODS_PRICE": 600000,
            "DAYS_BIRTH": -18000,
            "DAYS_EMPLOYED": -500,
            "DAYS_REGISTRATION": -5000,
            "DAYS_ID_PUBLISH": -2500,
            "OWN_CAR_AGE": 10,
            "CNT_CHILDREN": 2,
            "CNT_FAM_MEMBERS": 4,
            "REGION_POPULATION_RELATIVE": 0.025,
            "HOUR_APPR_PROCESS_START": 14
        },
        "application_id": "TEST_002",
        "explain": True
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_response("Prediction with SHAP", response)
    return response.ok


def test_shap_explanation():
    """Test dedicated SHAP explanation endpoint"""
    payload = {
        "features": {
            "EXT_SOURCE_1": 0.2,
            "EXT_SOURCE_2": 0.3,
            "EXT_SOURCE_3": 0.4,
            "AMT_CREDIT": 800000,
            "AMT_INCOME_TOTAL": 120000,
            "AMT_ANNUITY": 35000,
            "AMT_GOODS_PRICE": 800000,
            "DAYS_BIRTH": -20000,
            "DAYS_EMPLOYED": -100,
            "DAYS_REGISTRATION": -6000,
            "DAYS_ID_PUBLISH": -2000,
            "OWN_CAR_AGE": 15,
            "CNT_CHILDREN": 3,
            "CNT_FAM_MEMBERS": 5,
            "REGION_POPULATION_RELATIVE": 0.030,
            "HOUR_APPR_PROCESS_START": 16
        },
        "application_id": "TEST_003",
        "top_n_features": 10
    }
    
    response = requests.post(f"{BASE_URL}/explain", json=payload)
    print_response("SHAP Explanation", response)
    return response.ok


def test_batch_prediction():
    """Test batch prediction"""
    payload = {
        "applications": [
            {
                "features": {
                    "EXT_SOURCE_1": 0.5,
                    "EXT_SOURCE_2": 0.6,
                    "AMT_CREDIT": 400000,
                    "AMT_INCOME_TOTAL": 200000
                },
                "application_id": "BATCH_001"
            },
            {
                "features": {
                    "EXT_SOURCE_1": 0.3,
                    "EXT_SOURCE_2": 0.4,
                    "AMT_CREDIT": 700000,
                    "AMT_INCOME_TOTAL": 150000
                },
                "application_id": "BATCH_002"
            },
            {
                "features": {
                    "EXT_SOURCE_1": 0.7,
                    "EXT_SOURCE_2": 0.8,
                    "AMT_CREDIT": 300000,
                    "AMT_INCOME_TOTAL": 250000
                },
                "application_id": "BATCH_003"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print_response("Batch Prediction", response)
    return response.ok


def test_error_handling():
    """Test error handling with invalid input"""
    # Missing required features
    payload = {
        "features": {
            "INVALID_FEATURE": 123
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_response("Error Handling Test (should fail)", response)
    return not response.ok  # Should fail


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("🚀 Starting Model Serving API Tests")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Prediction with SHAP", test_prediction_with_shap),
        ("SHAP Explanation", test_shap_explanation),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    run_all_tests()