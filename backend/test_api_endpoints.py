import unittest
import json
import os
import sys

# Set test client configuration
from app import app, db, User

class TestFitGuideAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_1_health_checks(self):
        print("\n🧪 Running Health Check Tests...")
        # Test general health
        res = self.app.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get('status'), 'healthy')
        print("   ✓ /api/health is online and healthy")
        
        # Test db health
        res = self.app.get('/api/db_health')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data.get('database', {}).get('accessible'))
        print("   ✓ /api/db_health is connected to the SQLite database")

    def test_2_authentication_and_profile(self):
        print("\n🧪 Running Authentication, Profile & ML Engine Tests...")
        # 1. Register a new user
        test_email = "api_test_user@test.com"
        test_username = "ApiTestUser"
        
        # Clean up existing test user if any
        with app.app_context():
            existing = User.query.filter_by(email=test_email).first()
            if existing:
                db.session.delete(existing)
                db.session.commit()
                
        register_payload = {
            "username": test_username,
            "email": test_email,
            "password": "securepassword123",
            "gender": "male",
            "age": 28,
            "height": 175,
            "weight": 75,
            "diet_preference": "vegetarian",
            "activity_level": "active",
            "goals": "weight_loss"
        }
        
        res = self.app.post('/api/register', json=register_payload)
        self.assertEqual(res.status_code, 201)
        data = json.loads(res.data)
        self.assertIn("user_id", data)
        print("   ✓ User registration succeeds (201 Created)")

        # 2. Login
        login_payload = {
            "email": test_email,
            "password": "securepassword123"
        }
        res = self.app.post('/api/login', json=login_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("user", data)
        print("   ✓ User login succeeds (200 OK)")

        # 3. Fetch profile (authenticated session)
        res = self.app.get('/api/user_profile')
        self.assertEqual(res.status_code, 200)
        profile = json.loads(res.data)
        self.assertEqual(profile.get("username"), test_username)
        self.assertEqual(profile.get("diet_preference"), "vegetarian")
        print("   ✓ Retrieving profile succeeds (200 OK)")
        
        # 4. Fetch diet plan (authenticated session)
        res = self.app.get('/api/knn_diet_plan')
        self.assertEqual(res.status_code, 200)
        plan_data = json.loads(res.data)
        self.assertIn("weekly_diet_plan", plan_data)
        print("   ✓ Generating personalized weekly diet plan (KNN) succeeds (200 OK)")

        # 5. Fetch workout recommendations
        res = self.app.get('/api/workout_recommendations')
        self.assertEqual(res.status_code, 200)
        workout_data = json.loads(res.data)
        self.assertIn("workouts", workout_data)
        print("   ✓ Generating workout recommendations (KNN) succeeds (200 OK)")

        # Clean up
        with app.app_context():
            user = User.query.filter_by(email=test_email).first()
            if user:
                db.session.delete(user)
                db.session.commit()
                print("   ✓ Cleaned up test user from the database")

if __name__ == '__main__':
    unittest.main()
