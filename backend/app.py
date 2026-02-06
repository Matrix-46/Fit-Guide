# backend/app.py

# Load environment variables from .env file (for local development and production)
from dotenv import load_dotenv
load_dotenv()  # Load .env file if it exists

from flask import Flask, request, jsonify, session
import json
import os
import re
import random
import logging
from datetime import date, datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user, login_required, current_user)
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- App Configuration ---
app = Flask(__name__)

# IMMEDIATELY Trust Render Proxy headers (X-Forwarded-Proto, X-Forwarded-For)
# Render uses 1 proxy level.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_for=1, x_host=1, x_port=1)

# Configure Log Level IMMEDIATELY
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# --- CORS Configuration ---
frontend_url = os.environ.get('FRONTEND_URL')
cors_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://fitguide-frontend-g06v.onrender.com"
]
if frontend_url:
    cors_origins.append(frontend_url)

CORS(app,
     resources={r"/api/.*": {"origins": cors_origins}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=True,
     expose_headers=["Content-Length", "X-CSRFToken"])

@app.before_request
def log_incoming_request():
    app.logger.debug(f"--- Incoming Request: {request.method} {request.path} (Origin: {request.headers.get('Origin')}) ---")

@app.errorhandler(Exception)
def handle_error(e):
    app.logger.error(f"Global Error Handler caught: {e}", exc_info=True)
    response = jsonify({"message": str(e) or "An internal error occurred."})
    response.status_code = 500
    return response

# Secret Key Configuration
secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    # Use a stable but semi-secure fallback based on the app path to prevent session loss on every restart
    import hashlib
    secret_key = hashlib.sha256(os.path.abspath(__file__).encode()).hexdigest()
    app.logger.warning("⚠️ FLASK_SECRET_KEY not set! Using stable fallback key. Please set it in Render for security!")
app.config['SECRET_KEY'] = secret_key
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'None'
app.config['REMEMBER_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_NAME'] = 'fitguide_session'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
try:
    os.makedirs(instance_path, exist_ok=True)
except OSError: pass

# Database configuration
db_path = os.path.join(instance_path, 'fit_guide.db')
db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or f'sqlite:///{db_path}'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.session_protection = None # Loosened for Render load balancer compatibility

@app.before_request
def debug_request_info():
    app.logger.debug(f"--- Request: {request.method} {request.path} ---")
    app.logger.debug(f"Secure: {request.is_secure}, Origin: {request.headers.get('Origin')}")
    app.logger.debug(f"Cookies: {list(request.cookies.keys())}")

@app.after_request
def finalize_response(response):
    origin = request.headers.get('Origin')
    # Use dynamic regex matching for .onrender.com subdomains
    if origin and ('.onrender.com' in origin or 'localhost' in origin or '127.0.0.1' in origin):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        
        # Aggressively ensure cookies are cross-site compatible
        cookies = response.headers.getlist('Set-Cookie')
        if cookies:
            response.headers.remove('Set-Cookie')
            for cookie in cookies:
                new_cookie = cookie
                if 'SameSite=' not in new_cookie:
                    new_cookie += '; SameSite=None'
                else:
                    new_cookie = re.sub(r'SameSite=\w+', 'SameSite=None', new_cookie)
                
                if 'Secure' not in new_cookie:
                    new_cookie += '; Secure'
                
                response.headers.add('Set-Cookie', new_cookie)

    if request.method == 'OPTIONS':
        response.status_code = 204
        
    app.logger.debug(f"--- Response: {response.status_code} ---")
    return response

# --- Dataset Loading ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIET_DATASET_PATH = os.path.join(base_dir, 'datasets', 'diet_dataset_1000.csv')
FITNESS_DATASET_PATH = os.path.join(base_dir, 'datasets', 'fitness_dataset_1000.csv')
diet_df = None
fitness_df = None

try:
    diet_df = pd.read_csv(DIET_DATASET_PATH)
    if diet_df is not None: app.logger.info(f"Diet dataset loaded. Shape: {diet_df.shape}. Columns: {diet_df.columns.tolist()}")
    else: app.logger.warning("Diet dataset: pd.read_csv returned None. Using empty DataFrame."); diet_df = pd.DataFrame()
except FileNotFoundError: app.logger.error(f"CRITICAL ERROR: Diet dataset not found at {DIET_DATASET_PATH}"); diet_df = pd.DataFrame()
except Exception as e: app.logger.error(f"CRITICAL ERROR loading diet dataset from {DIET_DATASET_PATH}: {e}"); diet_df = pd.DataFrame()

try:
    fitness_df = pd.read_csv(FITNESS_DATASET_PATH)
    if fitness_df is not None: app.logger.info(f"Fitness dataset loaded. Shape: {fitness_df.shape}. Columns: {fitness_df.columns.tolist()}")
    else: app.logger.warning("Fitness dataset: pd.read_csv returned None. Using empty DataFrame."); fitness_df = pd.DataFrame()
except FileNotFoundError: app.logger.error(f"WARNING: Fitness dataset not found at {FITNESS_DATASET_PATH}"); fitness_df = pd.DataFrame()
except Exception as e: app.logger.error(f"ERROR loading fitness dataset from {FITNESS_DATASET_PATH}: {e}"); fitness_df = pd.DataFrame()

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False) # Display name (not used for login)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True) # Email used for login
    password_hash = db.Column(db.String(128), nullable=False)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    diet_preference = db.Column(db.String(50))
    activity_level = db.Column(db.String(50))
    goals = db.Column(db.String(100))
    preferred_cuisines = db.Column(db.String(200), nullable=True)
    is_admin_user = db.Column(db.Boolean, default=False, nullable=False) # Admin status
    is_superadmin = db.Column(db.Boolean, default=False, nullable=False) # Superadmin - cannot be deleted/modified by other admins

    def set_password(self, password): self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    def check_password(self, password): return bcrypt.check_password_hash(self.password_hash, password)

class WorkoutLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_workoutlog_user_id'), nullable=False, index=True) # Added index and explicit FK name
    log_date = db.Column(db.Date, nullable=False, default=date.today)
    exercise_name = db.Column(db.String(100), nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    # New column to store serialized pose/rep metrics (JSON string)
    pose_data = db.Column(db.Text, nullable=True)
    user = db.relationship('User', backref=db.backref('workout_logs', lazy='dynamic'))

class DietLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_dietlog_user_id'), nullable=False, index=True)
    log_date = db.Column(db.Date, nullable=False, default=date.today)
    meal_type = db.Column(db.String(50))
    food_items = db.Column(db.Text) # Consider JSON or separate table for structured items
    total_calories = db.Column(db.Integer)
    user = db.relationship('User', backref=db.backref('diet_logs', lazy='dynamic'))

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_todo_user_id'), nullable=False, index=True)
    task = db.Column(db.String(250), nullable=False)
    completed = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('todos', lazy='dynamic'))

@login_manager.user_loader
def load_user(user_id): return User.query.get(int(user_id))

# --- Helper Functions (BMI, BMR, TDEE, Target Calories) ---
def calculate_bmi(weight_kg, height_cm):
    if not weight_kg or not height_cm or height_cm == 0: return 0.0
    height_m = height_cm / 100; bmi = weight_kg / (height_m ** 2); return round(bmi, 1)
def get_bmi_category(bmi):
    if bmi == 0: return "N/A"
    if bmi < 18.5: return "Underweight"
    elif 18.5 <= bmi < 24.9: return "Normal weight"
    elif 25 <= bmi < 29.9: return "Overweight"
    else: return "Obesity"
def calculate_bmr(weight_kg, height_cm, age, gender):
    if not all([weight_kg, height_cm, age, gender]): return 0
    gender_l = gender.lower() if gender else ""
    if gender_l == 'male': bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    elif gender_l == 'female': bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    else: bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 78 # Average for 'other'
    return round(bmr)
def calculate_tdee(bmr, activity_level):
    if bmr == 0: return 0
    activity_multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725, 'very_active': 1.9}
    multiplier = activity_multipliers.get(activity_level.lower() if activity_level else 'sedentary', 1.2)
    return round(bmr * multiplier)
def get_target_calories(tdee, goal):
    if tdee == 0: return 0
    goal_lower = goal.lower() if goal else 'maintenance'
    if goal_lower == 'weight_loss': return tdee - 500
    elif goal_lower == 'muscle_gain': return tdee + 300
    return tdee


# --- Workout Recommendation Logic ---
def recommend_workouts_logic(user_goal, bmi_category, user_profile=None):
    # Expanded list of workouts
    all_workouts_full = [
        {"name": "Push-ups", "type": "strength", "target": "Chest, Shoulders, Triceps", "duration_suggestion": "3 sets of AMRAP", "difficulty": "Intermediate"},
        {"name": "Squats (Bodyweight)", "type": "strength", "target": "Quads, Glutes, Hamstrings", "duration_suggestion": "3 sets of 12-15 reps", "difficulty": "Beginner"},
        {"name": "Plank", "type": "core", "target": "Core", "duration_suggestion": "3 sets, hold 30-60s", "difficulty": "Beginner"},
        {"name": "Lunges (Bodyweight)", "type": "strength", "target": "Quads, Glutes", "duration_suggestion": "3 sets of 10-12 reps per leg", "difficulty": "Beginner"},
        {"name": "Burpees", "type": "cardio", "target": "Full Body", "duration_suggestion": "3 sets of 8-12 reps", "difficulty": "Intermediate"},
        {"name": "Jumping Jacks", "type": "cardio", "target": "Full Body", "duration_suggestion": "3-5 minutes", "difficulty": "Beginner"},
        {"name": "Running/Jogging (Moderate Pace)", "type": "cardio", "target": "Cardiovascular, Legs", "duration_suggestion": "20-30 minutes", "difficulty": "Beginner-Intermediate"},
        {"name": "Cycling (Moderate Intensity)", "type": "cardio", "target": "Legs, Cardiovascular", "duration_suggestion": "30-45 minutes", "difficulty": "Beginner-Intermediate"},
        {"name": "Bicep Curls (Dumbbells/Resistance Band)", "type": "strength", "target": "Biceps", "duration_suggestion": "3 sets of 10-15 reps", "difficulty": "Beginner"},
        {"name": "Overhead Press (Dumbbells/Resistance Band)", "type": "strength", "target": "Shoulders, Triceps", "duration_suggestion": "3 sets of 10-15 reps", "difficulty": "Beginner"},
        {"name": "Bent-Over Rows (Dumbbells/Resistance Band)", "type": "strength", "target": "Back, Biceps", "duration_suggestion": "3 sets of 10-15 reps", "difficulty": "Beginner"},
        {"name": "Crunches", "type": "core", "target": "Upper Abs", "duration_suggestion": "3 sets of 15-20 reps", "difficulty": "Beginner"},
        {"name": "Leg Raises (Lying)", "type": "core", "target": "Lower Abs", "duration_suggestion": "3 sets of 15-20 reps", "difficulty": "Beginner"},
        {"name": "Bird-Dog", "type": "core", "target": "Core Stability, Back", "duration_suggestion": "3 sets of 10-12 reps per side", "difficulty": "Beginner"},
        {"name": "Glute Bridges", "type": "strength", "target": "Glutes, Hamstrings", "duration_suggestion": "3 sets of 15-20 reps", "difficulty": "Beginner"},
        {"name": "Yoga Flow (Beginner)", "type": "flexibility", "target": "Full Body", "duration_suggestion": "20-30 minutes", "difficulty": "Beginner"},
        {"name": "Stretching Routine", "type": "flexibility", "target": "Major Muscle Groups", "duration_suggestion": "10-15 minutes post-workout", "difficulty": "Beginner"},
        {"name": "High-Intensity Interval Training (HIIT) - Bodyweight", "type": "cardio", "target": "Full Body, Fat Loss", "duration_suggestion": "15-20 mins (e.g., 30s work, 30s rest)", "difficulty": "Intermediate"},
        {"name": "Walking (Brisk)", "type": "cardio", "target": "General Fitness", "duration_suggestion": "30-60 minutes", "difficulty": "Beginner"},
    ]
    recommendations = []; goal_lower = user_goal.lower() if user_goal else "maintenance"

    # Simplified selection logic for brevity, can be made more sophisticated
    if goal_lower == 'muscle_gain':
        strength_workouts = [w for w in all_workouts_full if w['type'] == 'strength']
        core_workouts = [w for w in all_workouts_full if w['type'] == 'core']
        if strength_workouts: recommendations.extend(random.sample(strength_workouts, k=min(3, len(strength_workouts))))
        if core_workouts: recommendations.extend(random.sample(core_workouts, k=min(2, len(core_workouts))))
    elif goal_lower == 'weight_loss':
        cardio_workouts = [w for w in all_workouts_full if w['type'] == 'cardio']
        strength_core_workouts = [w for w in all_workouts_full if w['type'] in ['strength', 'core']]
        if cardio_workouts: recommendations.extend(random.sample(cardio_workouts, k=min(2, len(cardio_workouts))))
        if strength_core_workouts: recommendations.extend(random.sample(strength_core_workouts, k=min(3, len(strength_core_workouts))))
    elif goal_lower == 'endurance':
        cardio_workouts = [w for w in all_workouts_full if w['type'] == 'cardio']
        if cardio_workouts: recommendations.extend(random.sample(cardio_workouts, k=min(4, len(cardio_workouts))))
        # Add a core workout for endurance
        core_w = next((w for w in all_workouts_full if w['type'] == 'core'), None)
        if core_w and len(recommendations) < 5 : recommendations.append(core_w)
    else: # Maintenance or other
        if all_workouts_full: recommendations = random.sample(all_workouts_full, k=min(5, len(all_workouts_full)))

    # Ensure uniqueness and limit to 5
    seen_names = set(); unique_recs = []
    for rec in recommendations:
        if rec['name'] not in seen_names:
            unique_recs.append(rec)
            seen_names.add(rec['name'])

    final_recs = unique_recs[:5] # Take up to 5 unique recommendations

    # If less than 3, try to add some general ones to reach at least 3 if possible
    if len(final_recs) < 3 and len(all_workouts_full) >=3:
        general_fill = [w for w in all_workouts_full if w['name'] not in seen_names]
        if general_fill:
            final_recs.extend(random.sample(general_fill, k=min(3 - len(final_recs), len(general_fill))))

    # Calculate estimated calories for each recommendation
    if user_profile:
        for rec in final_recs:
            # Parse duration from suggestion (e.g. "30-45 minutes" -> 37)
            duration = 30 # Default
            try:
                dur_str = rec['duration_suggestion']
                # Simple extraction of numbers
                import re
                nums = [int(n) for n in re.findall(r'\d+', dur_str)]
                if nums:
                    duration = sum(nums) / len(nums) # Average of found numbers
            except:
                duration = 30
            
            estimated_cal = predict_calories_burned(
                user_profile['age'], 
                user_profile['gender'], 
                user_profile['height'], 
                user_profile['weight'], 
                rec['type'], 
                duration
            )
            rec['estimated_calories'] = estimated_cal
    
    return final_recs


# --- KNN-Based Calorie Prediction ---
def predict_calories_burned(age, gender, height_cm, weight_kg, workout_type, duration_minutes):
    """
    Use KNN to predict calories burned based on user profile and workout type.
    Uses the fitness_dataset_1000.csv to find similar users and estimate calorie burn.
    """
    global fitness_df
    
    if fitness_df is None or fitness_df.empty:
        # Fallback: simple estimation based on MET values
        met_values = {
            'strength': 5.0, 'cardio': 7.0, 'hiit': 8.0, 'yoga': 3.0, 
            'pilates': 3.5, 'flexibility': 2.5, 'core': 4.0
        }
        met = met_values.get(workout_type.lower(), 4.0)
        # Calories = MET × weight (kg) × duration (hours)
        estimated = int(met * weight_kg * (duration_minutes / 60))
        return estimated
    
    # Map app workout types to dataset workout types
    workout_type_mapping = {
        'strength': 'Strength',
        'cardio': 'Cardio',
        'hiit': 'HIIT',
        'yoga': 'Yoga',
        'pilates': 'Pilates',
        'flexibility': 'Yoga',  # Map flexibility to Yoga
        'core': 'Strength'       # Map core to Strength
    }
    
    dataset_workout_type = workout_type_mapping.get(workout_type.lower(), 'Cardio')
    
    # Filter dataset by workout type for more accurate predictions
    workout_filtered_df = fitness_df[fitness_df['Workout_type'] == dataset_workout_type].copy()
    
    if workout_filtered_df.empty or len(workout_filtered_df) < 5:
        # Fall back to all workouts if not enough samples
        workout_filtered_df = fitness_df.copy()
    
    try:
        # Prepare features: Age, Height, Weight, Duration
        feature_cols = ['Age', 'Height_cm', 'Weight_kg', 'Duration_min']
        X = workout_filtered_df[feature_cols].values
        y = workout_filtered_df['Calories_burned'].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create user feature vector (map gender isn't needed as dataset doesn't use it well for calories)
        user_features = np.array([[age, height_cm, weight_kg, duration_minutes]])
        user_scaled = scaler.transform(user_features)
        
        # Use KNN Regression approach (using KNeighborsClassifier but averaging target values)
        n_neighbors = min(5, len(workout_filtered_df))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        
        # Create dummy classes (we'll use distances to get neighbors)
        dummy_classes = np.arange(len(workout_filtered_df))
        knn.fit(X_scaled, dummy_classes)
        
        # Get distances and indices of nearest neighbors
        distances, indices = knn.kneighbors(user_scaled, n_neighbors=n_neighbors)
        
        # Weight the calories by inverse distance
        neighbor_calories = y[indices[0]]
        
        if np.all(distances[0] == 0):
            # Exact match found, take average
            estimated = int(np.mean(neighbor_calories))
        else:
            # Weight by inverse distance
            weights = 1 / (distances[0] + 0.001)  # Add small value to avoid division by zero
            estimated = int(np.average(neighbor_calories, weights=weights))
        
        return max(50, min(estimated, 1500))  # Clamp to reasonable range
        
    except Exception as e:
        app.logger.warning(f"KNN calorie prediction failed: {e}. Using fallback.")
        # Fallback: simple MET-based estimation
        met_values = {
            'strength': 5.0, 'cardio': 7.0, 'hiit': 8.0, 'yoga': 3.0, 
            'pilates': 3.5, 'flexibility': 2.5, 'core': 4.0
        }
        met = met_values.get(workout_type.lower(), 4.0)
        estimated = int(met * weight_kg * (duration_minutes / 60))
        return max(50, min(estimated, 1500))


# --- API Routes ---
@app.route('/')
def api_root_info(): return jsonify({"message": "Welcome to the Fit-Guide API! Backend is running."})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data: return jsonify({"message": "No input data provided"}), 400
    required_fields = ['username','email', 'password', 'gender', 'age', 'height', 'weight', 'diet_preference', 'activity_level', 'goals']
    missing = [field for field in required_fields if not data.get(field)]
    if missing: return jsonify({"message": f"Missing required fields: {', '.join(missing)}"}), 400

    username = data['username'].strip()
    email = data['email'].strip().lower()
    if not username: return jsonify({"message": "Username cannot be empty"}), 400
    if not email: return jsonify({"message": "Email cannot be empty"}), 400
    if User.query.filter_by(email=email).first(): return jsonify({"message": "Email already registered"}), 409

    try:
        age = int(data['age']); height = float(data['height']); weight = float(data['weight'])
        if not (12 <= age <= 120): raise ValueError("Age must be between 12 and 120.")
        if not (50 <= height <= 300): raise ValueError("Height must be between 50 and 300 cm.")
        if not (20 <= weight <= 500): raise ValueError("Weight must be between 20 and 500 kg.")
    except (ValueError, TypeError) as e: return jsonify({"message": f"Invalid data for age, height, or weight: {str(e)}"}), 400

    if len(data['password']) < 6:
        return jsonify({"message": "Password must be at least 6 characters long."}), 400

    preferred_cuisines_value = data.get('preferred_cuisines', "").strip()

    new_user = User(
        username=username, email=email, gender=data['gender'], age=age,
        height_cm=height, weight_kg=weight, diet_preference=data['diet_preference'],
        activity_level=data['activity_level'], goals=data['goals'],
        preferred_cuisines=preferred_cuisines_value,
        is_admin_user=False # Regular users are not admin by default
    )
    new_user.set_password(data['password'])

    try:
        db.session.add(new_user); db.session.commit()
        app.logger.info(f"User '{username}' registered successfully.")
        return jsonify({"message": "User registered successfully!"}), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Database error during registration for {username}: {e}", exc_info=True)
        return jsonify({"message": "Registration failed due to a server error."}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data: return jsonify({"message": "No input data provided"}), 400
    email = data.get('email')
    password = data.get('password')
    if not email or not password: return jsonify({"message": "Email and password are required"}), 400

    email = email.strip().lower()

    # Admin Check using environment variables (for security)
    admin_email = os.environ.get('ADMIN_EMAIL')
    admin_password = os.environ.get('ADMIN_PASSWORD')
    
    # Only process admin login if admin credentials are configured
    if admin_email and admin_password and email == admin_email:
        # Authenticate admin
        admin_user = User.query.filter_by(email=email).first()
        
        # Create admin user if doesn't exist
        if not admin_user:
            admin_user = User(
                username=os.environ.get('ADMIN_USERNAME', 'Admin'),
                email=admin_email,
                is_admin_user=True,
                gender="other",
                age=30,
                height_cm=160,
                weight_kg=60,
                diet_preference="any",
                activity_level="moderate",
                goals="maintenance"
            )
            admin_user.set_password(admin_password)
            try:
                db.session.add(admin_user)
                db.session.commit()
                app.logger.info(f"Admin user '{admin_user.username}' created successfully.")
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Could not create admin user: {e}")
                return jsonify({"message": "Admin setup error."}), 500
        
        # Verify password
        if not admin_user.check_password(password):
            app.logger.warning(f"Failed admin login attempt for email: {email} (password mismatch)")
            return jsonify({"message": "Invalid email or password"}), 401

        # Ensure admin flag is set
        if not admin_user.is_admin_user:
            admin_user.is_admin_user = True
            db.session.commit()

        login_user(admin_user, remember=True, duration=timedelta(days=7))
        app.logger.info(f"Admin User '{admin_user.username}' logged in successfully.")
        return jsonify({
            "message": "Admin login successful!",
            "user": {"id": admin_user.id, "username": admin_user.username, "email": admin_user.email, "is_admin": True }
        }), 200

    # Regular user login by email
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user, remember=True, duration=timedelta(days=7))
        app.logger.info(f"User '{user.username}' logged in successfully.")
        return jsonify({
            "message": "Login successful!",
            "user": {"id": user.id, "username": user.username, "email": user.email, "is_admin": user.is_admin_user}
        }), 200

    app.logger.warning(f"Failed login attempt for email: {email}")
    return jsonify({"message": "Invalid email or password"}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    user_name = current_user.username
    logout_user()
    app.logger.info(f"User '{user_name}' logged out successfully.")
    session.clear() # Explicitly clear session for good measure
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/api/user_profile', methods=['GET', 'PUT'])
@login_required
def user_profile():
    user = current_user

    if request.method == 'PUT':
        data = request.get_json()
        if not data: return jsonify({"message": "No input data provided for update"}), 400

        try:
            # Validate and update fields
            if 'age' in data:
                age = int(data['age'])
                if not (12 <= age <= 120): raise ValueError("Age must be between 12 and 120.")
                user.age = age
            if 'gender' in data and data['gender'] in ['male', 'female', 'other']:
                user.gender = data['gender']
            if 'height' in data:
                height = float(data['height'])
                if not (50 <= height <= 300): raise ValueError("Height must be between 50 and 300 cm.")
                user.height_cm = height
            if 'weight' in data:
                weight = float(data['weight'])
                if not (20 <= weight <= 500): raise ValueError("Weight must be between 20 and 500 kg.")
                user.weight_kg = weight
            if 'diet_preference' in data: user.diet_preference = data['diet_preference']
            if 'activity_level' in data: user.activity_level = data['activity_level']
            if 'goals' in data: user.goals = data['goals']
            if 'preferred_cuisines' in data: user.preferred_cuisines = data.get('preferred_cuisines', '').strip()

            db.session.commit()
            app.logger.info(f"User profile updated for {user.username}")

            # Return the full updated profile including calculated values
            bmi = calculate_bmi(user.weight_kg, user.height_cm); bmi_category = get_bmi_category(bmi)
            bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.gender); tdee = calculate_tdee(bmr, user.activity_level)
            target_calories = get_target_calories(tdee, user.goals)
            return jsonify({
                "message": "Profile updated successfully!",
                "user_profile": {
                    "username": user.username, "age": user.age, "gender": user.gender,
                    "height_cm": user.height_cm, "weight_kg": user.weight_kg,
                    "diet_preference": user.diet_preference, "activity_level": user.activity_level,
                    "goals": user.goals, "preferred_cuisines": user.preferred_cuisines or "",
                    "bmi": bmi, "bmi_category": bmi_category, "bmr": bmr, "tdee": tdee,
                    "target_daily_calories": target_calories, "is_admin": user.is_admin_user
                }
            }), 200
        except ValueError as e:
            db.session.rollback()
            return jsonify({"message": f"Invalid data: {str(e)}"}), 400
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating profile for {user.username}: {e}", exc_info=True)
            return jsonify({"message": "Profile update failed due to a server error."}), 500

    # GET request
    bmi = calculate_bmi(user.weight_kg, user.height_cm); bmi_category = get_bmi_category(bmi)
    bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.gender); tdee = calculate_tdee(bmr, user.activity_level)
    target_calories = get_target_calories(tdee, user.goals)
    return jsonify({
        "username": user.username, "age": user.age, "gender": user.gender,
        "height_cm": user.height_cm, "weight_kg": user.weight_kg,
        "diet_preference": user.diet_preference, "activity_level": user.activity_level,
        "goals": user.goals, "bmi": bmi, "bmi_category": bmi_category,
        "bmr": bmr, "tdee": tdee, "target_daily_calories": target_calories,
        "preferred_cuisines": user.preferred_cuisines or "",
        "is_admin": user.is_admin_user,
        "is_superadmin": user.is_superadmin
    }), 200



@app.route('/api/workout_recommendations', methods=['GET'])
@login_required
def get_workout_recommendations():
    user = current_user
    bmi = calculate_bmi(user.weight_kg, user.height_cm); bmi_category = get_bmi_category(bmi)
    
    user_profile = {
        'age': user.age,
        'gender': user.gender,
        'height': user.height_cm,
        'weight': user.weight_kg
    }
    
    workouts = recommend_workouts_logic(user.goals, bmi_category, user_profile)
    return jsonify({"goal": user.goals, "workouts": workouts}), 200

# --- Calorie Cycle API ---
@app.route('/api/calorie_cycle', methods=['GET'])
@login_required
def get_calorie_cycle():
    """
    Get user's calorie cycle for the past 7 days.
    Returns target daily calories vs actual calories burned from workouts.
    """
    user = current_user
    
    # Calculate target daily calories
    bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.gender)
    tdee = calculate_tdee(bmr, user.activity_level)
    target_calories = get_target_calories(tdee, user.goals)
    
    # Get date range for last 7 days
    today = date.today()
    days_data = []
    
    for i in range(6, -1, -1):  # 6 days ago to today
        day_date = today - timedelta(days=i)
        
        # Get workout logs for this day
        day_workouts = WorkoutLog.query.filter(
            WorkoutLog.user_id == user.id,
            WorkoutLog.log_date == day_date
        ).all()
        
        # Sum calories burned
        calories_burned = sum(
            (log.calories_burned or 0) for log in day_workouts
        )
        
        # Calculate calorie balance (positive = burned, negative = deficit from target)
        days_data.append({
            "date": day_date.isoformat(),
            "day_name": day_date.strftime("%A"),  # e.g., "Monday"
            "day_short": day_date.strftime("%a"),  # e.g., "Mon"
            "target_calories": target_calories,
            "calories_burned": calories_burned,
            "workout_count": len(day_workouts)
        })
    
    return jsonify({
        "target_daily_calories": target_calories,
        "tdee": tdee,
        "goal": user.goals,
        "days": days_data
    }), 200

# --- KNN-Based Diet Recommendation ---
def generate_knn_based_diet_recommendation(user_profile_data, num_recipes_per_meal=1):
    """
    Generate a 7-day personalized diet plan using KNN classification.
    Recommends recipes based on user's diet preference and preferred cuisines.
    """
    global diet_df
    if diet_df is None or diet_df.empty: 
        app.logger.error("Diet DataFrame is None or empty for KNN recommendation.")
        return {"error": "Diet data not available."}
    
    base_df = diet_df.copy()
    recipe_name_column = 'Recipe_name'
    required_cols = [recipe_name_column, 'Calories', 'Protein', 'Carbs', 'Fat', 'Cuisine', 'Diet_type']
    
    if not all(col in base_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in base_df.columns]
        return {"error": f"Diet data incomplete (missing: {', '.join(missing)})."}
    
    # Clean data
    base_df.dropna(subset=['Calories','Protein','Carbs','Fat','Diet_type'], inplace=True)
    for col in ['Calories', 'Protein', 'Carbs', 'Fat']:
        base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0).astype(int)
    base_df = base_df[base_df['Calories'] > 50]
    
    if base_df.empty: 
        return {"error": "No suitable food items after cleaning."}
    
    # Get user preferences
    diet_pref = user_profile_data.get('diet_preference', 'any').lower()
    preferred_cuisines_str = user_profile_data.get('preferred_cuisines', "")
    preferred_cuisines_list = [c.strip().lower() for c in preferred_cuisines_str.split(',') if c.strip()] if preferred_cuisines_str else []
    target_calories_total = user_profile_data.get('target_calories', 2000)
    
    # Step 1: Filter by diet preference using KNN
    # Prepare features for KNN: Calories, Protein, Carbs, Fat
    feature_cols = ['Calories', 'Protein', 'Carbs', 'Fat']
    
    # Create training set based on diet preference
    if diet_pref != 'any':
        if diet_pref == 'vegetarian':
            target_diet_types = ['Vegetarian', 'Vegan']
        elif diet_pref == 'vegan':
            target_diet_types = ['Vegan']
        elif diet_pref == 'non-vegetarian':
            target_diet_types = ['Non-vegetarian']
        else:
            target_diet_types = ['Vegetarian', 'Vegan', 'Non-vegetarian']
    else:
        target_diet_types = ['Vegetarian', 'Vegan', 'Non-vegetarian']
    
    # Filter by diet preference
    diet_filtered_df = base_df[base_df['Diet_type'].isin(target_diet_types)].copy()
    if diet_filtered_df.empty:
        app.logger.warning(f"No items for diet_pref '{diet_pref}', using all items.")
        diet_filtered_df = base_df.copy()
    
    # Step 2: Filter by preferred cuisines if provided
    if preferred_cuisines_list:
        cuisine_filtered_df = diet_filtered_df[
            diet_filtered_df['Cuisine'].notna() & 
            diet_filtered_df['Cuisine'].astype(str).str.lower().isin(preferred_cuisines_list)
        ].copy()
        if cuisine_filtered_df.empty:
            app.logger.warning("No items for preferred cuisines, using diet preference pool.")
            available_recipes_df = diet_filtered_df.copy()
        else:
            available_recipes_df = cuisine_filtered_df.copy()
    else:
        available_recipes_df = diet_filtered_df.copy()
    
    if available_recipes_df.empty:
        return {"error": "No recipes available after filtering preferences."}
    
    # Step 3: Use KNN to find similar recipes within preferred parameters
    # For each meal type per day, find recipes close to target calories using KNN
    
    def get_recipes_for_meal_knn(target_calories, available_df, num_recipes, exclude_recipes=None):
        """
        Use KNN to find recipes similar to target calorie profile.
        """
        if exclude_recipes is None:
            exclude_recipes = set()
        
        # Remove already used recipes
        candidate_df = available_df[~available_df[recipe_name_column].isin(exclude_recipes)].copy()
        if candidate_df.empty:
            return []
        
        if len(candidate_df) < 5:
            # If too few candidates, just return best match
            candidate_df['calorie_diff'] = abs(candidate_df['Calories'] - target_calories)
            best = candidate_df.nsmallest(num_recipes, 'calorie_diff')
            return best.to_dict('records')
        
        try:
            # Prepare features
            X = candidate_df[feature_cols].values
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create a target feature vector (based on target calories and reasonable macros)
            # Assuming 40% carbs, 30% protein, 30% fat distribution
            target_protein = (target_calories * 0.30) / 4  # 4 kcal per gram
            target_carbs = (target_calories * 0.40) / 4
            target_fat = (target_calories * 0.30) / 9    # 9 kcal per gram
            
            target_features = np.array([[target_calories, target_protein, target_carbs, target_fat]])
            target_scaled = scaler.transform(target_features)
            
            # Fit KNN with distance weighting (optimized for better accuracy)
            n_neighbors = min(num_recipes * 3, max(3, len(candidate_df) // 5))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
            # Use a dummy target for classification (we're using it for nearest neighbor finding)
            dummy_target = np.zeros(len(candidate_df))
            knn.fit(X_scaled, dummy_target)
            
            # Find nearest neighbors
            distances, indices = knn.kneighbors(target_scaled, n_neighbors=min(num_recipes, n_neighbors))
            
            # Get the closest recipes
            selected_recipes = candidate_df.iloc[indices[0]].to_dict('records')
            return selected_recipes[:num_recipes]
        
        except Exception as e:
            app.logger.warning(f"KNN recommendation failed: {e}. Using fallback method.")
            # Fallback: simple calorie-based selection
            candidate_df['calorie_diff'] = abs(candidate_df['Calories'] - target_calories)
            best = candidate_df.nsmallest(num_recipes, 'calorie_diff')
            return best.to_dict('records')
    
    # Step 4: Generate 7-day meal plan
    calorie_dist = {'breakfast': 0.25, 'lunch': 0.40, 'dinner': 0.35}
    weekly_plan = []
    used_recipes_overall = set()
    
    # Get available cuisines for variety and shuffle for randomization on regenerate
    available_cuisines = available_recipes_df['Cuisine'].unique().tolist()
    random.shuffle(available_cuisines)  # Randomize cuisine order for variety on each regeneration
    app.logger.info(f"Available cuisines for KNN recommendations (shuffled): {available_cuisines}")
    
    for day_num in range(7):
        # For each day, try to use a different cuisine if possible
        day_cuisine = available_cuisines[day_num % len(available_cuisines)] if available_cuisines else None
        
        daily_meals = {}
        daily_recipes_this_day = set()
        
        for meal_type, proportion in calorie_dist.items():
            target_meal_calories = target_calories_total * proportion
            
            # Filter to specific cuisine for variety
            if day_cuisine:
                meal_recipe_pool = available_recipes_df[available_recipes_df['Cuisine'] == day_cuisine].copy()
                if meal_recipe_pool.empty:
                    meal_recipe_pool = available_recipes_df.copy()
            else:
                meal_recipe_pool = available_recipes_df.copy()
            
            # Get recommended recipes using KNN
            exclude_set = used_recipes_overall.union(daily_recipes_this_day)
            recommended_recipes = get_recipes_for_meal_knn(
                target_meal_calories, 
                meal_recipe_pool, 
                num_recipes_per_meal, 
                exclude_set
            )
            
            if recommended_recipes:
                meal_options = []
                for recipe in recommended_recipes:
                    meal_options.append({
                        "name": str(recipe.get(recipe_name_column, "N/A")),
                        "calories": int(recipe.get('Calories', 0)),
                        "protein": int(recipe.get('Protein', 0)),
                        "carbs": int(recipe.get('Carbs', 0)),
                        "fat": int(recipe.get('Fat', 0)),
                        "cuisine": str(recipe.get('Cuisine', 'N/A'))
                    })
                    daily_recipes_this_day.add(recipe.get(recipe_name_column))
                daily_meals[meal_type] = meal_options
            else:
                daily_meals[meal_type] = [{"name": "N/A - No suitable recipe", "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "cuisine": "N/A"}]
        
        day_total_calories = sum(
            meal_options[0]['calories'] 
            for meal_options in daily_meals.values() 
            if meal_options and meal_options[0].get('calories', 0) > 0
        )
        
        weekly_plan.append({
            "day": day_num + 1,
            "daily_summary": {
                "meals": daily_meals,
                "total_calories_for_day": day_total_calories
            }
        })
        
        used_recipes_overall.update(daily_recipes_this_day)
    
    return {"weekly_diet_plan": weekly_plan}

@app.route('/api/knn_diet_plan', methods=['GET'])
@login_required
def get_knn_diet_plan():
    """Get KNN-based personalized weekly diet plan"""
    user = current_user
    # Recalculate BMR/TDEE/Target based on current profile
    bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.gender)
    tdee = calculate_tdee(bmr, user.activity_level)
    target_calories = get_target_calories(tdee, user.goals)

    user_profile_for_diet = {
        'target_calories': target_calories,
        'diet_preference': user.diet_preference,
        'preferred_cuisines': user.preferred_cuisines
    }
    
    knn_diet_plan = generate_knn_based_diet_recommendation(user_profile_for_diet, num_recipes_per_meal=1)
    
    if "error" in knn_diet_plan:
        return jsonify(knn_diet_plan), 400
    
    return jsonify(knn_diet_plan), 200

# --- Workout Log API Endpoints ---
@app.route('/api/workout_logs', methods=['POST'])
@login_required
def log_workout():
    data = request.get_json()
    if not data: return jsonify({"message": "No data provided for workout log"}), 400

    exercise_name = data.get('exercise_name')
    duration_minutes_str = data.get('duration_minutes')
    calories_burned_str = data.get('calories_burned')
    log_date_str = data.get('log_date') # Expect YYYY-MM-DD from frontend
    feedback = data.get('feedback')
    # Optional structured pose/rep data (JSON string or object) from frontend
    pose_data = data.get('pose_data')

    if not exercise_name or not exercise_name.strip() or duration_minutes_str is None:
        return jsonify({"message": "Exercise name and duration are required"}), 400

    try:
        duration_minutes = int(duration_minutes_str)
        calories_burned = int(calories_burned_str) if calories_burned_str is not None else None

        log_date_to_save = date.today() # Default
        if log_date_str:
            try: log_date_to_save = datetime.strptime(log_date_str, '%Y-%m-%d').date()
            except ValueError: app.logger.warning(f"Invalid date format '{log_date_str}' for workout log. Defaulting to today.")

        if duration_minutes <= 0: return jsonify({"message": "Duration must be a positive number."}), 400
        if calories_burned is not None and calories_burned < 0: return jsonify({"message": "Calories burned cannot be negative."}), 400

        new_log = WorkoutLog(
            user_id=current_user.id, exercise_name=exercise_name.strip(),
            duration_minutes=duration_minutes, calories_burned=calories_burned,
            log_date=log_date_to_save, feedback=feedback.strip() if feedback else None,
            pose_data=(pose_data if isinstance(pose_data, str) else (json.dumps(pose_data) if pose_data is not None else None))
        )
        db.session.add(new_log); db.session.commit()
        app.logger.info(f"Workout logged for user {current_user.username}: {exercise_name}")
        return jsonify({
            "message": "Workout logged successfully!",
            "log": {
                "id": new_log.id, "exercise_name": new_log.exercise_name,
                "duration_minutes": new_log.duration_minutes, "calories_burned": new_log.calories_burned,
                "log_date": new_log.log_date.isoformat(), "feedback": new_log.feedback,
                "pose_data": (new_log.pose_data if new_log.pose_data else None)
            }
        }), 201
    except ValueError: return jsonify({"message": "Invalid data type for duration or calories (must be numbers)."}), 400
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error logging workout for user {current_user.username}: {e}", exc_info=True)
        return jsonify({"message": "Failed to log workout due to a server error."}), 500

@app.route('/api/workout_logs', methods=['GET'])
@login_required
def get_workout_logs():
    try:
        year = request.args.get('year', type=int); month = request.args.get('month', type=int); day = request.args.get('day', type=int)
        query = WorkoutLog.query.filter_by(user_id=current_user.id)
        if year and month and day:
            if not (1 <= month <= 12 and 1 <= day <= 31): return jsonify({"message": "Invalid month or day parameter."}), 400
            try: specific_date = date(year, month, day); query = query.filter(WorkoutLog.log_date == specific_date)
            except ValueError: return jsonify({"message": "Invalid date constructed from year, month, day."}), 400
        elif year and month:
            if not (1 <= month <= 12): return jsonify({"message": "Invalid month parameter."}), 400
            query = query.filter(db.extract('year', WorkoutLog.log_date) == year, db.extract('month', WorkoutLog.log_date) == month)

        user_logs = query.order_by(WorkoutLog.log_date.desc(), WorkoutLog.id.desc()).all()
        logs_data = [{
            "id": log.id, "exercise_name": log.exercise_name,
            "duration_minutes": log.duration_minutes, "calories_burned": log.calories_burned,
            "log_date": log.log_date.isoformat(), "feedback": log.feedback
        } for log in user_logs]
        return jsonify(logs_data), 200
    except Exception as e:
        app.logger.error(f"Error fetching workout logs for user {current_user.username}: {e}", exc_info=True)
        return jsonify({"message": "Failed to fetch workout logs due to a server error."}), 500

# --- To-Do List API Endpoints ---
@app.route('/api/todos', methods=['GET'])
@login_required
def get_todos():
    user_todos = Todo.query.filter_by(user_id=current_user.id).order_by(Todo.created_at.asc()).all()
    return jsonify([{"id": todo.id, "task": todo.task, "completed": todo.completed} for todo in user_todos]), 200

@app.route('/api/todos', methods=['POST'])
@login_required
def add_todo():
    data = request.get_json()
    if not data or not data.get('task') or not data.get('task').strip():
        return jsonify({"message": "Task content is required and cannot be empty"}), 400
    new_todo = Todo(user_id=current_user.id, task=data['task'].strip(), completed=False)
    try:
        db.session.add(new_todo); db.session.commit()
        app.logger.info(f"Todo '{new_todo.task}' added for user {current_user.username}")
        return jsonify({"id": new_todo.id, "task": new_todo.task, "completed": new_todo.completed}), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error adding todo for user {current_user.username}: {e}", exc_info=True)
        return jsonify({"message": "Failed to add todo due to server error."}), 500

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
@login_required
def update_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id) # Returns 404 if not found
    if todo.user_id != current_user.id: return jsonify({"message": "Unauthorized to modify this todo"}), 403

    data = request.get_json()
    if data is None: return jsonify({"message": "No data provided for update"}), 400

    updated = False
    if 'task' in data:
        task_content = data['task']
        if task_content is None or not task_content.strip():
            return jsonify({"message": "Task content cannot be empty if provided"}), 400
        if todo.task != task_content.strip():
            todo.task = task_content.strip()
            updated = True

    if 'completed' in data and isinstance(data['completed'], bool):
        if todo.completed != data['completed']:
            todo.completed = data['completed']
            updated = True

    if updated:
        try:
            db.session.commit()
            app.logger.info(f"Todo ID {todo_id} updated for user {current_user.username}")
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating todo ID {todo_id} for user {current_user.username}: {e}", exc_info=True)
            return jsonify({"message": "Failed to update todo due to server error."}), 500

    return jsonify({"id": todo.id, "task": todo.task, "completed": todo.completed}), 200

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
@login_required
def delete_todo_item(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    if todo.user_id != current_user.id: return jsonify({"message": "Unauthorized to delete this todo"}), 403
    try:
        db.session.delete(todo); db.session.commit()
        app.logger.info(f"Todo ID {todo_id} deleted for user {current_user.username}")
        return jsonify({"message": "Todo deleted successfully"}), 200 # 200 with message is fine
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting todo ID {todo_id} for user {current_user.username}: {e}", exc_info=True)
        return jsonify({"message": "Failed to delete todo due to server error."}), 500

# --- Admin API Endpoints ---
from functools import wraps

def admin_required(f):
    """Decorator to require admin privileges for an endpoint."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin_user:
            return jsonify({"message": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    """Get application statistics for admin dashboard."""
    try:
        total_users = User.query.count()
        total_workouts = WorkoutLog.query.count()
        total_diet_logs = DietLog.query.count()
        total_todos = Todo.query.count()
        admin_count = User.query.filter_by(is_admin_user=True).count()
        
        # Recent activity (last 7 days)
        week_ago = date.today() - timedelta(days=7)
        recent_workouts = WorkoutLog.query.filter(WorkoutLog.log_date >= week_ago).count()
        recent_users = User.query.filter(User.id.in_(
            db.session.query(User.id).order_by(User.id.desc()).limit(10)
        )).count()
        
        return jsonify({
            "total_users": total_users,
            "total_workouts": total_workouts,
            "total_diet_logs": total_diet_logs,
            "total_todos": total_todos,
            "admin_count": admin_count,
            "recent_workouts_7d": recent_workouts
        }), 200
    except Exception as e:
        app.logger.error(f"Error fetching admin stats: {e}", exc_info=True)
        return jsonify({"message": "Failed to fetch statistics"}), 500

@app.route('/api/model_evaluation', methods=['GET'])
@admin_required
def get_model_evaluation():
    """
    Get KNN model evaluation metrics (admin only).
    
    Returns comprehensive ML metrics:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix
    - Per-class metrics
    - 5-fold Cross-validation results
    """
    try:
        from model_evaluation import DietModelEvaluator
        
        evaluator = DietModelEvaluator()
        
        if not evaluator.load_data():
            return jsonify({"message": "Failed to load dataset for evaluation"}), 500
        
        results = evaluator.evaluate_all_metrics()
        
        app.logger.info(f"Model evaluation completed. Accuracy: {results['overall_metrics']['accuracy']}")
        
        return jsonify(results), 200
    except Exception as e:
        app.logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return jsonify({"message": f"Model evaluation failed: {str(e)}"}), 500

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_admin_users():
    """Get list of all users for admin management."""
    try:
        users = User.query.order_by(User.id.asc()).all()
        users_data = []
        for user in users:
            workout_count = WorkoutLog.query.filter_by(user_id=user.id).count()
            users_data.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "gender": user.gender,
                "age": user.age,
                "goals": user.goals,
                "is_admin": user.is_admin_user,
                "is_superadmin": user.is_superadmin,
                "workout_count": workout_count
            })
        return jsonify({"users": users_data}), 200
    except Exception as e:
        app.logger.error(f"Error fetching users for admin: {e}", exc_info=True)
        return jsonify({"message": "Failed to fetch users"}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_admin_user(user_id):
    """Update a user (admin can toggle admin status, update profile)."""
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        if not data:
            return jsonify({"message": "No data provided"}), 400
        
        # Prevent admin from removing their own admin status
        if 'is_admin' in data and user.id == current_user.id and not data['is_admin']:
            return jsonify({"message": "Cannot remove your own admin status"}), 400
        
        # Prevent modifying superadmin's admin status (unless you are a superadmin)
        if 'is_admin' in data and user.is_superadmin and not current_user.is_superadmin:
            return jsonify({"message": "Cannot modify superadmin's admin status"}), 403
        
        # Update fields
        if 'is_admin' in data and isinstance(data['is_admin'], bool):
            user.is_admin_user = data['is_admin']
        if 'username' in data and data['username']:
            user.username = data['username'].strip()
        if 'goals' in data and data['goals']:
            user.goals = data['goals']
        
        db.session.commit()
        app.logger.info(f"Admin {current_user.username} updated user {user.username} (ID: {user_id})")
        return jsonify({
            "message": "User updated successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin_user,
                "goals": user.goals
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating user {user_id}: {e}", exc_info=True)
        return jsonify({"message": "Failed to update user"}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_admin_user(user_id):
    """Delete a user account (admin only)."""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent admin from deleting themselves
        if user.id == current_user.id:
            return jsonify({"message": "Cannot delete your own account from admin panel"}), 400
        
        # Prevent deleting superadmin (unless you are a superadmin)
        if user.is_superadmin and not current_user.is_superadmin:
            return jsonify({"message": "Cannot delete superadmin user"}), 403
        
        username = user.username
        
        # Delete related records first
        WorkoutLog.query.filter_by(user_id=user_id).delete()
        DietLog.query.filter_by(user_id=user_id).delete()
        Todo.query.filter_by(user_id=user_id).delete()
        
        db.session.delete(user)
        db.session.commit()
        
        app.logger.info(f"Admin {current_user.username} deleted user {username} (ID: {user_id})")
        return jsonify({"message": f"User '{username}' deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
        return jsonify({"message": "Failed to delete user"}), 500

# --- Error Handlers ---
@app.errorhandler(404) # For general 404s not caught by get_or_404
def not_found_error(error):
    app.logger.warning(f"Resource not found: {request.path} - Error: {error}")
    return jsonify({"message": "API Resource not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback() # Rollback in case of DB error
    app.logger.error(f"API Server Error: {error}", exc_info=True)
    return jsonify({"message": "An internal server error occurred."}), 500

@login_manager.unauthorized_handler
def unauthorized_api_access():
    app.logger.warning(f"Unauthorized API access: {request.path} (Origin: {request.headers.get('Origin')})")
    return jsonify(message="Unauthorized: Authentication is required."), 401

@app.errorhandler(403)
def forbidden_api_error(error):
    app.logger.warning(f"Forbidden access: {request.path} by user {current_user.username if current_user.is_authenticated else 'Anonymous'} (Origin: {request.headers.get('Origin')})")
    return jsonify({"message": "Forbidden: You don't have permission to access this."}), 403

@app.errorhandler(400) # For general bad requests
def bad_request_api_error(error):
    message = error.description if hasattr(error, 'description') and error.description else "Bad API request. Please check your input."
    app.logger.warning(f"Bad request: {request.path} - Message: {message} - Error: {error}")
    return jsonify({"message": message}), 400


# --- Initialize Database and Admin ---
def init_db():
    try:
        with app.app_context():
            app.logger.info("Ensuring database tables exist...")
            db.create_all()
            
            # --- Migration: ensure `pose_data` column exists on WorkoutLog table ---
            try:
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    res = conn.execute(text("PRAGMA table_info('workout_log')"))
                    rows = res.fetchall()
                    cols = [r[1] for r in rows]
                    if 'pose_data' not in cols:
                        app.logger.info("Migration: adding 'pose_data' column to 'workout_log' table.")
                        try:
                            conn.execute(text("ALTER TABLE workout_log ADD COLUMN pose_data TEXT"))
                            app.logger.info("Migration: 'pose_data' column added to workout_log.")
                        except Exception as me:
                            app.logger.error(f"Migration failed to add 'pose_data' column: {me}")
            except Exception as e:
                app.logger.warning(f"Could not run migration check for 'pose_data' column: {e}")

            # Ensure admin user with admin email exists
            admin_username = "Abhinandan"
            admin_email = os.environ.get('ADMIN_EMAIL', 'abhinandan@admin.com')
            admin_password = os.environ.get('ADMIN_PASSWORD', '123456')
            
            try:
                admin = User.query.filter_by(email=admin_email).first()
                if admin:
                    if not admin.is_admin_user:
                        admin.is_admin_user = True
                        db.session.commit()
                        app.logger.info(f"User with email '{admin_email}' found and ensured admin status.")
                else:
                    # Create a fresh admin user
                    admin = User(
                        username=admin_username, email=admin_email,
                        gender="other", age=30, height_cm=160, weight_kg=60,
                        diet_preference="any", activity_level="moderate", goals="maintenance",
                        is_admin_user=True
                    )
                    admin.set_password(admin_password)
                    db.session.add(admin)
                    db.session.commit()
                    app.logger.info(f"Admin user '{admin_email}' created with default password.")
            except Exception as e:
                app.logger.error(f"Error initializing admin user: {e}")

            app.logger.info("Database initialization complete.")
    except Exception as e:
        app.logger.error(f"Failed to initialize database: {e}")

# Run initialization on startup/import
init_db()

# --- Main Execution ---
if __name__ == '__main__':
    app.logger.info("Starting Fit-Guide API Backend Server in debug mode...")
    app.run(debug=True, host='0.0.0.0', port=5000)
