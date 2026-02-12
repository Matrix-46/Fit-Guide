"""
User Persistence Test Script
============================
This script tests whether user accounts persist in the database across server restarts.

Usage:
    python test_user_persistence.py
"""

import sys
import os
import random
import string
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import app components
from app import app, db, User

def generate_test_email():
    """Generate a unique test email"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"test_{timestamp}_{random_suffix}@test.com"

def run_persistence_test():
    """
    Test user persistence by:
    1. Creating a test user
    2. Verifying the user exists
    3. Checking total user count
    """
    
    print("="*80)
    print("USER PERSISTENCE TEST")
    print("="*80)
    
    with app.app_context():
        # Show current database configuration
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        print(f"\n📍 Database URI: {db_uri}")
        
        # Check if database is using PostgreSQL or SQLite
        if 'postgresql://' in db_uri or 'postgres://' in db_uri:
            print("🐘 Database type: PostgreSQL")
        elif 'sqlite:///' in db_uri:
            print("🗄️  Database type: SQLite")
            db_path = db_uri.replace('sqlite:///', '')
            if os.path.exists(db_path):
                print(f"✅ Database file exists: {db_path}")
                print(f"📊 File size: {os.path.getsize(db_path):,} bytes")
            else:
                print(f"❌ Database file NOT found: {db_path}")
        
        # Get initial user count
        try:
            initial_count = User.query.count()
            print(f"\n👥 Initial user count: {initial_count}")
        except Exception as e:
            print(f"\n❌ Error accessing database: {e}")
            return False
        
        # List existing users
        if initial_count > 0:
            print("\n📝 Existing users:")
            users = User.query.limit(10).all()
            for user in users:
                print(f"   - ID: {user.id:3d} | {user.username:20s} | {user.email}")
            if initial_count > 10:
                print(f"   ... and {initial_count - 10} more")
        
        # Create a test user
        test_email = generate_test_email()
        test_username = f"TestUser_{datetime.now().strftime('%H%M%S')}"
        test_password = "test123456"
        
        print(f"\n🧪 Creating test user:")
        print(f"   Email: {test_email}")
        print(f"   Username: {test_username}")
        print(f"   Password: {test_password}")
        
        try:
            new_user = User(
                username=test_username,
                email=test_email,
                gender="other",
                age=25,
                height_cm=170,
                weight_kg=70,
                diet_preference="any",
                activity_level="moderate",
                goals="maintenance",
                is_admin_user=False
            )
            new_user.set_password(test_password)
            
            db.session.add(new_user)
            db.session.commit()
            
            print(f"\n✅ User created successfully with ID: {new_user.id}")
            
            # Immediately verify the user exists
            verification_user = User.query.filter_by(email=test_email).first()
            if verification_user:
                print(f"✅ VERIFICATION: User found in database (ID: {verification_user.id})")
            else:
                print(f"❌ VERIFICATION FAILED: User NOT found in database after commit!")
                return False
            
            # Check updated user count
            final_count = User.query.count()
            print(f"\n👥 Final user count: {final_count} (increase: +{final_count - initial_count})")
            
            print("\n" + "="*80)
            print("✅ PERSISTENCE TEST PASSED")
            print("="*80)
            print("\nNext steps:")
            print("1. Restart the backend server")
            print("2. Try logging in with:")
            print(f"   Email: {test_email}")
            print(f"   Password: {test_password}")
            print("3. If login succeeds, database persistence is working correctly!")
            print("\nOr run this command to verify user still exists:")
            print(f"   python -c \"from app import app, db, User; app.app_context().push(); user = User.query.filter_by(email='{test_email}').first(); print('User exists!' if user else 'User NOT found!')\"")
            print("="*80)
            
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error creating test user: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = run_persistence_test()
    sys.exit(0 if success else 1)
