"""
Test script to verify which database file Flask is actually using
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import app and db
from app import app, db, User

with app.app_context():
    print("=" * 80)
    print("FLASK DATABASE CONFIGURATION TEST")
    print("=" * 80)
    
    # Check what database URI Flask is using
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"\n📍 Flask is configured to use:")
    print(f"   {db_uri}")
    
    # Extract the actual file path if using SQLite
    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            # Make it absolute
            db_path = os.path.abspath(db_path)
        print(f"\n📁 Absolute database file path:")
        print(f"   {db_path}")
        print(f"\n✓ File exists: {os.path.exists(db_path)}")
        if os.path.exists(db_path):
            print(f"✓ File size: {os.path.getsize(db_path):,} bytes")
    
    # Try to query users
    try:
        user_count = User.query.count()
        print(f"\n👥 Users found via Flask ORM: {user_count}")
        
        if user_count > 0:
            print("\n📝 User list:")
            users = User.query.all()
            for user in users:
                print(f"   - {user.username} ({user.email})")
    except Exception as e:
        print(f"\n❌ Error querying users: {e}")
    
    print("\n" + "=" * 80)
