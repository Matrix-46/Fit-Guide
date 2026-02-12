import sqlite3
import os
from datetime import datetime

# Check both database files
db_files = ['instance/fit_guide.db', 'instance/app.db']

print(f"\n{'='*80}")
print(f"DATABASE DIAGNOSTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('='*80)

# Check environment variable
from dotenv import load_dotenv
load_dotenv()
db_url = os.environ.get('DATABASE_URL')
print(f"\nDATABASE_URL environment variable: {db_url if db_url else 'NOT SET (will use SQLite)'}")

for db_file in db_files:
    print(f"\n{'='*80}")
    print(f"Checking: {db_file}")
    print('='*80)
    
    if not os.path.exists(db_file):
        print(f"❌ Database file does not exist at this path!")
        continue
    
    # Get file size and modification time
    file_size = os.path.getsize(db_file)
    mod_time = datetime.fromtimestamp(os.path.getmtime(db_file))
    print(f"📊 File size: {file_size:,} bytes")
    print(f"🕒 Last modified: {mod_time}")
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        print(f"\n📋 Tables: {', '.join(table_names)}")
        
        # Check user table
        if 'user' in table_names:
            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            print(f"\n👥 Total users: {user_count}")
            
            # Show all users with more details
            cursor.execute("SELECT id, username, email, is_admin_user FROM user ORDER BY id")
            users = cursor.fetchall()
            if users:
                print("\n📝 User List:")
                for user in users:
                    admin_flag = " [ADMIN]" if user[3] else ""
                    print(f"   ID: {user[0]:3d} | Username: {user[1]:20s} | Email: {user[2]:30s}{admin_flag}")
            else:
                print("   ⚠️  No users found in table")
                
            # Check workout logs
            if 'workout_log' in table_names:
                cursor.execute("SELECT COUNT(*) FROM workout_log")
                workout_count = cursor.fetchone()[0]
                print(f"\n🏋️  Total workout logs: {workout_count}")
            
            # Check diet logs
            if 'diet_log' in table_names:
                cursor.execute("SELECT COUNT(*) FROM diet_log")
                diet_count = cursor.fetchone()[0]
                print(f"🍎 Total diet logs: {diet_count}")
        else:
            print("\n❌ 'user' table not found!")
        
        conn.close()
        print(f"\n✅ Database check complete for {db_file}")
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}\n")
