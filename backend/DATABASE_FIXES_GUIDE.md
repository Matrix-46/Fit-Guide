# Database Persistence Fixes - User Guide

## 🎯 What Was Fixed

Your Fit-Guide app now has several enhancements to ensure user accounts persist properly in the database:

### 1. ✅ Database Health Check Endpoint
**Access:** `http://your-backend-url/api/db_health`

This endpoint shows:
- Database type (SQLite or PostgreSQL)
- Total user count
- Database accessibility status
- Admin user configuration

**Local Example:** `http://localhost:5000/api/db_health`

### 2. ✅ Enhanced Logging
The backend now logs detailed information about:
- User registration success/failure with verification
- User login attempts with specific error messages
- Total user count when login fails
- Database operation status

### 3. ✅ User Persistence Test Script
**File:** `backend/test_user_persistence.py`

Run this script to verify users persist across server restarts:
```bash
cd backend
python test_user_persistence.py
```

## 🔍 Diagnosing Your Issue

### Step 1: Check Which Environment Has the Problem

**Are you experiencing this on:**
- Your computer (localhost)? → Follow **Local Testing** below
- Render (production)? → Follow **Production Testing** below  
- Both? → Do both tests

### Step 2: Local Testing

1. **Check database health:**
   ```bash
   cd backend
   python check_db.py
   ```

2. **Test user persistence:**
   ```bash
   python test_user_persistence.py
   ```
   - This creates a test user and shows you credentials
   - Restart your server: Stop (`Ctrl+C`) and run `python app.py` again
   - Try logging in with the test credentials
   - ✅ If login works → Database persistence is fine locally

3. **Check logs when registering/logging in:**
   - Look for `✅` (success) or `❌` (failure) symbols in terminal output
   - Registration should show: `✅ User 'username' registered successfully with ID: X`
   `✅ VERIFICATION: User 'username' found in database`
   - Login should show: `✅ User 'username' (ID: X, email: ...) logged in successfully`

### Step 3: Production Testing (Render)

1. **Access health check endpoint:**
   - Visit: `https://your-backend-url.onrender.com/api/db_health`
   - Check the response shows:
     - `"type": "PostgreSQL"` (not SQLite!)
     - `"accessible": true`
     - `"total_count"`: Should show number of users

2. **Check Render logs:**
   - Go to Render Dashboard → Your Backend Service → Logs
   - Look for registration/login attempts
   - Search for `✅` or `❌` symbols
   - If you see `❌ VERIFICATION FAILED` → database is not persisting

3. **Common Render Issues:**

   **Issue: Database shows 0 users even after registration**
   - **Cause:** DATABASE_URL not set or pointing to wrong database
   - **Fix:** 
     1. Go to Render Dashboard → Your Service → Environment
     2. Verify `DATABASE_URL` is set to a PostgreSQL database
     3. If missing, add a PostgreSQL database in Render
     4. Redeploy the service

   **Issue: Users exist but can't login**
   - **Cause:** Session cookie issues (browser blocking cookies)
   - **Fix:** This is likely a browser/CORS issue, not database
     - Try in a different browser
     - Clear cookies and cache
     - Check browser console for cookie errors

## 📊 Understanding the Logs

### Good Registration Log:
```
✅ User 'JohnDoe' (email: john@example.com) registered successfully with ID: 5
✅ VERIFICATION: User 'JohnDoe' found in database immediately after commit (ID: 5)
```

### Bad Registration Log:
```
✅ User 'JohnDoe' (email: john@example.com) registered successfully with ID: 5
❌ VERIFICATION FAILED: User 'JohnDoe' NOT found in database after commit!
```
☝️ This means database commits are not working!

### Good Login Log:
```
✅ User 'JohnDoe' (ID: 5, email: john@example.com) logged in successfully.
```

### Bad Login Log (User Not Found):
```
❌ Login failed: No user found with email 'john@example.com'
📊 Total users in database: 3
```
☝️ This means the user was never saved OR was deleted OR using wrong email

### Bad Login Log (Wrong Password):
```
❌ Login failed for email 'john@example.com': Incorrect password
```

## 🛠️ Quick Fixes

### Fix 1: Database File Issues (Local Only)

If using SQLite locally and users disappear:
```bash
cd backend
# Check if database file exists
dir instance
# Should see: fit_guide.db

# Check users in database
python check_db.py
```

If database file is missing → Something is deleting it. Check:
- Are you running from different directories?
- Is instance folder in `.gitignore`? (It should be!)
- Are you using virtual environment consistently?

### Fix 2: Render PostgreSQL Not Set Up

1. Go to Render Dashboard
2. Create a new PostgreSQL database
3. Copy the "Internal Database URL"
4. Add as `DATABASE_URL` environment variable in your web service
5. Deploy

### Fix 3: Session vs Database Confusion

**Important:** If the issue is "I can login but then get logged out", that's a **session** problem, not database!

Database stores: User accounts (permanent)
Sessions store: "Who is currently logged in" (temporary)

If you can login immediately after registration but can't stay logged in, that's a session/cookie issue, not a database problem.

## 📞 Still Having Issues?

Run all diagnostics and share the output:

```bash
cd backend

# 1. Check database status
python check_db.py > db_status.txt

# 2. Test user persistence  
python test_user_persistence.py > persistence_test.txt

# 3. Check Flask configuration
python test_db_config.py > config_test.txt

# 4. Check backend health (while server is running)
# Visit: http://localhost:5000/api/db_health
# Save the JSON response
```

Share these files to get help debugging!

## ✅ Verification Checklist

- [ ] `/api/db_health` endpoint returns user count
- [ ] `check_db.py` shows users in database
- [ ] `test_user_persistence.py` creates user successfully
- [ ] Can login with test user after server restart
- [ ] Backend logs show `✅` for registration
- [ ] Backend logs show `✅` for login
