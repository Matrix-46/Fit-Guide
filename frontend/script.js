// frontend/script.js
document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = typeof CONFIG !== 'undefined' ? CONFIG.API_BASE_URL : 'http://localhost:5000/api';

    // --- DOM Elements ---
    const authChoiceView = document.getElementById('authChoiceView');
    const loginView = document.getElementById('loginView');
    const registerView = document.getElementById('registerView');
    const dashboardSection = document.getElementById('dashboardSection');
    const allMainViews = [authChoiceView, loginView, registerView, dashboardSection].filter(Boolean);

    const showLoginBtnFromChoice = document.getElementById('showLoginBtnFromChoice');
    const showRegisterBtnFromChoice = document.getElementById('showRegisterBtnFromChoice');
    const switchToRegisterBtn = document.getElementById('switchToRegisterBtn');
    const switchToLoginBtn = document.getElementById('switchToLoginBtn');
    const backToLoginFromRegister = document.getElementById('backToLoginFromRegister');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const authChoiceMessage = document.getElementById('authChoiceMessage');
    const loginMessage = document.getElementById('loginMessage');
    const registerMessage = document.getElementById('registerMessage');

    const logoutBtn = document.getElementById('logoutBtn');
    const dashboardHeader = document.getElementById('dashboardHeader');
    const dashboardWelcomeMessage = document.getElementById('dashboardWelcomeMessage');
    const usernameDisplay = document.getElementById('usernameDisplay');
    const dashboardOverviewPage = document.getElementById('dashboardOverviewPage');
    const dashboardNavItems = document.querySelectorAll('.dashboard-nav-item'); // Will re-query if admin card is added
    const dashboardDetailContent = document.getElementById('dashboardDetailContent');
    const backToDashboardOverviewBtn = document.getElementById('backToDashboardOverviewBtn');
    let allDashboardDetailPages = document.querySelectorAll('#dashboardDetailContent > .dashboard-page'); // Will re-query
    const dashboardApiMessage = document.getElementById('dashboardApiMessage');
    const adminPanelCard = document.getElementById('adminPanelCard'); // For admin link


    const headerTitle = document.getElementById('headerTitle');
    const currentYearSpan = document.getElementById('currentYear');

    // Registration Cuisines
    const regPreferredCuisinesContainer = document.getElementById('regPreferredCuisinesContainer');

    // Profile Display Elements
    const profileUsername = document.getElementById('profileUsername');
    const profileAge = document.getElementById('profileAge');
    const profileGender = document.getElementById('profileGender');
    const profileHeight = document.getElementById('profileHeight');
    const profileWeight = document.getElementById('profileWeight');
    const profileDietPref = document.getElementById('profileDietPref');
    const profileCuisines = document.getElementById('profileCuisines');
    const profileActivityLevel = document.getElementById('profileActivityLevel');
    const profileGoals = document.getElementById('profileGoals');
    const profileBmiValue = document.getElementById('profileBmiValue');
    const profileBmiCategory = document.getElementById('profileBmiCategory');
    const profileBmr = document.getElementById('profileBmr');
    const profileTdee = document.getElementById('profileTdee');
    const profileTargetCalories = document.getElementById('profileTargetCalories');

    // Profile Edit Elements
    const editProfileBtn = document.getElementById('editProfileBtn');
    const saveProfileChangesBtn = document.getElementById('saveProfileChangesBtn');
    const cancelProfileEditBtn = document.getElementById('cancelProfileEditBtn');
    const editProfileForm = document.getElementById('editProfileForm');
    const profileDisplayView = document.getElementById('profileDisplayView');
    const profileUsernameDisplay = document.getElementById('profileUsernameDisplay');

    const editProfileAgeInput = document.getElementById('editProfileAge');
    const editProfileGenderSelect = document.getElementById('editProfileGender');
    const editProfileHeightInput = document.getElementById('editProfileHeight');
    const editProfileWeightInput = document.getElementById('editProfileWeight');
    const editProfileDietPrefSelect = document.getElementById('editProfileDietPref');
    const editProfileActivityLevelSelect = document.getElementById('editProfileActivityLevel');
    const editProfileGoalsSelect = document.getElementById('editProfileGoals');
    const editProfileCuisinesContainer = document.getElementById('editProfileCuisinesContainer');
    const editProfileMessage = document.getElementById('editProfileMessage');


    // Diet Detail Elements
    const weeklyDietPlanTabsContainer = document.getElementById('weeklyDietPlanTabs');
    const dietChartContainer = document.getElementById('dietChartContainer');
    const regenerateWeeklyDietBtn = document.getElementById('regenerateWeeklyDietBtn');

    // Workout Detail Elements
    const workoutListContainer = document.getElementById('workoutListContainer');
    const workoutTimerDiv = document.getElementById('workoutTimer');
    const currentWorkoutNameTimerElement = document.getElementById('currentWorkoutNameTimer');
    const timerDisplay = document.getElementById('timerDisplay');
    const startTimerBtn = document.getElementById('startTimerBtn');
    const pauseTimerBtn = document.getElementById('pauseTimerBtn');
    const resetTimerBtn = document.getElementById('resetTimerBtn');
    const logThisWorkoutBtn = document.getElementById('logThisWorkoutBtn');
    const workoutTimerMessage = document.getElementById('workoutTimerMessage');


    // Pose Detection Elements
    const exerciseSelectForPose = document.getElementById('exerciseSelectForPose');
    const startPoseBtn = document.getElementById('startPoseBtn');
    const webcamFeed = document.getElementById('webcamFeed');
    const poseCanvas = document.getElementById('poseCanvas');
    const poseCanvasCtx = poseCanvas ? poseCanvas.getContext('2d') : null;
    const poseFeedback = document.getElementById('poseFeedback');
    const poseRepCount = document.getElementById('poseRepCount');

    // Logs Elements
    const logsList = document.getElementById('logsList');

    // To-Do List Elements
    const todoForm = document.getElementById('todoForm');
    const todoInput = document.getElementById('todoInput');
    const todoListUl = document.getElementById('todoList');

    // --- State Variables ---
    let currentUserData = null; // Will store { id, username, is_admin (bool), ... other profile details }
    let currentRawProfileData = null;
    let currentWeeklyDietPlan = null;
    let todos = [];
    let poseDetectionActive = false;
    let currentPoseInstance = null;
    let camera = null;
    let timerInterval;
    let timerSeconds = 0;
    let timerRunning = false;
    let currentWorkoutNameForTimer = "";
    let currentWorkoutDurationSuggestion = "0";
    let currentTimerSessionDetails = { name: null, startTime: null, durationSeconds: 0 };
    // Rep-counter state for pose detection
    let repCount = 0;
    let repState = ''; // e.g. 'extended' | 'down' | 'contracted'
    let lastRepTimestamp = 0;
    const repCooldownMs = 1000; // debounce between counted reps (ms)

    // Small debug toggle to enable extra console logs for pose values
    const POSE_DEBUG = false;

    // Simple smoothing (moving average) helper for recent numeric values
    const SMOOTH_FRAMES = 5;
    const _valueHistory = {};
    function smoothValue(key, value) {
        if (typeof value !== 'number' || !isFinite(value)) return value;
        if (!_valueHistory[key]) _valueHistory[key] = [];
        _valueHistory[key].push(value);
        if (_valueHistory[key].length > SMOOTH_FRAMES) _valueHistory[key].shift();
        const sum = _valueHistory[key].reduce((a, b) => a + b, 0);
        return sum / _valueHistory[key].length;
    }

    function avgVisibilityFor(...landmarksArr) {
        const vals = landmarksArr.map(l => (l && typeof l.visibility === 'number') ? l.visibility : 0);
        const rawAvg = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
        return smoothValue('vis_' + landmarksArr.map((l, i) => i).join('_'), rawAvg);
    }

    const TRANSITION_DURATION_CARD = 400;
    const TRANSITION_DURATION_DASHBOARD_SECTION = 500;
    const TRANSITION_DURATION_DETAIL_PAGE = 350;

    // Available Cuisines
    const availableCuisines = [
        "Italian", "Indian", "Mexican", "Chinese", "American", "Mediterranean",
        "Thai", "Japanese", "French", "Spanish", "Greek", "Vietnamese", "Korean", "Other"
    ];

    function populateCuisineCheckboxes(containerElement, selectedCuisinesString = "") {
        if (!containerElement) return;
        containerElement.innerHTML = '';
        const selectedArray = selectedCuisinesString ? selectedCuisinesString.split(',').map(c => c.trim().toLowerCase()) : [];
        const selectedSet = new Set(selectedArray);

        availableCuisines.forEach(cuisine => {
            const checkboxId = `cuisine-${cuisine.replace(/\s+/g, '-')}-${containerElement.id.replace('Container', '') || 'reg'}`;
            const label = document.createElement('label');
            label.htmlFor = checkboxId;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = checkboxId;
            checkbox.name = 'preferredCuisines';
            checkbox.value = cuisine;
            if (selectedSet.has(cuisine.toLowerCase())) {
                checkbox.checked = true;
            }

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` ${cuisine}`));
            containerElement.appendChild(label);
        });
    }

    if (regPreferredCuisinesContainer) {
        populateCuisineCheckboxes(regPreferredCuisinesContainer);
    }

    function showUserMessage(element, message, isError = false, autoClear = true) {
        let targetElement = element;
        if (!element && dashboardSection.classList.contains('view-active') && dashboardApiMessage) {
            targetElement = dashboardApiMessage;
        }

        if (targetElement) {
            targetElement.textContent = message;
            targetElement.className = isError ? 'message error' : 'message success';
            if (targetElement === dashboardApiMessage) {
                targetElement.style.backgroundColor = isError ? 'rgba(248, 215, 218, 0.9)' : 'rgba(212, 237, 218, 0.9)';
                targetElement.style.color = isError ? '#721c24' : '#155724';
                targetElement.style.padding = '10px';
                targetElement.style.borderRadius = '5px';
                targetElement.style.border = `1px solid ${isError ? '#f5c6cb' : '#c3e6cb'}`;
            }
            targetElement.style.display = 'block';
            if (autoClear) { setTimeout(() => { if (targetElement && targetElement.textContent === message) { clearUserMessage(targetElement); } }, 5000); }
        } else { console.warn("Message element not found for:", message); }
    }
    function clearUserMessage(element) {
        if (element) {
            element.textContent = ''; element.className = 'message'; element.style.display = 'none';
            if (element === dashboardApiMessage) {
                element.style.backgroundColor = 'transparent'; element.style.color = 'white';
                element.style.padding = '0'; element.style.border = 'none';
            }
        }
    }

    function showMainView(viewIdToShow) {
        allMainViews.forEach(view => {
            if (view) {
                if (view.id === viewIdToShow) {
                    view.style.display = 'block'; setTimeout(() => view.classList.add('view-active'), 10);
                } else {
                    view.classList.remove('view-active');
                    const duration = view.id === 'dashboardSection' ? TRANSITION_DURATION_DASHBOARD_SECTION : TRANSITION_DURATION_CARD;
                    setTimeout(() => { if (view && !view.classList.contains('view-active')) view.style.display = 'none'; }, duration + 50);
                }
            }
        });
        if (logoutBtn) logoutBtn.style.display = (viewIdToShow === 'dashboardSection') ? 'inline-block' : 'none';
        document.body.scrollTop = 0; document.documentElement.scrollTop = 0;

        // Admin Panel Card visibility
        if (adminPanelCard) {
            adminPanelCard.style.display = (currentUserData && currentUserData.is_admin && viewIdToShow === 'dashboardSection') ? 'block' : 'none';
        }
    }

    function showDashboardPage(pageIdToShow) {
        if (dashboardOverviewPage) dashboardOverviewPage.style.display = 'none';
        if (dashboardDetailContent) dashboardDetailContent.style.display = 'block';

        allDashboardDetailPages = document.querySelectorAll('#dashboardDetailContent > .dashboard-page'); // Re-query in case admin page was added

        allDashboardDetailPages.forEach(page => {
            if (page) {
                if (page.id === pageIdToShow) {
                    page.style.display = 'block'; setTimeout(() => page.classList.add('active-detail-page'), 10);
                } else {
                    page.classList.remove('active-detail-page');
                    setTimeout(() => { if (page && !page.classList.contains('active-detail-page')) page.style.display = 'none'; }, TRANSITION_DURATION_DETAIL_PAGE + 50);
                }
            }
        });
        const targetPageElement = document.getElementById(pageIdToShow);
        if (targetPageElement) targetPageElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (pageIdToShow === 'logsPage') { fetchAndDisplayWorkoutLogs_v2(); fetchAndDisplayCalorieCycle(); }
        if (pageIdToShow === 'posePage') populateExercisesForPose();
        if (pageIdToShow === 'adminPage') initAdminDashboard();
        if (pageIdToShow === 'profilePage') {
            if (editProfileForm) editProfileForm.style.display = 'none';
            if (profileDisplayView) profileDisplayView.style.display = 'block';
            if (editProfileBtn) editProfileBtn.style.display = 'inline-block';
        }
    }

    function showDashboardOverview() {
        if (dashboardDetailContent) {
            allDashboardDetailPages.forEach(page => { if (page) { page.classList.remove('active-detail-page'); page.style.display = 'none'; } });
            dashboardDetailContent.style.display = 'none';
        }
        if (dashboardOverviewPage) {
            dashboardOverviewPage.style.display = 'block';
            dashboardOverviewPage.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        // Ensure admin card is visible on overview if user is admin
        if (adminPanelCard) {
            adminPanelCard.style.display = (currentUserData && currentUserData.is_admin) ? 'block' : 'none';
        }
    }

    const routes = {
        'login': 'loginView', 'register': 'registerView', 'auth-choice': 'authChoiceView',
        'dashboard/profile': 'profilePage', 'dashboard/diet': 'dietPage',
        'dashboard/workouts': 'workoutsPage', 'dashboard/pose': 'posePage',
        'dashboard/logs': 'logsPage', 'dashboard': 'dashboardOverviewPage',
        'dashboard/admin': 'adminPage' // Admin route
    };
    const defaultInitialView = 'loginView'; const defaultDashboardRoute = 'dashboard';

    function router() {
        const hash = window.location.hash.substring(1);
        if (currentUserData) {
            showMainView('dashboardSection'); // This also handles admin card visibility
            const targetPageKey = hash || defaultDashboardRoute;
            const targetPageId = routes[targetPageKey];

            // Special check for admin page access
            if (targetPageId === 'adminPage' && (!currentUserData || !currentUserData.is_admin)) {
                showUserMessage(dashboardApiMessage, "Access Denied: Admin privileges required.", true);
                navigateTo(defaultDashboardRoute); // Redirect non-admins
                return;
            }

            if (targetPageId === 'dashboardOverviewPage') showDashboardOverview();
            else if (targetPageId && document.getElementById(targetPageId)) showDashboardPage(targetPageId);
            else showDashboardOverview();
        } else {
            const targetAuthViewId = routes[hash] || defaultInitialView;
            if (document.getElementById(targetAuthViewId)) showMainView(targetAuthViewId);
            else { showMainView(defaultInitialView); if (window.location.hash && hash !== 'login') window.location.hash = 'login'; }
        }
    }
    function navigateTo(routeKey) { window.location.hash = routeKey; }

    async function apiCall(endpoint, method = 'GET', body = null) {
        const options = { method, headers: {}, credentials: 'include' };
        if (method !== 'GET' && method !== 'HEAD') { options.headers['Content-Type'] = 'application/json'; }
        if (body && (method === 'POST' || method === 'PUT')) { options.body = JSON.stringify(body); }
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
            let responseData = {}; const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) { responseData = await response.json(); }
            else if (response.ok && response.status === 204) { return { message: "Operation successful (No Content)" }; }
            else if (!response.ok && response.status !== 204) {
                const errorText = await response.text();
                throw new Error(responseData.message || errorText || `HTTP error ${response.status}`);
            }
            if (!response.ok) {
                const errorMessage = responseData.message || response.statusText || `HTTP error ${response.status}`;
                throw new Error(errorMessage);
            }
            return responseData;
        } catch (error) {
            console.error(`API call to ${endpoint} (${method}) failed:`, error);
            const currentHash = window.location.hash.substring(1);
            let messageElement;
            if (currentHash === 'register' && registerMessage) messageElement = registerMessage;
            else if (currentHash === 'login' && loginMessage) messageElement = loginMessage;
            else if (currentHash === 'auth-choice' && authChoiceMessage) messageElement = authChoiceMessage;
            else if (editProfileForm && editProfileForm.style.display !== 'none' && editProfileMessage) messageElement = editProfileMessage;
            else if (workoutTimerDiv && workoutTimerDiv.style.display !== 'none' && workoutTimerMessage) messageElement = workoutTimerMessage;
            else if (dashboardSection && dashboardSection.classList.contains('view-active') && dashboardApiMessage) messageElement = dashboardApiMessage;
            else messageElement = loginMessage;

            if (messageElement) {
                showUserMessage(messageElement, `API Error: ${error.message}`, true, (messageElement !== dashboardApiMessage && messageElement !== editProfileMessage));
            }
            throw error;
        }
    }

    if (showLoginBtnFromChoice) showLoginBtnFromChoice.addEventListener('click', () => navigateTo('login'));
    if (showRegisterBtnFromChoice) showRegisterBtnFromChoice.addEventListener('click', () => navigateTo('register'));
    if (switchToRegisterBtn) switchToRegisterBtn.addEventListener('click', () => navigateTo('register'));
    if (switchToLoginBtn) switchToLoginBtn.addEventListener('click', () => navigateTo('login'));
    if (backToLoginFromRegister) backToLoginFromRegister.addEventListener('click', () => navigateTo('login'));

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault(); clearUserMessage(loginMessage);
            const email = loginForm.loginEmail.value; const password = loginForm.loginPassword.value;
            if (!email || !password) { showUserMessage(loginMessage, 'Email and password are required.', true); return; }
            try {
                const data = await apiCall('/login', 'POST', { email, password });
                currentUserData = data.user; // data.user is { id, username, is_admin }
                localStorage.setItem('Fit-GuideUser', JSON.stringify(currentUserData));

                if (currentUserData && currentUserData.is_admin) {
                    console.log("Admin user logged in:", currentUserData.username);
                    showUserMessage(dashboardApiMessage || loginMessage, "Admin login successful. Welcome, Abhinandan!", false, true);
                    if (adminPanelCard) adminPanelCard.style.display = 'block'; // Show admin card on overview
                } else if (currentUserData) {
                    console.log("Regular user logged in:", currentUserData.username);
                    if (adminPanelCard) adminPanelCard.style.display = 'none'; // Hide admin card
                }

                await fetchDashboardData(); // Fetches full profile including is_admin from backend
                navigateTo(defaultDashboardRoute);
            } catch (error) { /* error displayed by apiCall */ }
        });
    }
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault(); clearUserMessage(registerMessage);
            const selectedCuisines = [];
            if (regPreferredCuisinesContainer) {
                regPreferredCuisinesContainer.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
                    selectedCuisines.push(checkbox.value);
                });
            }
            const userData = {
                username: registerForm.regUsername.value, email: registerForm.regEmail.value, password: registerForm.regPassword.value,
                gender: registerForm.regGender.value, age: registerForm.regAge.value,
                height: registerForm.regHeight.value, weight: registerForm.regWeight.value,
                diet_preference: registerForm.regDietPreference.value,
                preferred_cuisines: selectedCuisines.join(','),
                activity_level: registerForm.regActivityLevel.value, goals: registerForm.regGoals.value,
            };
            for (const key in userData) {
                if (key === 'preferred_cuisines') continue;
                if (!userData[key] && key !== 'password') {
                    showUserMessage(registerMessage, `Please fill in the "${key.replace('reg', '').replace(/([A-Z])/g, ' $1').toLowerCase()}" field.`, true); return;
                }
            }
            if (userData.password.length < 6) {
                showUserMessage(registerMessage, `Password must be at least 6 characters.`, true); return;
            }
            try {
                const data = await apiCall('/register', 'POST', userData);
                showUserMessage(loginMessage, data.message + " Please login.", false);
                registerForm.reset();
                populateCuisineCheckboxes(regPreferredCuisinesContainer);
                navigateTo('login');
            } catch (error) { /* error displayed by apiCall */ }
        });
    }
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async () => {
            try { await apiCall('/logout', 'POST'); } catch (error) { console.error("Logout API call failed:", error); }
            finally {
                currentUserData = null; localStorage.removeItem('Fit-GuideUser');
                currentRawProfileData = null; currentWeeklyDietPlan = null; todos = [];
                if (adminPanelCard) adminPanelCard.style.display = 'none'; // Hide admin card on logout
                navigateTo('login');
                showUserMessage(loginMessage, 'Logged out successfully.', false);
            }
        });
    }

    // Re-initialize dashboardNavItems after potential admin card addition
    function initializeDashboardNavItems() {
        document.querySelectorAll('.dashboard-nav-item').forEach(item => {
            // Remove any existing listeners to prevent duplicates if this is called multiple times
            item.replaceWith(item.cloneNode(true)); // Simple way to remove listeners
        });
        // Add new listeners
        document.querySelectorAll('.dashboard-nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const targetPageId = item.dataset.targetPage;
                if (targetPageId) {
                    if (targetPageId === 'adminPage' && (!currentUserData || !currentUserData.is_admin)) {
                        showUserMessage(dashboardApiMessage, "Access Denied.", true);
                        return; // Don't navigate
                    }
                    navigateTo(`dashboard/${targetPageId.replace('Page', '').toLowerCase()}`);
                }
            });
        });
    }


    if (backToDashboardOverviewBtn) { backToDashboardOverviewBtn.addEventListener('click', () => navigateTo(defaultDashboardRoute)); }
    if (headerTitle) { headerTitle.addEventListener('click', () => { if (currentUserData) navigateTo(defaultDashboardRoute); else navigateTo('login'); }); }

    else { showUserMessage(dashboardApiMessage || loginMessage, `Error loading dashboard: ${error.message}`, true, false); }
}
    }
    function updateProfileDisplay(profile) {
        if (!profile || Object.keys(profile).length === 0) { console.warn("UpdateProfileDisplay: Profile data is empty/null."); return; }
        currentRawProfileData = profile;

        if (usernameDisplay) usernameDisplay.textContent = profile.username || 'User';
        if (profileUsername) profileUsername.textContent = profile.username || '-';
        if (profileAge) profileAge.textContent = profile.age || '-';
        if (profileGender) profileGender.textContent = profile.gender ? profile.gender.charAt(0).toUpperCase() + profile.gender.slice(1) : '-';
        if (profileHeight) profileHeight.textContent = profile.height_cm ? `${profile.height_cm}` : '-';
        if (profileWeight) profileWeight.textContent = profile.weight_kg ? `${profile.weight_kg}` : '-';
        if (profileDietPref) profileDietPref.textContent = profile.diet_preference ? profile.diet_preference.charAt(0).toUpperCase() + profile.diet_preference.slice(1) : '-';
        if (profileActivityLevel) profileActivityLevel.textContent = profile.activity_level ? profile.activity_level.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : '-';
        if (profileGoals) profileGoals.textContent = profile.goals ? profile.goals.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : '-';
        if (profileCuisines) profileCuisines.textContent = profile.preferred_cuisines || 'Not set';

        if (profileBmiValue) profileBmiValue.textContent = profile.bmi !== undefined && profile.bmi !== null ? profile.bmi.toFixed(1) : '-';
        if (profileBmiCategory) profileBmiCategory.textContent = profile.bmi_category || '-';
        if (profileBmr) profileBmr.textContent = profile.bmr ? `${Math.round(profile.bmr)}` : '-';
        if (profileTdee) profileTdee.textContent = profile.tdee ? `${Math.round(profile.tdee)}` : '-';
        if (profileTargetCalories) profileTargetCalories.textContent = profile.target_daily_calories ? `${Math.round(profile.target_daily_calories)}` : '-';

        const bmiValueCard = document.getElementById('bmiValue');
        const bmiCategoryCard = document.getElementById('bmiCategory');
        const targetCaloriesDisplayCard = document.getElementById('targetCaloriesDisplay');
        if (bmiValueCard) bmiValueCard.textContent = profile.bmi !== undefined && profile.bmi !== null ? profile.bmi.toFixed(1) : '-';
        if (bmiCategoryCard) bmiCategoryCard.textContent = profile.bmi_category || '-';
        if (targetCaloriesDisplayCard) targetCaloriesDisplayCard.textContent = profile.target_daily_calories ? `${Math.round(profile.target_daily_calories)}` : '-';
    }

    if (editProfileBtn) {
    editProfileBtn.addEventListener('click', () => {
        if (!currentRawProfileData) { showUserMessage(editProfileMessage, "Profile data not loaded.", true); return; }
        if (profileUsernameDisplay) profileUsernameDisplay.textContent = currentRawProfileData.username;
        if (editProfileAgeInput) editProfileAgeInput.value = currentRawProfileData.age || '';
        if (editProfileGenderSelect) editProfileGenderSelect.value = currentRawProfileData.gender || 'male';
        if (editProfileHeightInput) editProfileHeightInput.value = currentRawProfileData.height_cm || '';
        if (editProfileWeightInput) editProfileWeightInput.value = currentRawProfileData.weight_kg || '';
        if (editProfileDietPrefSelect) editProfileDietPrefSelect.value = currentRawProfileData.diet_preference || 'any';
        if (editProfileActivityLevelSelect) editProfileActivityLevelSelect.value = currentRawProfileData.activity_level || 'sedentary';
        if (editProfileGoalsSelect) editProfileGoalsSelect.value = currentRawProfileData.goals || 'maintenance';
        populateCuisineCheckboxes(editProfileCuisinesContainer, currentRawProfileData.preferred_cuisines || "");
        if (profileDisplayView) profileDisplayView.style.display = 'none';
        if (editProfileForm) editProfileForm.style.display = 'block';
        editProfileBtn.style.display = 'none';
        clearUserMessage(editProfileMessage);
    });
}
if (cancelProfileEditBtn) {
    cancelProfileEditBtn.addEventListener('click', () => {
        if (profileDisplayView) profileDisplayView.style.display = 'block';
        if (editProfileForm) editProfileForm.style.display = 'none';
        if (editProfileBtn) editProfileBtn.style.display = 'inline-block';
        clearUserMessage(editProfileMessage);
    });
}
if (editProfileForm) {
    editProfileForm.addEventListener('submit', async (e) => {
        e.preventDefault(); clearUserMessage(editProfileMessage);
        const selectedCuisines = [];
        if (editProfileCuisinesContainer) {
            editProfileCuisinesContainer.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
                selectedCuisines.push(checkbox.value);
            });
        }
        const updatedData = {
            age: parseInt(editProfileAgeInput.value), gender: editProfileGenderSelect.value,
            height: parseFloat(editProfileHeightInput.value), weight: parseFloat(editProfileWeightInput.value),
            diet_preference: editProfileDietPrefSelect.value, activity_level: editProfileActivityLevelSelect.value,
            goals: editProfileGoalsSelect.value, preferred_cuisines: selectedCuisines.join(','),
        };
        // Basic Frontend Validation
        if (isNaN(updatedData.age) || updatedData.age < 12 || updatedData.age > 120) { showUserMessage(editProfileMessage, "Valid age (12-120) is required.", true); return; }
        if (isNaN(updatedData.height) || updatedData.height < 50 || updatedData.height > 300) { showUserMessage(editProfileMessage, "Valid height (50-300cm) is required.", true); return; }
        if (isNaN(updatedData.weight) || updatedData.weight < 20 || updatedData.weight > 500) { showUserMessage(editProfileMessage, "Valid weight (20-500kg) is required.", true); return; }

        try {
            if (saveProfileChangesBtn) { saveProfileChangesBtn.textContent = 'Saving...'; saveProfileChangesBtn.disabled = true; }
            const response = await apiCall('/user_profile', 'PUT', updatedData);

            currentRawProfileData = response.user_profile;
            const isAdminStatus = currentUserData.is_admin; // Preserve admin status
            currentUserData = { ...response.user_profile, is_admin: isAdminStatus };
            localStorage.setItem('Fit-GuideUser', JSON.stringify(currentUserData));

            updateProfileDisplay(response.user_profile);
            showUserMessage(editProfileMessage, response.message || "Profile updated successfully!", false);
            if (profileDisplayView) profileDisplayView.style.display = 'block';
            if (editProfileForm) editProfileForm.style.display = 'none';
            if (editProfileBtn) editProfileBtn.style.display = 'inline-block';
        } catch (error) { /* apiCall handles message */ }
        finally { if (saveProfileChangesBtn) { saveProfileChangesBtn.textContent = 'Save Changes'; saveProfileChangesBtn.disabled = false; } }
    });
}

function setupDayTabsForDietPlan() { /* ... (same as before) ... */ if (!weeklyDietPlanTabsContainer || !currentWeeklyDietPlan || currentWeeklyDietPlan.length === 0) return; weeklyDietPlanTabsContainer.innerHTML = ''; currentWeeklyDietPlan.forEach((dayData, index) => { const dayButton = document.createElement('button'); dayButton.textContent = `Day ${dayData.day}`; dayButton.classList.add('day-tab'); if (index === 0) dayButton.classList.add('active'); dayButton.addEventListener('click', () => { displayWeeklyDietPlan(index); weeklyDietPlanTabsContainer.querySelectorAll('.day-tab').forEach(btn => btn.classList.remove('active')); dayButton.classList.add('active'); }); weeklyDietPlanTabsContainer.appendChild(dayButton); }); }
function displayWeeklyDietPlan(dayIndexToShow) { /* ... (same as before, ensure meal types include 'snacks' if your data has it) ... */ if (!dietChartContainer || !currentWeeklyDietPlan || !currentWeeklyDietPlan[dayIndexToShow]) { if (dietChartContainer) dietChartContainer.innerHTML = '<p>No diet plan data for this day.</p>'; return; } dietChartContainer.innerHTML = ''; const dayData = currentWeeklyDietPlan[dayIndexToShow]; const { meals, total_calories_for_day } = dayData.daily_summary; if (!meals || Object.keys(meals).length === 0) { dietChartContainer.innerHTML = `<p>No meal details for Day ${dayData.day}.</p>`; return; } let html = `<h4 style="text-align:center; margin-bottom: 15px;">Day ${dayData.day}: Approx. ${total_calories_for_day ? Math.round(total_calories_for_day) : 0} kcal</h4>`; const mealOrder = ['breakfast', 'lunch', 'dinner', 'snacks']; mealOrder.forEach(mealType => { if (meals[mealType]) { const optionsList = meals[mealType]; html += `<div class="meal-type-section"><h5>${mealType.charAt(0).toUpperCase() + mealType.slice(1)}</h5>`; if (optionsList && optionsList.length > 0 && !(optionsList.length === 1 && String(optionsList[0].name).startsWith("N/A"))) { html += '<ul>'; optionsList.forEach((meal, index) => { html += `<li><strong>${optionsList.length > 1 ? `Option ${index + 1}: ` : ''}${meal.name || 'N/A'}</strong> (${meal.calories ? Math.round(meal.calories) : 0} kcal) <br><small>Cuisine: ${meal.cuisine || 'N/A'} | P: ${meal.protein ? Math.round(meal.protein) : 0}g, C: ${meal.carbs ? Math.round(meal.carbs) : 0}g, F: ${meal.fat ? Math.round(meal.fat) : 0}g</small></li>`; }); html += '</ul>'; } else { html += '<p>No specific options found for this meal.</p>'; } html += `</div>`; } }); dietChartContainer.innerHTML = html; }
if (regenerateWeeklyDietBtn) {
    regenerateWeeklyDietBtn.addEventListener('click', async () => {
        regenerateWeeklyDietBtn.textContent = 'Generating...';
        regenerateWeeklyDietBtn.disabled = true;
        if (dietChartContainer) dietChartContainer.innerHTML = '<p>Generating new diet plan using AI recommendations...</p>';
        if (weeklyDietPlanTabsContainer) weeklyDietPlanTabsContainer.innerHTML = '';
        try {
            const newWeeklyDiet = await apiCall('/knn_diet_plan');
            currentWeeklyDietPlan = newWeeklyDiet.weekly_diet_plan;
            displayWeeklyDietPlan(0);
            setupDayTabsForDietPlan();
        } catch (error) {
            if (dietChartContainer) dietChartContainer.innerHTML = `<p class="message error">Could not regenerate diet plan: ${error.message}</p>`;
        } finally {
            regenerateWeeklyDietBtn.textContent = 'Regenerate Full Week Plan';
            regenerateWeeklyDietBtn.disabled = false;
        }
    });
}

function populateExercisesForPose() {
    if (!exerciseSelectForPose) return;
    exerciseSelectForPose.innerHTML = '<option value="">-- Select Exercise for Pose Check --</option>';

    const defaultExercises = [
        { value: 'wrist rotation', text: 'Wrist Rotation' },
        { value: 'squat', text: 'Squat' },
        { value: 'bicep curl', text: 'Bicep Curl' },
        { value: 'plank', text: 'Plank' }
    ];

    defaultExercises.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        exerciseSelectForPose.appendChild(option);
    });

    if (startPoseBtn) startPoseBtn.disabled = (exerciseSelectForPose.options.length <= 1);
}

function displayWorkouts(workouts) { /* ... (same as before, ensure commonExercises list is good) ... */
    if (!workoutListContainer || !exerciseSelectForPose) return;
    workoutListContainer.innerHTML = '';
    exerciseSelectForPose.innerHTML = '<option value="">-- Select Exercise for Pose Check --</option>';

    // Always provide a few common exercise options so pose testing works without backend data
    const defaultExercises = [
        { value: 'wrist rotation', text: 'Wrist Rotation' },
        { value: 'squat', text: 'Squat' },
        { value: 'bicep curl', text: 'Bicep Curl' },
        { value: 'plank', text: 'Plank' }
    ];
    defaultExercises.forEach(opt => {
        const exists = Array.from(exerciseSelectForPose.options).some(o => o.value === opt.value);
        if (!exists) {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.text;
            exerciseSelectForPose.appendChild(option);
        }
    });

    if (workouts && workouts.length > 0) {
        let html = '<ul>';
        workouts.forEach(workout => {
            html += `<li><strong>${workout.name}</strong> (${workout.type || 'N/A'}) <br>
                <small>Target: ${workout.target || 'N/A'} | Suggestion: ${workout.duration_suggestion || 'N/A'} </small><br>
                <small style="color: #28a745; font-weight: bold;">Estimated Burn: ${workout.estimated_calories ? '~' + workout.estimated_calories + ' kcal' : 'N/A'}</small>
                <button class="start-workout-btn" data-duration-suggestion="${workout.duration_suggestion || '15 min'}" data-name="${workout.name}">Start Timer</button></li>`;
            const commonExercises = ['squat', 'push-up', 'lunge', 'plank', 'bicep curl', 'overhead press', 'burpee', 'jumping jack', 'row', 'crunch', 'leg raise'];
            if (commonExercises.some(ex => workout.name.toLowerCase().includes(ex))) {
                const option = document.createElement('option'); option.value = workout.name.toLowerCase(); option.textContent = workout.name; exerciseSelectForPose.appendChild(option);
            }
        });
        html += '</ul>';
        workoutListContainer.innerHTML = html;
        document.querySelectorAll('#workoutListContainer .start-workout-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                currentWorkoutDurationSuggestion = e.target.dataset.durationSuggestion; currentWorkoutNameForTimer = e.target.dataset.name; currentTimerSessionDetails = { name: currentWorkoutNameForTimer, startTime: Date.now(), durationSeconds: 0 }; if (currentWorkoutNameTimerElement) currentWorkoutNameTimerElement.textContent = currentWorkoutNameForTimer; const match = currentWorkoutDurationSuggestion.match(/(\d+)\s*min/); const durationMinutes = match ? parseInt(match[1]) : 15; resetTimer(durationMinutes * 60); if (workoutTimerDiv) workoutTimerDiv.style.display = 'block'; if (logThisWorkoutBtn) logThisWorkoutBtn.style.display = 'inline-block'; clearUserMessage(workoutTimerMessage);
            });
        });
    } else {
        workoutListContainer.innerHTML = '<li>No specific workout recommendations found. Please check back later or ensure your profile goals are set.</li>';
    }

    if (startPoseBtn) startPoseBtn.disabled = (exerciseSelectForPose.options.length <= 1);
}
function formatTime(totalSeconds) { /* ... (same) ... */ const minutes = Math.floor(totalSeconds / 60); const seconds = totalSeconds % 60; return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`; }
function updateTimerDisplay() { /* ... (same) ... */ if (timerDisplay) timerDisplay.textContent = formatTime(timerSeconds); }
if (startTimerBtn) startTimerBtn.addEventListener('click', () => { /* ... (same) ... */ if (!timerRunning) { timerRunning = true; startTimerBtn.style.display = 'none'; if (pauseTimerBtn) pauseTimerBtn.style.display = 'inline-block'; timerInterval = setInterval(() => { timerSeconds++; if (currentTimerSessionDetails) currentTimerSessionDetails.durationSeconds = timerSeconds; updateTimerDisplay(); if (timerSeconds > 5 && logThisWorkoutBtn && logThisWorkoutBtn.style.display === 'none') { logThisWorkoutBtn.style.display = 'inline-block'; } }, 1000); clearUserMessage(workoutTimerMessage); } });
if (pauseTimerBtn) pauseTimerBtn.addEventListener('click', () => { /* ... (same) ... */ if (timerRunning) { timerRunning = false; clearInterval(timerInterval); if (startTimerBtn) { startTimerBtn.style.display = 'inline-block'; startTimerBtn.textContent = 'Resume'; } if (pauseTimerBtn) pauseTimerBtn.style.display = 'none'; } });
function resetTimer(initialSeconds = 0) { /* ... (same) ... */ timerRunning = false; clearInterval(timerInterval); timerSeconds = initialSeconds; updateTimerDisplay(); if (startTimerBtn) { startTimerBtn.style.display = 'inline-block'; startTimerBtn.textContent = 'Start'; } if (pauseTimerBtn) pauseTimerBtn.style.display = 'none'; if (logThisWorkoutBtn) { if (initialSeconds > 0 || (currentTimerSessionDetails && currentTimerSessionDetails.durationSeconds > 0)) { logThisWorkoutBtn.style.display = 'inline-block'; } else { logThisWorkoutBtn.style.display = 'none'; } } clearUserMessage(workoutTimerMessage); }
if (resetTimerBtn) resetTimerBtn.addEventListener('click', () => { /* ... (same) ... */ const match = currentWorkoutDurationSuggestion.match(/(\d+)\s*min/); const durationMinutes = match ? parseInt(match[1]) : 0; currentTimerSessionDetails = { name: currentWorkoutNameForTimer, startTime: Date.now(), durationSeconds: 0 }; resetTimer(durationMinutes * 60); if (logThisWorkoutBtn) logThisWorkoutBtn.style.display = 'inline-block'; });

function onPoseResults(results) { /* ... (same, check Pose.POSE_LANDMARKS) ... */ if (!poseCanvasCtx || !poseCanvas) return; poseCanvasCtx.save(); poseCanvasCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height); if (results.poseLandmarks) { if (window.drawConnectors && window.POSE_CONNECTIONS && window.drawLandmarks) { drawConnectors(poseCanvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 }); drawLandmarks(poseCanvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 1, radius: 3 }); } analyzePoseWithReps(results.poseLandmarks); } poseCanvasCtx.restore(); }

// New pose analysis that includes rep counting for common exercises
function analyzePoseWithReps(landmarks) {
    if (!poseFeedback || !exerciseSelectForPose) return;
    const selectedExercise = exerciseSelectForPose.value;
    let feedbackMsg = "Position yourself for feedback.";
    let correctPosture = false;
    if (typeof POSE_LANDMARKS === 'undefined' || !POSE_LANDMARKS) {
        if (poseFeedback) poseFeedback.textContent = "Pose landmarks not loaded.";
        return;
    }

    // Debug: log selected exercise and landmarks
    console.log("=== POSE ANALYSIS ===");
    console.log("Selected exercise:", selectedExercise, "Landmarks count:", landmarks.length);
    console.log("Current repState:", repState, "Current repCount:", repCount);

    function updateRepDisplay() {
        if (!poseRepCount) return;
        poseRepCount.textContent = `Reps: ${repCount}`;
        poseRepCount.style.display = 'block';
        // Ensure feedback is visible when a rep is counted
        if (poseFeedback) { poseFeedback.style.display = 'block'; }
        console.log("✓ REP COUNTED! Total reps now:", repCount);
    }

    if (selectedExercise.includes('squat')) {
        const hip = landmarks[POSE_LANDMARKS.LEFT_HIP];
        const knee = landmarks[POSE_LANDMARKS.LEFT_KNEE];
        const ankle = landmarks[POSE_LANDMARKS.LEFT_ANKLE];
        const hipVis = hip ? (hip.visibility || 0) : 0;
        const kneeVis = knee ? (knee.visibility || 0) : 0;
        const ankleVis = ankle ? (ankle.visibility || 0) : 0;
        const avgVis = avgVisibilityFor(hip, knee, ankle);
        if (POSE_DEBUG) console.log("SQUAT - Raw vis:", hipVis.toFixed(2), kneeVis.toFixed(2), ankleVis.toFixed(2), "Smoothed avg:", avgVis.toFixed(2));

        if (hip && knee && ankle && avgVis > 0.55) {
            // Calculate knee angle
            const kneeVector = { x: ankle.x - knee.x, y: ankle.y - knee.y };
            const hipVector = { x: hip.x - knee.x, y: hip.y - knee.y };
            const dotProduct = kneeVector.x * hipVector.x + kneeVector.y * hipVector.y;
            let kneeAngle = Math.acos(Math.max(-1, Math.min(1, dotProduct / (Math.sqrt(kneeVector.x ** 2 + kneeVector.y ** 2) * Math.sqrt(hipVector.x ** 2 + hipVector.y ** 2))))) * (180 / Math.PI);
            kneeAngle = smoothValue('squat_knee_angle', kneeAngle);
            if (POSE_DEBUG) console.log("SQUAT - Knee angle (smoothed):", kneeAngle.toFixed(1), "°");

            // Determine posture based on depth (smoothed)
            const yDiff = knee.y - hip.y;
            const yDiffSm = smoothValue('squat_yDiff', yDiff);
            if (yDiffSm > 0.07) {
                feedbackMsg = "Good squat depth!";
                correctPosture = true;
            } else if (yDiffSm <= 0) {
                feedbackMsg = "Go deeper with your hips.";
            } else {
                feedbackMsg = "Keep chest up, back straight.";
            }

            // Rep counting using smoothed knee angle
            const now = Date.now();
            if (kneeAngle > 150) {
                if (repState === 'down' && (now - lastRepTimestamp) > repCooldownMs) {
                    if (POSE_DEBUG) console.log("SQUAT REP TRIGGER: Extended after down");
                    repCount += 1; lastRepTimestamp = now; updateRepDisplay();
                }
                repState = 'extended';
            } else if (kneeAngle < 95) {
                repState = 'down';
            }
            if (POSE_DEBUG) console.log("SQUAT - New repState:", repState);
        } else {
            feedbackMsg = "Squat landmarks not clear. Ensure side view.";
            if (POSE_DEBUG) console.log("SQUAT - Not all landmarks visible or low visibility", hipVis, kneeVis, ankleVis, "avgVis", avgVis);
        }

    } else if (selectedExercise.includes('plank')) {
        const shoulderL = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
        const hipL = landmarks[POSE_LANDMARKS.LEFT_HIP];
        const ankleL = landmarks[POSE_LANDMARKS.LEFT_ANKLE];
        const avgVisPlank = avgVisibilityFor(shoulderL, hipL, ankleL);
        if (POSE_DEBUG) console.log("PLANK - vis avg:", avgVisPlank.toFixed(2));
        if (shoulderL && hipL && ankleL && avgVisPlank > 0.55) {
            const canvasH = poseCanvas ? poseCanvas.height : 480;
            const yThreshold = 0.08 * canvasH;
            const shoulderHipDiff = smoothValue('plank_shHip', Math.abs(shoulderL.y - hipL.y));
            const hipAnkleDiff = smoothValue('plank_hipAnk', Math.abs(hipL.y - ankleL.y));
            if (POSE_DEBUG) console.log("PLANK - shHip, hipAnk:", shoulderHipDiff.toFixed(3), hipAnkleDiff.toFixed(3), "thresh", yThreshold.toFixed(3));

            if (shoulderHipDiff < yThreshold && hipAnkleDiff < yThreshold) {
                feedbackMsg = "Good plank form! Core tight."; correctPosture = true;
            } else if (hipL.y < Math.min(shoulderL.y, ankleL.y) - yThreshold * 0.7) {
                feedbackMsg = "Hips too high! Lower them.";
            } else if (hipL.y > Math.max(shoulderL.y, ankleL.y) + yThreshold * 0.7) {
                feedbackMsg = "Hips sagging! Lift them.";
            } else {
                feedbackMsg = "Straighten your back.";
            }
        } else {
            feedbackMsg = "Plank landmarks not clear. Side view needed.";
            if (POSE_DEBUG) console.log("PLANK - low vis", avgVisPlank);
        }

    } else if (selectedExercise.includes('bicep curl')) {
        const shoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
        const elbow = landmarks[POSE_LANDMARKS.LEFT_ELBOW];
        const wrist = landmarks[POSE_LANDMARKS.LEFT_WRIST];
        const avgVisBicep = avgVisibilityFor(shoulder, elbow, wrist);
        if (POSE_DEBUG) console.log("BICEP CURL - vis avg:", avgVisBicep.toFixed(2));
        if (shoulder && elbow && wrist && avgVisBicep > 0.6) {
            const upperArm = { x: elbow.x - shoulder.x, y: elbow.y - shoulder.y };
            const forearm = { x: wrist.x - elbow.x, y: wrist.y - elbow.y };
            const dotProduct = upperArm.x * forearm.x + upperArm.y * forearm.y;
            let angle = Math.acos(Math.max(-1, Math.min(1, dotProduct / (Math.sqrt(upperArm.x ** 2 + upperArm.y ** 2) * Math.sqrt(forearm.x ** 2 + forearm.y ** 2))))) * (180 / Math.PI);
            angle = smoothValue('bicep_angle', angle);
            if (POSE_DEBUG) console.log("BICEP CURL - Elbow angle (smoothed):", angle.toFixed(1));

            if (angle < 55) { feedbackMsg = "Full curl! Good contraction."; correctPosture = true; }
            else if (angle > 160) { feedbackMsg = "Arm extended. Ready to curl."; }
            else if (angle >= 60 && angle <= 160) { feedbackMsg = "Curling... good range of motion."; }
            else { feedbackMsg = "Keep elbow stable."; }

            const now = Date.now();
            if (angle > 155) {
                if (repState === 'contracted' && (now - lastRepTimestamp) > repCooldownMs) { if (POSE_DEBUG) console.log("BICEP CURL REP TRIGGER"); repCount += 1; lastRepTimestamp = now; updateRepDisplay(); }
                repState = 'extended';
            } else if (angle < 60) {
                repState = 'contracted';
            }
            if (POSE_DEBUG) console.log("BICEP CURL - New repState:", repState);
        } else {
            feedbackMsg = "Bicep curl landmarks not clear.";
            if (POSE_DEBUG) console.log("BICEP CURL - low vis", avgVisBicep);
        }

    } else if (selectedExercise.includes('wrist rotation')) {
        const wrist = landmarks[POSE_LANDMARKS.LEFT_WRIST];
        const elbow = landmarks[POSE_LANDMARKS.LEFT_ELBOW];
        const avgVisWrist = avgVisibilityFor(wrist, elbow);
        if (POSE_DEBUG) console.log("WRIST ROTATION - vis avg:", avgVisWrist.toFixed(2));
        if (wrist && elbow && avgVisWrist > 0.35) {
            // Simple wrist rotation: track x-axis movement of wrist relative to elbow
            let relativeX = wrist.x - elbow.x;
            relativeX = smoothValue('wrist_relx', relativeX);
            if (POSE_DEBUG) console.log("WRIST ROTATION - Relative X (smoothed):", relativeX.toFixed(3));

            feedbackMsg = "Rotating wrist...";
            correctPosture = true;

            const now = Date.now();
            // Left side: relativeX < -0.12, Right side: relativeX > 0.12
            if (relativeX < -0.12) {
                if (repState === 'right' && (now - lastRepTimestamp) > repCooldownMs) {
                    if (POSE_DEBUG) console.log("WRIST ROTATION REP TRIGGER: Moved left after being right");
                    repCount += 1; lastRepTimestamp = now; updateRepDisplay();
                }
                repState = 'left';
            } else if (relativeX > 0.12) {
                if (repState === 'left' && (now - lastRepTimestamp) > repCooldownMs) {
                    if (POSE_DEBUG) console.log("WRIST ROTATION REP TRIGGER: Moved right after being left");
                    repCount += 1; lastRepTimestamp = now; updateRepDisplay();
                }
                repState = 'right';
            }
            if (POSE_DEBUG) console.log("WRIST ROTATION - New repState:", repState);
        } else {
            feedbackMsg = "Wrist landmarks not clear. Raise your arm slightly.";
            if (POSE_DEBUG) console.log("WRIST ROTATION - low vis", avgVisWrist);
        }

    } else if (selectedExercise) {
        feedbackMsg = `Pose analysis for ${selectedExercise} not implemented yet.`;
        console.log("NO EXERCISE MATCHED - Exercise:", selectedExercise);
    }

    if (poseFeedback) {
        poseFeedback.textContent = feedbackMsg;
        poseFeedback.className = correctPosture ? 'message success' : 'message error';
        if (poseFeedback.style.display !== 'block') poseFeedback.style.display = 'block';
    }
    console.log("===================\n");
}
if (startPoseBtn) startPoseBtn.addEventListener('click', () => {
    if (!poseDetectionActive) {
        if (!exerciseSelectForPose || !exerciseSelectForPose.value) { alert("Please select an exercise first!"); return; }
        if (typeof Pose === 'undefined' || typeof Camera === 'undefined' || !webcamFeed || !poseCanvas) { alert("MediaPipe components not fully loaded or HTML elements missing."); return; }

        poseDetectionActive = true;
        // Reset rep counter state and ensure UI shows counter
        repState = ''; repCount = 0; lastRepTimestamp = 0;
        if (poseRepCount) { poseRepCount.style.display = 'block'; poseRepCount.textContent = 'Reps: 0'; }
        // Ensure feedback box is visible when detection starts
        if (poseFeedback) { poseFeedback.style.display = 'block'; poseFeedback.textContent = 'Position yourself for feedback.'; poseFeedback.className = 'message'; }

        startPoseBtn.textContent = 'Stop Pose Detection';
        webcamFeed.style.display = 'block';
        poseCanvas.style.display = 'block';
        webcamFeed.onloadedmetadata = () => { if (poseCanvas) { poseCanvas.width = webcamFeed.videoWidth; poseCanvas.height = webcamFeed.videoHeight; } };

        currentPoseInstance = new Pose({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });
        currentPoseInstance.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        currentPoseInstance.onResults(onPoseResults);

        camera = new Camera(webcamFeed, {
            onFrame: async () => {
                if (webcamFeed.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && currentPoseInstance) {
                    try { await currentPoseInstance.send({ image: webcamFeed }); }
                    catch (err) { console.error("Error sending frame to MediaPipe", err); }
                }
            },
            width: 640, height: 480
        });

        camera.start().then(() => {
            if (poseFeedback) { poseFeedback.textContent = "Pose detection started. Position yourself."; poseFeedback.className = 'message'; }
        }).catch(err => {
            console.error("Error starting camera:", err);
            if (poseFeedback) { poseFeedback.textContent = "Error starting camera. Check permissions."; poseFeedback.className = 'message error'; }
            stopPoseDetectionInternal();
        });
    } else {
        stopPoseDetectionInternal();
    }
});
function stopPoseDetectionInternal() { /* ... (same) ... */
    poseDetectionActive = false;
    if (startPoseBtn) startPoseBtn.textContent = 'Start Pose Detection';
    if (camera) { camera.stop(); camera = null; }
    if (currentPoseInstance) { currentPoseInstance.close(); currentPoseInstance = null; }
    if (webcamFeed) { webcamFeed.style.display = 'none'; if (webcamFeed.srcObject) { webcamFeed.srcObject.getTracks().forEach(track => track.stop()); webcamFeed.srcObject = null; } }
    if (poseCanvas) poseCanvas.style.display = 'none';
    if (poseFeedback) { poseFeedback.textContent = "Pose detection stopped."; poseFeedback.className = 'message'; }
    // hide and reset rep counter state
    if (poseRepCount) { poseRepCount.style.display = 'none'; poseRepCount.textContent = 'Reps: 0'; }
    repState = ''; repCount = 0; lastRepTimestamp = 0;
}

async function renderTodos() { /* ... (same) ... */ if (!todoListUl) return; todoListUl.innerHTML = ''; if (todos.length === 0) { todoListUl.innerHTML = '<li>No tasks yet. Add one!</li>'; return; } todos.forEach((todo) => { const li = document.createElement('li'); const taskSpan = document.createElement('span'); taskSpan.textContent = todo.task; taskSpan.className = 'task-text'; if (todo.completed) li.classList.add('completed'); const actionsDiv = document.createElement('div'); actionsDiv.className = 'todo-actions'; const completeBtn = document.createElement('button'); completeBtn.textContent = todo.completed ? 'Undo' : 'Done'; completeBtn.classList.add('complete-btn'); completeBtn.addEventListener('click', () => toggleTodoItem(todo.id)); const deleteBtn = document.createElement('button'); deleteBtn.textContent = 'Delete'; deleteBtn.classList.add('delete-btn'); deleteBtn.addEventListener('click', () => deleteTodoItem(todo.id)); actionsDiv.appendChild(completeBtn); actionsDiv.appendChild(deleteBtn); li.appendChild(taskSpan); li.appendChild(actionsDiv); todoListUl.appendChild(li); }); }
if (todoForm) { todoForm.addEventListener('submit', async (e) => { /* ... (same) ... */ e.preventDefault(); if (!todoInput) return; const taskText = todoInput.value.trim(); if (taskText && currentUserData) { try { const newTodo = await apiCall('/todos', 'POST', { task: taskText }); todos.push(newTodo); todoInput.value = ''; renderTodos(); } catch (error) { /* Handled */ } } else if (!taskText) { alert("Task cannot be empty."); } }); }
async function toggleTodoItem(todoId) { /* ... (same) ... */ const todo = todos.find(t => t.id === todoId); if (todo && currentUserData) { try { const updatedTodo = await apiCall(`/todos/${todoId}`, 'PUT', { completed: !todo.completed, task: todo.task }); const todoIndex = todos.findIndex(t => t.id === todoId); if (todoIndex > -1) todos[todoIndex] = updatedTodo; renderTodos(); } catch (error) { /* Handled */ } } }
async function deleteTodoItem(todoId) { /* ... (same) ... */ if (confirm("Are you sure you want to delete this task?") && currentUserData) { try { await apiCall(`/todos/${todoId}`, 'DELETE'); todos = todos.filter(t => t.id !== todoId); renderTodos(); } catch (error) { /* Handled */ } } }

if (logThisWorkoutBtn) { /* ... (same, ensure workoutTimerMessage is used) ... */ logThisWorkoutBtn.addEventListener('click', async () => {
    if (!currentUserData) { showUserMessage(workoutTimerMessage, "Please login to log workouts.", true); navigateTo('login'); return; } const workoutNameToLog = currentTimerSessionDetails ? currentTimerSessionDetails.name : ""; const elapsedSeconds = currentTimerSessionDetails ? currentTimerSessionDetails.durationSeconds : 0; if (!workoutNameToLog || elapsedSeconds <= 5) { showUserMessage(workoutTimerMessage, "Workout too short or no name. Not logged.", true); return; } const durationMinutes = Math.max(1, Math.round(elapsedSeconds / 60)); let estimatedCaloriesBurned = durationMinutes * 7;
    // Include pose feedback and rep count (if available) in the log
    const feedbackText = (typeof poseFeedback !== 'undefined' && poseFeedback && poseFeedback.textContent) ? poseFeedback.textContent : '';
    const feedbackSummary = feedbackText + (repCount ? ` Reps: ${repCount}` : '');
    const poseData = {
        exercise: exerciseSelectForPose ? exerciseSelectForPose.value : workoutNameToLog,
        feedback: feedbackText,
        reps: repCount || 0,
        start_timestamp: currentTimerSessionDetails && currentTimerSessionDetails.startTime ? new Date(currentTimerSessionDetails.startTime).toISOString() : null,
        duration_seconds: currentTimerSessionDetails ? currentTimerSessionDetails.durationSeconds : elapsedSeconds
    };
    const logData = { exercise_name: workoutNameToLog, duration_minutes: durationMinutes, calories_burned: estimatedCaloriesBurned, feedback: feedbackSummary, pose_data: poseData };
    try { logThisWorkoutBtn.textContent = "Logging..."; logThisWorkoutBtn.disabled = true; const response = await apiCall('/workout_logs', 'POST', logData); showUserMessage(workoutTimerMessage, response.message || "Workout logged!", false); resetTimer(0); if (workoutTimerDiv) workoutTimerDiv.style.display = 'none'; currentTimerSessionDetails = { name: null, startTime: null, durationSeconds: 0 }; currentWorkoutNameForTimer = ""; if (window.location.hash.includes('logs')) fetchAndDisplayWorkoutLogs_v2(); } catch (error) { showUserMessage(workoutTimerMessage, `Log failed: ${error.message}`, true, false); } finally { if (logThisWorkoutBtn) { logThisWorkoutBtn.textContent = "Log This Workout"; logThisWorkoutBtn.disabled = false; logThisWorkoutBtn.style.display = 'none'; } }
});
}


// Updated logs renderer that displays structured pose_data when available
async function fetchAndDisplayWorkoutLogs_v2() {
    if (!logsList || !currentUserData) {
        if (logsList) logsList.innerHTML = '<li>Please login to view workout logs.</li>';
        return;
    }
    logsList.innerHTML = '<li><p>Loading your workout logs...</p></li>';
    try {
        const fetchedLogs = await apiCall('/workout_logs');
        logsList.innerHTML = '';
        if (!fetchedLogs || fetchedLogs.length === 0) {
            logsList.innerHTML = '<li>You have no workout logs recorded yet.</li>';
            return;
        }

        fetchedLogs.forEach(log => {
            const li = document.createElement('li');
            const logDate = log.log_date ? new Date(log.log_date + 'T00:00:00Z') : new Date();
            const formattedDate = logDate.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });

            let html = `<strong>${log.exercise_name || 'Workout'}</strong> - ${formattedDate}`;
            html += `<br> <small>Duration: ${log.duration_minutes !== null && log.duration_minutes !== undefined ? log.duration_minutes : '-'} min | Calories: ${log.calories_burned !== null && log.calories_burned !== undefined ? Math.round(log.calories_burned) : '-'} kcal`;

            if (log.pose_data) {
                try {
                    const pd = (typeof log.pose_data === 'string') ? JSON.parse(log.pose_data) : log.pose_data;
                    if (pd) {
                        html += `<br><em>Feedback: ${pd.feedback || ''}</em>`;
                        html += `<br><small>Exercise: ${pd.exercise || ''} | Reps: ${pd.reps || 0} | Start: ${pd.start_timestamp ? (new Date(pd.start_timestamp)).toLocaleString() : '-'} | Duration: ${pd.duration_seconds ? pd.duration_seconds + 's' : '-'}</small>`;
                    } else if (log.feedback) {
                        html += `<br><em>Feedback: ${log.feedback}</em>`;
                    }
                } catch (e) {
                    console.warn('Could not parse pose_data for log:', e);
                    if (log.feedback) html += `<br><em>Feedback: ${log.feedback}</em>`;
                }
            } else if (log.feedback) {
                html += `<br><em>Feedback: ${log.feedback}</em>`;
            }

            html += '</small>';
            li.innerHTML = html;
            logsList.appendChild(li);
        });

    } catch (error) {
        console.error('Failed to fetch workout logs:', error);
        if (logsList) logsList.innerHTML = '<li><p class="message error">Error loading workout logs. Please try again.</p></li>';
    }
}

// Calorie Cycle - displays target vs actual calories burned for past 7 days
async function fetchAndDisplayCalorieCycle() {
    const calorieCycleList = document.getElementById('calorieCycleList');
    const calorieCycleDisplay = document.getElementById('calorieCycleDisplay');

    if (!calorieCycleList || !currentUserData) {
        return;
    }

    calorieCycleList.innerHTML = '<li>Loading calorie data...</li>';

    try {
        const data = await apiCall('/calorie_cycle');
        calorieCycleList.innerHTML = '';

        if (!data || !data.days || data.days.length === 0) {
            calorieCycleList.innerHTML = '<li>No calorie data available.</li>';
            return;
        }

        // Update header with target info
        const headerP = calorieCycleDisplay.querySelector('p');
        if (headerP) {
            headerP.innerHTML = `Target: <strong>${data.target_daily_calories || 0}</strong> kcal/day (TDEE: ${data.tdee || 0} kcal) | Goal: ${data.goal ? data.goal.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : 'N/A'}`;
        }

        // Display each day
        data.days.forEach((day) => {
            const li = document.createElement('li');
            li.className = 'calorie-day';

            const burnedPercent = day.target_calories > 0 ? Math.min(100, (day.calories_burned / day.target_calories) * 100) : 0;
            const hasWorkout = day.calories_burned > 0;

            li.innerHTML = `
                    <div class="calorie-day-header">
                        <span class="day-name">${day.day_name}</span>
                        <span class="day-date">${new Date(day.date + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                    </div>
                    <div class="calorie-stats">
                        <span class="target">Target: ${day.target_calories} kcal</span>
                        <span class="burned ${hasWorkout ? 'has-workout' : ''}">Burned: ${day.calories_burned} kcal</span>
                        ${day.workout_count > 0 ? `<span class="workout-count">(${day.workout_count} workout${day.workout_count > 1 ? 's' : ''})</span>` : ''}
                    </div>
                    <div class="calorie-bar">
                        <div class="calorie-bar-fill ${hasWorkout ? 'active' : ''}" style="width: ${burnedPercent}%"></div>
                    </div>
                `;

            calorieCycleList.appendChild(li);
        });

    } catch (error) {
        console.error('Failed to fetch calorie cycle:', error);
        calorieCycleList.innerHTML = '<li class="error">Error loading calorie data.</li>';
    }
}

// --- Admin Dashboard Functions ---
let adminUsersData = [];

async function fetchAdminStats() {
    try {
        const stats = await apiCall('/admin/stats');
        document.getElementById('statTotalUsers').textContent = stats.total_users || 0;
        document.getElementById('statAdminCount').textContent = stats.admin_count || 0;
        document.getElementById('statTotalWorkouts').textContent = stats.total_workouts || 0;
        document.getElementById('statRecentWorkouts').textContent = stats.recent_workouts_7d || 0;
        document.getElementById('statTotalTodos').textContent = stats.total_todos || 0;
        document.getElementById('statDietLogs').textContent = stats.total_diet_logs || 0;
    } catch (error) {
        console.error('Failed to fetch admin stats:', error);
        showAdminMessage('Failed to load statistics', true);
    }
}

async function fetchAdminUsers() {
    try {
        const data = await apiCall('/admin/users');
        adminUsersData = data.users || [];
        renderAdminUserTable(adminUsersData);
    } catch (error) {
        console.error('Failed to fetch admin users:', error);
        const tbody = document.getElementById('adminUserTableBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="7" class="error">Failed to load users</td></tr>';
    }
}

function renderAdminUserTable(users) {
    const tbody = document.getElementById('adminUserTableBody');
    if (!tbody) return;

    if (!users || users.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7">No users found</td></tr>';
        return;
    }

    const currentUserId = currentUserData ? currentUserData.id : null;
    const currentUserIsSuperadmin = currentUserData ? currentUserData.is_superadmin : false;

    tbody.innerHTML = users.map(user => {
        const isSelf = currentUserId && user.id === currentUserId;
        const goalDisplay = user.goals ? user.goals.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : '-';
        const isSuperadmin = user.is_superadmin;
        const canModify = !isSelf && (!isSuperadmin || currentUserIsSuperadmin);

        return `
                <tr class="${user.is_admin ? 'admin-row' : ''} ${isSuperadmin ? 'superadmin-row' : ''}">
                    <td>${user.id}</td>
                    <td>${user.username}${isSelf ? ' <span class="you-badge">(You)</span>' : ''}${isSuperadmin ? ' <span class="superadmin-badge">★ Super</span>' : ''}</td>
                    <td>${user.email}</td>
                    <td>${goalDisplay}</td>
                    <td>${user.workout_count}</td>
                    <td>
                        <label class="admin-toggle">
                            <input type="checkbox" ${user.is_admin ? 'checked' : ''} 
                                   onchange="toggleUserAdmin(${user.id}, this.checked)"
                                   ${!canModify ? 'disabled' : ''}>
                            <span class="toggle-slider"></span>
                        </label>
                    </td>
                    <td>
                        <button class="btn-danger btn-small" onclick="deleteUser(${user.id}, '${user.username}')" 
                                ${!canModify ? 'disabled' : ''}>Delete</button>
                    </td>
                </tr>
            `;
    }).join('');
}

window.toggleUserAdmin = async function (userId, isAdmin) {
    try {
        const result = await apiCall(`/admin/users/${userId}`, 'PUT', { is_admin: isAdmin });
        showAdminMessage(result.message || 'User updated', false);
        fetchAdminStats();
        fetchAdminUsers();
    } catch (error) {
        console.error('Failed to toggle admin status:', error);
        showAdminMessage(error.message || 'Failed to update user', true);
        fetchAdminUsers(); // Refresh to reset toggle state
    }
};

window.deleteUser = async function (userId, username) {
    if (!confirm(`Are you sure you want to delete user "${username}"?\n\nThis will also delete all their workout logs, diet logs, and todos. This action cannot be undone.`)) {
        return;
    }

    try {
        const result = await apiCall(`/admin/users/${userId}`, 'DELETE');
        showAdminMessage(result.message || 'User deleted', false);
        fetchAdminStats();
        fetchAdminUsers();
    } catch (error) {
        console.error('Failed to delete user:', error);
        showAdminMessage(error.message || 'Failed to delete user', true);
    }
};

function showAdminMessage(message, isError) {
    const msgEl = document.getElementById('adminMessage');
    if (msgEl) {
        msgEl.textContent = message;
        msgEl.className = `message ${isError ? 'error' : 'success'}`;
        msgEl.style.display = 'block';
        setTimeout(() => { msgEl.style.display = 'none'; }, 4000);
    }
}

function initAdminDashboard() {
    fetchAdminStats();
    fetchAdminUsers();

    // Search functionality
    const searchInput = document.getElementById('adminUserSearch');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = adminUsersData.filter(user =>
                user.username.toLowerCase().includes(query) ||
                user.email.toLowerCase().includes(query) ||
                (user.goals && user.goals.toLowerCase().includes(query))
            );
            renderAdminUserTable(filtered);
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('adminRefreshUsers');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            fetchAdminStats();
            fetchAdminUsers();
        });
    }
}

async function initializeApp() {
    if (currentYearSpan) currentYearSpan.textContent = new Date().getFullYear();
    const storedUserString = localStorage.getItem('Fit-GuideUser');
    let initialRoute = defaultInitialView;
    if (storedUserString) {
        try {
            const storedUserData = JSON.parse(storedUserString);
            // Set a temporary currentUserData to allow /user_profile call
            currentUserData = storedUserData;

            const profileData = await apiCall('/user_profile');

            // Combine stored admin status with fresh profile data from backend
            // The backend's /user_profile now returns is_admin flag.
            currentUserData = {
                ...profileData, // This contains the full fresh profile from backend
                id: storedUserData.id, // Ensure ID from storage is kept if not in profileData
                username: profileData.username || storedUserData.username, // Prefer backend username
                is_admin: profileData.is_admin !== undefined ? profileData.is_admin : (storedUserData.is_admin || false)
            };
            localStorage.setItem('Fit-GuideUser', JSON.stringify(currentUserData));

            if (currentUserData.is_admin) {
                console.log("Admin session restored:", currentUserData.username);
                if (adminPanelCard) adminPanelCard.style.display = 'block';
            } else {
                if (adminPanelCard) adminPanelCard.style.display = 'none';
            }
            initializeDashboardNavItems();

            // Show UI immediately, then fetch data in background
            initialRoute = window.location.hash.substring(1) || defaultDashboardRoute;
            router();
            fetchDashboardData();

        } catch (error) {
            console.warn("Session validation/initial data fetch error:", error.message);
            currentUserData = null; localStorage.removeItem('Fit-GuideUser');
            currentRawProfileData = null;
            if (adminPanelCard) adminPanelCard.style.display = 'none';
            initialRoute = defaultInitialView;
            if (!window.location.hash.includes('login') && !window.location.hash.includes('register')) {
                navigateTo('login');
            }
        }
    } else {
        if (adminPanelCard) adminPanelCard.style.display = 'none';
    }

    // Final navigation decision for non-stored users
    if (!storedUserString) {
        if (!window.location.hash || window.location.hash === "#" || window.location.hash.includes('dashboard')) {
            navigateTo(initialRoute);
        } else {
            router();
        }
    }
}
window.addEventListener('hashchange', router);
initializeApp();

});
