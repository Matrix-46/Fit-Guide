// Global configuration for Fit-Guide Frontend
const CONFIG = {
    API_BASE_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname.startsWith('192.168.'))
        ? 'http://localhost:5000/api'
        : 'https://fit-guide.onrender.com/api'
};
