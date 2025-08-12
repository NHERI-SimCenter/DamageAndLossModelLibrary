"""
Simple authentication with common fixes applied.
"""

import streamlit as st
import requests
import secrets
from urllib.parse import urlencode, quote
import hashlib
import base64


def init_auth():
    """Initialize authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.auth_initialized = True


def login_page():
    """Display login page."""
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Get Auth0 config
    auth_config = st.secrets["auth0"]
    
    # Check for callback
    query_params = st.query_params
    if 'code' in query_params:
        # Important: Get state to verify it matches
        state_param = query_params.get('state', [''])[0] if isinstance(query_params.get('state'), list) else query_params.get('state', '')
        
        # Only process if state matches (CSRF protection)
        if state_param and state_param == st.session_state.get('auth_state'):
            code = query_params['code'][0] if isinstance(query_params['code'], list) else query_params['code']
            if _handle_callback(code, auth_config):
                st.success("✅ Successfully authenticated!")
                st.balloons()
                st.rerun()
        else:
            # State doesn't match, regenerate auth URLs
            st.session_state.pop('auth_urls_generated', None)
            st.warning("Session expired. Please sign in again.")
    
    # Generate auth URLs if needed
    if 'auth_urls_generated' not in st.session_state:
        _generate_auth_urls(auth_config)
    
    # Show login UI
    st.markdown("# 🔐 Welcome")
    st.markdown("Please sign in to continue")
    
    col1, col2, _ = st.columns([1, 1, 2])
    
    with col1:
        st.link_button(
            "🔑 Sign In",
            st.session_state.signin_url,
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.link_button(
            "📝 Sign Up",
            st.session_state.signup_url,
            use_container_width=True
        )
    
    return False


def logout():
    """Sign out the current user."""
    # Clear all auth-related session state
    keys_to_clear = [
        'authenticated', 'user', 'auth_urls_generated',
        'code_verifier', 'auth_state', 'signin_url', 'signup_url'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    
    # Redirect to Auth0 logout
    auth_config = st.secrets["auth0"]
    logout_url = f"https://{auth_config['domain']}/v2/logout?" + urlencode({
        'returnTo': auth_config.get('logout_url', 'https://your-app.streamlit.app'),
        'client_id': auth_config['client_id']
    })
    
    st.link_button("Click to complete logout", logout_url)


def get_user():
    """Get current user info."""
    return st.session_state.get('user', None)


def show_user_info():
    """Display user info in sidebar."""
    if st.session_state.get('authenticated', False):
        user = st.session_state.get('user', {})
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**👤 {user.get('name', 'User')}**")
        st.sidebar.caption(user.get('email', ''))
        if st.sidebar.button("Sign Out", use_container_width=True):
            logout()
        st.sidebar.markdown("---")


def require_auth():
    """Decorator to require authentication."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                st.error("🔒 Please sign in to access this feature")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --- Helper functions ---

def _normalize_url(url):
    """Normalize URL to ensure consistency."""
    # Remove trailing slash and ensure https
    url = url.rstrip('/')
    if url.startswith('http://localhost'):
        return url  # Keep http for localhost
    elif not url.startswith('https://'):
        url = 'https://' + url.replace('http://', '')
    return url


def _generate_auth_urls(config):
    """Generate Auth0 URLs with PKCE parameters."""
    # Generate cryptographically secure state
    state = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    st.session_state.auth_state = state
    
    # Generate PKCE verifier and challenge
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(43)).decode('utf-8').rstrip('=')
    st.session_state.code_verifier = verifier
    
    # Create challenge
    challenge_bytes = hashlib.sha256(verifier.encode('ascii')).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
    
    # Normalize callback URL
    callback_url = _normalize_url(config['callback_url'])
    
    # Store for debugging
    st.session_state.auth_callback_url = callback_url
    
    # Base parameters
    base_params = {
        'response_type': 'code',
        'client_id': config['client_id'],
        'redirect_uri': callback_url,
        'scope': 'openid profile email',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    }
    
    # Generate URLs
    auth_base = f"https://{config['domain']}/authorize"
    
    # Sign in URL
    st.session_state.signin_url = f"{auth_base}?{urlencode(base_params)}"
    
    # Sign up URL
    signup_params = base_params.copy()
    signup_params['screen_hint'] = 'signup'
    st.session_state.signup_url = f"{auth_base}?{urlencode(signup_params)}"
    
    st.session_state.auth_urls_generated = True


def _handle_callback(code, config):
    """Handle Auth0 callback."""
    # Use the same callback URL that was used in authorization
    callback_url = st.session_state.get('auth_callback_url')
    
    # Fallback if not in session
    if not callback_url:
        callback_url = _normalize_url(config['callback_url'])
    
    # Get code verifier from session
    code_verifier = st.session_state.get('code_verifier', '')
    
    if not code_verifier:
        st.error("Session expired. Please try signing in again.")
        # Clear auth state to force regeneration
        st.session_state.pop('auth_urls_generated', None)
        return False
    
    # Prepare token exchange
    token_url = f"https://{config['domain']}/oauth/token"
    token_data = {
        'grant_type': 'authorization_code',
        'client_id': config['client_id'],
        'code': code,
        'redirect_uri': callback_url,
        'code_verifier': code_verifier,
    }
    
    try:
        # Exchange code for token
        response = requests.post(
            token_url,
            json=token_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Token exchange failed: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('error_description', error_data.get('error', 'Unknown error'))}"
            except:
                error_msg += f" - {response.text}"
            st.error(error_msg)
            return False
        
        tokens = response.json()
        
        # Get user info
        userinfo_url = f"https://{config['domain']}/userinfo"
        user_response = requests.get(
            userinfo_url,
            headers={'Authorization': f"Bearer {tokens['access_token']}"}
        )
        
        if user_response.status_code != 200:
            st.error(f"Failed to get user info: {user_response.status_code}")
            return False
        
        # Success! Store in session
        st.session_state.authenticated = True
        st.session_state.user = user_response.json()
        
        # Clear auth parameters
        st.session_state.pop('auth_urls_generated', None)
        st.session_state.pop('code_verifier', None)
        st.session_state.pop('auth_state', None)
        st.session_state.pop('auth_callback_url', None)
        
        # Clear URL parameters
        st.query_params.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        # Clear auth state to allow retry
        st.session_state.pop('auth_urls_generated', None)
        return False