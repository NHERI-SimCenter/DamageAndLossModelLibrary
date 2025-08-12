"""
Simple authentication using Auth0 with link buttons.
Alternative approach for Streamlit Cloud.
"""

import streamlit as st
import requests
import secrets
from urllib.parse import urlencode
import hashlib
import base64


def init_auth():
    """Initialize authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None


def login_page():
    """Display login page with link buttons."""
    # Check if already authenticated
    if st.session_state.authenticated:
        return True
    
    # Get Auth0 config
    auth_config = st.secrets["auth0"]
    
    # Check for callback
    query_params = st.query_params
    if 'code' in query_params:
        if _handle_callback(query_params['code'], auth_config):
            st.rerun()
    
    # Generate and store PKCE parameters
    if 'auth_urls_generated' not in st.session_state:
        _generate_auth_urls(auth_config)
    
    # Show login UI with link buttons
    st.markdown("# 🔐 Welcome")
    st.markdown("Please sign in to continue")
    
    col1, col2, _ = st.columns([1, 1, 2])
    
    with col1:
        # Use link_button for sign in
        st.link_button(
            "🔑 Sign In",
            st.session_state.signin_url,
            type="primary",
            use_container_width=True
        )
    
    with col2:
        # Use link_button for sign up
        st.link_button(
            "📝 Sign Up",
            st.session_state.signup_url,
            use_container_width=True
        )
    
    st.info("👆 Click a button above to authenticate with Auth0")
    
    return False


def logout():
    """Sign out the current user."""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.pop('auth_urls_generated', None)
    
    # Create logout URL
    auth_config = st.secrets["auth0"]
    logout_url = f"https://{auth_config['domain']}/v2/logout?" + urlencode({
        'returnTo': auth_config['logout_url'],
        'client_id': auth_config['client_id']
    })
    
    # Show logout link
    st.link_button("Click here to complete logout", logout_url)


def get_user():
    """Get current user info."""
    return st.session_state.get('user', None)


def show_user_info():
    """Display user info in sidebar."""
    if st.session_state.authenticated:
        user = st.session_state.user
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

def _generate_auth_urls(config):
    """Generate Auth0 URLs with PKCE parameters."""
    # Generate state
    state = secrets.token_urlsafe(32)
    st.session_state.auth_state = state
    
    # Generate PKCE verifier and challenge
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    st.session_state.code_verifier = verifier
    
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode('utf-8').rstrip('=')
    
    # Base parameters
    base_params = {
        'response_type': 'code',
        'client_id': config['client_id'],
        'redirect_uri': config['callback_url'],
        'scope': 'openid profile email',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    }
    
    # Sign in URL
    signin_params = base_params.copy()
    st.session_state.signin_url = f"https://{config['domain']}/authorize?" + urlencode(signin_params)
    
    # Sign up URL
    signup_params = base_params.copy()
    signup_params['screen_hint'] = 'signup'
    st.session_state.signup_url = f"https://{config['domain']}/authorize?" + urlencode(signup_params)
    
    st.session_state.auth_urls_generated = True


def _handle_callback(code, config):
    """Handle Auth0 callback."""
    token_url = f"https://{config['domain']}/oauth/token"
    token_data = {
        'grant_type': 'authorization_code',
        'client_id': config['client_id'],
        'code': code,
        'redirect_uri': config['callback_url'],
        'code_verifier': st.session_state.get('code_verifier', ''),
    }
    
    try:
        # Get token
        response = requests.post(token_url, json=token_data)
        response.raise_for_status()
        tokens = response.json()
        
        # Get user info
        userinfo_url = f"https://{config['domain']}/userinfo"
        user_response = requests.get(
            userinfo_url,
            headers={'Authorization': f"Bearer {tokens['access_token']}"}
        )
        user_response.raise_for_status()
        
        # Store in session
        st.session_state.authenticated = True
        st.session_state.user = user_response.json()
        st.session_state.pop('auth_urls_generated', None)
        
        # Clear URL parameters
        st.query_params.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False