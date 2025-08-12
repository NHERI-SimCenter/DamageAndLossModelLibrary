"""
Simple authentication for Streamlit using Auth0.
Just works. No complexity.
"""

import streamlit as st
import requests
import secrets
from urllib.parse import urlencode
import hashlib
import base64


def init_auth():
    """Initialize authentication. Call this once at the start of your app."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None


def login_page():
    """
    Display login page. Returns True if authenticated, False otherwise.
    
    Usage:
        if not login_page():
            st.stop()
    """
    # Check if already authenticated
    if st.session_state.authenticated:
        return True
    
    # Get Auth0 config from Streamlit secrets
    auth_config = st.secrets["auth0"]
    
    # Check for callback (user returning from Auth0)
    query_params = st.query_params
    if 'code' in query_params:
        if _handle_callback(query_params['code'], auth_config):
            st.rerun()
    
    # Show login UI
    st.markdown("# 🔐 Welcome")
    st.markdown("Please sign in to continue")
    
    col1, col2, _ = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Sign In", type="primary", use_container_width=True):
            _redirect_to_auth0(auth_config)
    
    with col2:
        if st.button("Sign Up", use_container_width=True):
            _redirect_to_auth0(auth_config, signup=True)
    
    return False


def require_auth():
    """
    Simple decorator to require authentication.
    
    Usage:
        @require_auth()
        def my_protected_function():
            return "Secret data"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                st.error("🔒 Please sign in to access this feature")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def logout():
    st.session_state.authenticated = False
    st.session_state.user = None

    cfg = st.secrets["auth0"]
    logout_url = "https://{}/v2/logout?{}".format(
        cfg['domain'],
        urlencode({'returnTo': cfg['logout_url'], 'client_id': cfg['client_id']})
    )

    st.markdown(
        f"""
        <script>
          if (window.top) {{
            window.top.location.href = "{logout_url}";
          }} else {{
            window.location.href = "{logout_url}";
          }}
        </script>
        """,
        unsafe_allow_html=True,
    )


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


# --- Private helper functions ---

def _redirect_to_auth0(config, signup=False):
    import secrets, base64, hashlib
    state = secrets.token_urlsafe(32)
    st.session_state.auth_state = state

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('=')
    st.session_state.code_verifier = verifier

    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip('=')

    params = {
        'response_type': 'code',
        'client_id': config['client_id'],
        'redirect_uri': config['callback_url'],
        'scope': 'openid profile email',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    }
    if signup:
        params['screen_hint'] = 'signup'

    auth_url = f"https://{config['domain']}/authorize?{urlencode(params)}"

    # Force top-level navigation (works even if the app is iframed)
    st.markdown(
        f"""
        <script>
          if (window.top) {{
            window.top.location.href = "{auth_url}";
          }} else {{
            window.location.href = "{auth_url}";
          }}
        </script>
        """,
        unsafe_allow_html=True,
    )

    
    if signup:
        params['screen_hint'] = 'signup'
    
    auth_url = f"https://{config['domain']}/authorize?" + urlencode(params)
    
    # Redirect using meta tag
    st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', 
                unsafe_allow_html=True)


def _handle_callback(code, config):
    """Handle Auth0 callback."""
    # Exchange code for token
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
        
        # Clear URL parameters
        st.query_params.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False