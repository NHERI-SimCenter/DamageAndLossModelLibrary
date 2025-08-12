import streamlit as st

def is_authenticated() -> bool:
    """True if the user is logged in via Streamlit auth."""
    # st.user is always available; use its is_logged_in flag per docs.
    return bool(getattr(st.user, "is_logged_in", False))

def current_user() -> dict:
    """
    Return a dict with common fields from st.user (name, email, sub, picture).
    Keys may vary by IdP claims—handle missing attrs gracefully.
    """
    u = st.user
    return {
        "name": getattr(u, "name", None),
        "email": getattr(u, "email", None),
        "sub": getattr(u, "id", None),        # Streamlit surfaces the subject identifier
        "picture": getattr(u, "picture", None),
        "raw": u,                              # full object for advanced use
    }

def ensure_login(provider: str = "auth0") -> None:
    """
    If user isn’t logged in, render a 'Log in' button that calls st.login(provider).
    Call this near the top of your page. If the user is logged in, returns immediately.
    """
    if not is_authenticated():
        if st.button("Log in", type="primary", use_container_width=True):
            st.login(provider)  # redirects to Auth0
        # Stop further execution until user logs in
        st.stop()

def logout_button(label: str = "Log out") -> None:
    """
    Renders a logout button if the user is logged in.
    """
    if is_authenticated():
        if st.button(label, use_container_width=True):
            st.logout()

def require_auth(provider: str = "auth0"):
    """
    Decorator to protect functions (simple guard).
    Usage:
        @require_auth()
        def my_view():
            st.write("secret")
    """
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            if not is_authenticated():
                # Show a minimal gate and stop.
                st.info("🔒 Please log in to continue.")
                if st.button("Log in", type="primary"):
                    st.login(provider)
                st.stop()
            return fn(*args, **kwargs)
        return _wrapped
    return _decorator
