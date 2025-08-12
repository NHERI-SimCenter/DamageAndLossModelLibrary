import streamlit as st
from .simple_auth import is_authenticated, current_user, logout_button

def render_login_panel(
    title: str = "🔐 Welcome",
    subtitle: str = "Sign in to continue",
    provider: str = "auth0",
) -> None:
    """
    Minimal, clean login UI.
    - If logged out: shows a single 'Log in' button.
    - If logged in: shows a compact user chip + 'Log out'.
    """
    st.markdown(f"## {title}")
    if not is_authenticated():
        st.caption(subtitle)
        col = st.container()
        if col.button("Log in", type="primary"):
            st.login(provider)
        st.divider()
        st.caption("You’ll be redirected to Auth0.")
    else:
        u = current_user()
        with st.container(border=True):
            left, right = st.columns([1, 1])
            with left:
                avatar = u.get("picture")
                if avatar:
                    st.image(avatar, width=48)
                st.markdown(f"**{u.get('name') or 'User'}**")
                if u.get("email"):
                    st.caption(u["email"])
            with right:
                logout_button("Log out")
