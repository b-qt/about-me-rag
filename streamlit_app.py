"""Entry point: `streamlit run streamlit_app.py` (also the default main file on Streamlit Community Cloud)."""
import streamlit as st

from app.common import setting

st.set_page_config(page_title=f"About {setting('SUBJECT_NAME')}", page_icon="🧭", layout="wide")

pages = [
    st.Page("app/chat.py", title="Ask", icon="💬", default=True),
    st.Page("app/growth.py", title="Growth", icon="📈"),
    st.Page("app/mind_map.py", title="Mind Map", icon="🧭"),
]
st.navigation(pages).run()
