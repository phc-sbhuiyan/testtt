import streamlit as st
from logger import AppLogger
import random
from constant import *

def get_logger():
    """
    This function returns the logger.
    If the logger is not in the session state, it creates a new logger
    and adds it to the session state before returning it.
    """
    if "logger" not in st.session_state:
        st.session_state.logger = AppLogger().logger
    return st.session_state.logger

def generate_random_id():
    return str(random.randint(1000, 9999))

def create_sidebar():
    with st.sidebar:
        st.sidebar.markdown(f"## {HACKATHON_TITLE}", unsafe_allow_html=True)
