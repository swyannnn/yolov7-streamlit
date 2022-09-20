import streamlit as st
import os
from deta import Deta  # pip install deta
from dotenv import load_dotenv  # pip install python-dotenv
import bcrypt

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

import streamlit as st
import streamlit.components.v1 as components

# embed streamlit docs in a streamlit app
components.iframe("https://docs.streamlit.io/en/latest")