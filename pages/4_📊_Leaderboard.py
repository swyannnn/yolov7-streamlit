import streamlit as st

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

st.experimental_set_query_params()
{"show_map": ["True"], "selected": ["asia", "america"]}