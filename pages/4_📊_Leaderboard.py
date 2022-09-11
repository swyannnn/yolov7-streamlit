import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

from deta import Deta
DETA_KEY="c02438ym_H2yB9nr6ho7bCBabFs8D8ecLLqTnpy5C"
# Initialize with a project key
deta = Deta(DETA_KEY)
db = deta.Base("users_db")

class Deta():
    def fetch_all_users():
        """Returns a dict of all users"""
        res = db.fetch()
        return res.items
    

users = Deta.fetch_all_users()

keys = [user["key"] for user in users]
points = [user["point"] for user in users]
df = pd.DataFrame({'Username': keys, 'Point': points})
st.table(df)