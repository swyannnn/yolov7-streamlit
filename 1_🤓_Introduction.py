from turtle import onclick
from dotenv import load_dotenv
import os
import streamlit as st
from deta import Deta

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# Load the environment variables
# load_dotenv(".env")
# DETA_KEY = os.getenv("DETA_KEY")
DETA_KEY="c02438ym_H2yB9nr6ho7bCBabFs8D8ecLLqTnpy5C"
# Initialize with a project key
deta = Deta(DETA_KEY)
# This is how to create/connect a database
db = deta.Base("users_db")

class Deta():
    def insert_user(username, email):
        """Returns the user on a successful user creation, otherwise raises and error"""
        return db.put({"key": username, "email": email, "point" : 0})
    def fetch_all_users():
        """Returns a dict of all users"""
        res = db.fetch()
        return res.items
    def get_user(username):
        """If not found, the function will return None"""
        return db.get(username)
    def update_user(username, updates):
        """If the item is updated, returns None. Otherwise, an exception is raised"""
        return db.update(updates, username)
    def delete_user(username):
        """Always returns None, even if the key does not exist"""
        return db.delete(username)

users = Deta.fetch_all_users()
usernames = [user["key"] for user in users]

class Authenticator():
    def initialize():
        if 'key' not in st.session_state:
            st.session_state['key'] = None
        if 'logout' not in st.session_state:
            st.session_state['logout'] = None
        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = None
        return st.session_state['key'],st.session_state['logout'],st.session_state['authentication_status']
    def register():
        register_user_form = st.sidebar.form('Register user')
        register_user_form.subheader("Register")
        new_email = register_user_form.text_input('Email')
        new_username = register_user_form.text_input('Username').lower()
        if register_user_form.form_submit_button('Register'):
            if len(new_email) and len(new_username) > 0:
                if new_username not in usernames:
                        Deta.insert_user(new_username, new_email)
                        st.session_state['key'] = new_username
                        st.session_state['authentication_status'] = True
                        st.sidebar.success('New account registered successfully')
                else:
                    st.sidebar.warning('Username already taken')
            else:
                st.warning('Please enter an email, username and password')
        return st.session_state['key'], st.session_state['authentication_status']
    def login():
        login_user_form = st.sidebar.form('Login')
        login_user_form.subheader('Login')
        username = login_user_form.text_input('Username').lower()
        if login_user_form.form_submit_button('Login'):
            if username in usernames:
                info = Deta.get_user(username)
                st.session_state['authentication_status'] = True
                st.session_state['key'] = info['key']
            else:
                st.session_state['authentication_status'] = False
                st.sidebar.warning('Please enter your username')
        return st.session_state['key'], st.session_state['authentication_status']
    def logout():
        st.session_state['username'] = None
        st.session_state['authentication_status'] = None

st.session_state['key'],st.session_state['logout'],st.session_state['authentication_status'] = Authenticator.initialize()


def callback_to_login_button():
    st.session_state['authentication_status']=False

def callback_to_logout_button():
    st.session_state['authentication_status']=True

user_status = st.empty()

if not st.session_state['authentication_status']:
    user_status.write("You are not logged in")
    pages_name = ['Yes','No']
    page = st.sidebar.radio('Already have an account?',pages_name, horizontal=True)
    if page == 'Yes':
        st.session_state['key'], st.session_state['authentication_status'] = Authenticator.login()
        if st.session_state['authentication_status']:
            user_status.write(f"You are now logged in as {st.session_state['key']}")
    if page == 'No':
        st.session_state['key'], st.session_state['authentication_status'] = Authenticator.register()
        if st.session_state['authentication_status']:
            user_status.write(f"You are now logged in as {st.session_state['key']}")

else:
    user_status.write(f"You are now logged in as {st.session_state['key']}")
    logout = st.sidebar.button('Log Out', on_click = callback_to_login_button)
    if logout:
        Authenticator.logout()

