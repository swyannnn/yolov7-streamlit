import streamlit as st
import os
from email_validator import validate_email
import streamlit_authenticator as stauth
from deta import Deta  # pip install deta
from dotenv import load_dotenv  # pip install python-dotenv

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

def insert_user(username, email, password):
    """Returns the user on a successful user creation, otherwise raises and error"""
    return db.put({"key": username, "email": email, "password": stauth.Hasher(password).generate()})

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


users = fetch_all_users()
usernames = [user["key"] for user in users]


register_user_form = st.form('Register user')
register_user_form.subheader("Register")
new_email = register_user_form.text_input('Email')
new_username = register_user_form.text_input('Username').lower()
new_password = register_user_form.text_input('Password', type='password')
new_password_repeat = register_user_form.text_input('Repeat password', type='password')
if register_user_form.form_submit_button('Register'):
    if len(new_email) and len(new_username) and len(new_password) > 0:
        if new_username not in usernames:
            if new_password == new_password_repeat:
                # if email = validate_email(new_email).email:
                    st.text(validate_email(new_email, dns_resolver=resolver).email)
                    insert_user(new_username, new_email, new_password)
                    st.text(f'Welcome {new_username}')
                # else:
                #     st.warning('Invalid email')
            else:
                st.warning('Passwords do not match')
        else:
            st.warning('Username already taken')
    else:
        st.warning('Please enter an email, username and password')
    