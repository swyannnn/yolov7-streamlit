import streamlit_authenticator as stauth

import database as db

usernames = ["yann", "teh"]
emails = ["swyannnn@gmail.com", "bubblesteh@hotmail.com"]
passwords = ["abc123", "def456"]
# hashed_passwords = stauth.Hasher(passwords).generate()


for (username, email, password) in zip(usernames, emails, passwords):
    db.insert_user(username, email, password)