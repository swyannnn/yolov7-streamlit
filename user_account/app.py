import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
from authenticate import Authenticate
import database as db

# def create_credential():
users = db.fetch_all_users()

usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
emails = [user["email"] for user in users]
passwords = [user["password"] for user in users]

credentials = {"usernames":{}}

for name, username, password in zip(names, usernames, passwords):
    user_dict = {"name":name, "password":password}
    credentials["usernames"].update({username:user_dict})
st.text(credentials)
authenticator = stauth.Authenticate(credentials,
    "detect-electronics", "barbie", cookie_expiry_days=30,preauthorized=True)

#     return authenticator,credentials

# authenticator,credentials = create_credential()

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')

elif authentication_status == False:
    st.text(authentication_status)
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username,password')

if authentication_status:
    try:
        if authenticator.reset_password(username, 'Reset password'):
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)

try:
    if authenticator.register_user('Register user'):
        st.success('User registered successfully')
except Exception as e:
    st.error(e)


