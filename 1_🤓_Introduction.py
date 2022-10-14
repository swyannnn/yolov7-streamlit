import os
import streamlit as st
from deta import Deta
import streamlit.components.v1 as components

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# Load the environment variables
deta = Deta(st.secrets["deta_key"])
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

def user_status():
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
user_status()


ewaste = st.container()
ewaste.subheader("What is E-waste?")
ewaste.write("“E-waste” is a broken, non-working or old/obsolete electric electronic appliance such as TV, PC, air conditioner, washing machine and refrigerator.")
ewaste.write("It's been estimated that Malaysia produces more than 365,000 tonnes of e-waste every single year — That's heavier than the weight of the Petronas Twin Towers! Based on research, estimation shows Malaysia generates 24.5 million units of E-waste in 2025. (That's a lot!🤯)")

ewaste.subheader("What does an E-waste normally contain?")
ewaste.write("Component in E-waste contain *toxic* and *hazardous* material such as:")
ewaste.write("mercury, lead, cadmium, arsenic, bromine, beryllium will permeate into the earth and subsequently water sources as well as threaten the aquatic and human life if E-waste is not disposed in environmentally sound manner.")
ewaste.write("Component in E-waste also contain *precious metals* such as gold, copper, palladium and silver which has high recycling value.")

footprint = st.container()
footprint.subheader("Carbon Footprint")
footprint.write("A carbon footprint is the total amount of greenhouse gases that are caused by the choices and actions of an individual, company or a nation. Carbon footprint is measured in terms of carbon dioxide emissions (CO2).")
footprint.write("Carbon Footprint per person in Malaysia are equivalent to 8.68 tons per person.Globally, the average carbon footprint is closer to 4 tons. To have the best chance of avoiding a 2°C rise in global temperatures, the average global carbon footprint per year needs to drop to under 2 tons by 2050.")

why_recycle = st.container()
why_recycle.subheader("Why we should do recycling?")
why_recycle.write("Recycling helps reduce greenhouse gas emissions by reducing energy consumption. Using recycled materials to make new products reduces the need for virgin materials. This avoids greenhouse gas emissions that would result from extracting or mining virgin materials. In addition, manufacturing products from recycled materials typically requires less energy than making products from virgin materials.")

ewaste_manner = st.container()
ewaste_manner.subheader("Why we should properly managed e-waste in environmentally sound manner?")
ewaste_manner.write("E-waste is becoming a global issue. The more electrical and electronic equipment are being produced, the more E-waste need to be disposed or managed properly.")
ewaste_manner.write("If e-waste is discarded without implementing environmentally sound manner such as into the river, landfill, burning or sent to informal sector, e-waste may endanger our life, affecting human health and causing deterioration of environmental quality.")

components.html("""
    <object data=”https://allgreenrecycling.com/CarbonFootprintCalc.swf” type=”application/x-shockwave-flash” width=”398″ height=”407″><param name=”movie” value=”https://allgreenrecycling.com/CarbonFootprintCalc.swf” /></object>
     """)