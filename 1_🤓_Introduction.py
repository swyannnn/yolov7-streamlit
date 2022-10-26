import os
import streamlit as st
from deta import Deta
import streamlit.components.v1 as components

from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
from PIL import Image

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# # Load the environment variables
# deta = Deta(st.secrets["deta_key"])
# # This is how to create/connect a database
# db = deta.Base("users_db")

# class Deta():
#     def insert_user(username):
#         """Returns the user on a successful user creation, otherwise raises and error"""
#         return db.put({"key": username, "point" : 0})
#     def fetch_all_users():
#         """Returns a dict of all users"""
#         res = db.fetch()
#         return res.items
#     def get_user(username):
#         """If not found, the function will return None"""
#         return db.get(username)
#     def update_user(username, updates):
#         """If the item is updated, returns None. Otherwise, an exception is raised"""
#         return db.update(updates, username)
#     def delete_user(username):
#         """Always returns None, even if the key does not exist"""
#         return db.delete(username)

# users = Deta.fetch_all_users()
# usernames = [user["key"] for user in users]

# class Authenticator():
#     def initialize():
#         if 'key' not in st.session_state:
#             st.session_state['key'] = None
#         if 'logout' not in st.session_state:
#             st.session_state['logout'] = None
#         if 'authentication_status' not in st.session_state:
#             st.session_state['authentication_status'] = None
#         return st.session_state['key'],st.session_state['logout'],st.session_state['authentication_status']
#     def register():
#         register_user_form = st.sidebar.form('Register user')
#         register_user_form.subheader("Register")
#         new_username = register_user_form.text_input('Username').lower()
#         if register_user_form.form_submit_button('Register'):
#             if len(new_username) > 0:
#                 if new_username not in usernames:
#                         Deta.insert_user(new_username)
#                         st.session_state['key'] = new_username
#                         st.session_state['authentication_status'] = True
#                         st.sidebar.success('New account registered successfully')
#                 else:
#                     st.sidebar.warning('Username already taken')
#             else:
#                 st.warning('Please enter an email, username and password')
#         return st.session_state['key'], st.session_state['authentication_status']
#     def login():
#         login_user_form = st.sidebar.form('Login')
#         login_user_form.subheader('Login')
#         username = login_user_form.text_input('Username').lower()
#         if login_user_form.form_submit_button('Login'):
#             if username in usernames:
#                 info = Deta.get_user(username)
#                 st.session_state['authentication_status'] = True
#                 st.session_state['key'] = info['key']
#             else:
#                 st.session_state['authentication_status'] = False
#                 st.sidebar.warning('Please enter your username')
#         return st.session_state['key'], st.session_state['authentication_status']
#     def logout():
#         st.session_state['username'] = None
#         st.session_state['authentication_status'] = None

# def user_status():
    # st.session_state['key'],st.session_state['logout'],st.session_state['authentication_status'] = Authenticator.initialize()

    # def callback_to_login_button():
    #     st.session_state['authentication_status']=False
    # def callback_to_logout_button():
    #     st.session_state['authentication_status']=True
    # user_status = st.empty()

    # if not st.session_state['authentication_status']:
    #     user_status.write("You are not logged in")
    #     pages_name = ['Yes','No']
    #     page = st.sidebar.radio('Already have an account?',pages_name, horizontal=True)
    #     if page == 'Yes':
    #         st.session_state['key'], st.session_state['authentication_status'] = Authenticator.login()
    #         if st.session_state['authentication_status']:
    #             user_status.write(f"You are now logged in as {st.session_state['key']}")
    #     if page == 'No':
    #         st.session_state['key'], st.session_state['authentication_status'] = Authenticator.register()
    #         if st.session_state['authentication_status']:
    #             user_status.write(f"You are now logged in as {st.session_state['key']}")

    # else:
    #     user_status.write(f"You are now logged in as {st.session_state['key']}")
    #     logout = st.sidebar.button('Log Out', on_click = callback_to_login_button)
    #     if logout:
    #         Authenticator.logout()
# user_status()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# lottie_recycle = load_lottiefile("./lottiefiles/54940-recycle-icon-animation.json")
# col1,col2=st.columns((2,8))
# with col1:
#     st_lottie(lottie_recycle, height=100, key="recycle")
# with col2:
#     ewaste = st.container()
#     ewaste.subheader("What is E-waste?")
#     ewaste.write('When a electric electronic appliance is old, broken or non-working, we called it an "E-waste".')

# FAQ1------------------------------------------------------------------------------------------------------
def display_lottie_electronics():
    lottie_phone1 = load_lottiefile("./lottiefiles/97008-phone-3d.json")
    lottie_laptop1 = load_lottiefile("./lottiefiles/23662-laptop-animation-pink-navy-blue-white.json")
    lottie_microwaveoven1 = load_lottiefile("./lottiefiles/3139-microwave-oven.json")
    lottie_tv1 = load_lottiefile("./lottiefiles/49117-tv-bad-weather.json")
    lottie_washingmachine1 = load_lottiefile("./lottiefiles/3138-washing-machine.json")
    lottie_ac1 = load_lottiefile("./lottiefiles/75551-air-conditioner.json")
    col1, col2, col3=st.columns((2,2,2))
    with col1:
        st_lottie(lottie_phone1, height=100, key="phone1")
        st_lottie(lottie_tv1, height=100, key="tv1")
    with col2:
        st_lottie(lottie_laptop1, height=100, key="laptop1")
        st_lottie(lottie_washingmachine1, height=100, key="washingmachine1")
    with col3:
        st_lottie(lottie_microwaveoven1, height=100, key="microwaveoven1")
        st_lottie(lottie_ac1, height=100, key="ac1")
def display_ewaste_barchart():
    HtmlFile = open("ewastebar.html", "r")
    source_code = HtmlFile.read() 
    components.html(source_code, height = 600,width=900)

st.subheader("What is E-waste?")
st.write('When a electric electronic appliance is old, broken or non-working, we called it an "E-waste".')
st.write('These are the examples of electronic items:')
display_lottie_electronics()
st.write("Play around with the slider to see the amount of worldwide E-waste genarated over the years!")
display_ewaste_barchart()
st.write("In Malaysia, we produces more than 365,000 tonnes of e-waste every single year — That's heavier than the weight of the Petronas Twin Towers! Based on research, estimation shows Malaysia generates 24.5 million units of E-waste in 2025. (That's a lot!🤯)")
# FAQ1------------------------------------------------------------------------------------------------------

# FAQ2------------------------------------------------------------------------------------------------------
def display_precious_components_image():
    gold, silver, copper, palladium=st.columns((2,2,2,2))
    with gold:
        image = Image.open('images/precious/gold.png')
        st.image(image, caption='Gold', width=100)
    with silver:
        image = Image.open('images/precious/silver.png')
        st.image(image, caption='Silver', width=100)
    with copper:
        image = Image.open('images/precious/copper.png')
        st.image(image, caption='Copper', width=100)
    with palladium:
        image = Image.open('images/precious/palladium.png')
        st.image(image, caption='Palladium', width=100)
def display_toxic_components_image():
    arsenic, beryllium, bromine, cadmium, lead, mercury=st.columns((2,2,2,2,2,2))
    with arsenic:
        image = Image.open('images/toxic/arsenic.png')
        st.image(image, caption='Arsenic', width=100)
    with beryllium:
        image = Image.open('images/toxic/beryllium.png')
        st.image(image, caption='Beryllium', width=100)
    with bromine:
        image = Image.open('images/toxic/bromine.png')
        st.image(image, caption='Bromine', width=100)
    with cadmium:
        image = Image.open('images/toxic/cadmium.png')
        st.image(image, caption='Cadmium', width=100)
    with lead:
        image = Image.open('images/toxic/lead.png')
        st.image(image, caption='Lead', width=100)
    with mercury:
        image = Image.open('images/toxic/mercury.png')
        st.image(image, caption='Mercury', width=100)

st.subheader("What does an E-waste normally contain?")
st.write("Component in E-waste contains *precious* and *valuable* materials such as:")
display_precious_components_image()
st.write("However, E-waste contains *toxic* and *hazardous* materials such as:")
display_toxic_components_image()
# FAQ2------------------------------------------------------------------------------------------------------

# FAQ3------------------------------------------------------------------------------------------------------
lottie_recycle_icon = load_lottiefile("./lottiefiles/54940-recycle-icon-animation.json")
lottie_recycle_text = load_lottiefile("./lottiefiles/115879-recycle-text-animation.json")
st.subheader("How can we reduce the usage of the *toxic* and *hazardous* materials?")
col1,col2=st.columns((3,7))
with col1:
    st_lottie(lottie_recycle_text, height=120, key="recycle_text")
    st_lottie(lottie_recycle_icon, height=150, key="recycle_icon")
with col2:
    st.write("Recycling helps reduce greenhouse gas emissions by reducing energy consumption.")
    st.write("Using recycled materials to make new products reduces the need for virgin materials.")
    st.write("This avoids greenhouse gas emissions that would result from extracting or mining virgin materials.")     
    components.html(
"""
<!DOCTYPE html>
<html>
<head>
<style>
div {
  width: 420px;
  padding: 5px;
  border: 3px solid gray;
  margin: 0;
}
</style>
</head>
<body>
<h2>Do You Know That</h2>
<div>Manufacturing products from recycled materials typically requires <b>less energy</b> than making products from virgin materials,<br>which means lesser <b>CARBON FOOTPRINT</b>!!🥳</div>

</body>
</html>""")
# FAQ3------------------------------------------------------------------------------------------------------

# FAQ4------------------------------------------------------------------------------------------------------
lottie_say_no_CO2 = load_lottiefile("./lottiefiles/33545-carbon-dioxide-emission.json")
st.subheader("Carbon Footprint")
col1,col2=st.columns((2,8))
with col1:
    st_lottie(lottie_say_no_CO2, height=100, key="say_no_CO2")
    image = Image.open('images/others/Earth-Footprints.png')
    st.image(image, width=90)
with col2:
    st.write("Carbon footprint is the total amount of greenhouse gases that are caused by the choices and actions of an individual, company or a nation.")
    st.write("Carbon footprint is measured in terms of carbon dioxide emissions (CO2).")
    st.write("Carbon Footprint per person in Malaysia are equivalent to 8.68 tons per person. Globally, the average carbon footprint is closer to 4 tons.")
    st.write("To have the best chance of avoiding a 2°C rise in global temperatures, the average global carbon footprint per year needs to drop to under 2 tons by 2050.")
# FAQ4------------------------------------------------------------------------------------------------------

# FAQ5------------------------------------------------------------------------------------------------------
lottie_questionman = load_lottiefile("./lottiefiles/32045-question.json")
col1,col2=st.columns((8,2))
with col1:
    st.subheader("We should recycle the E-waste instead of throwing it into rubbish bin. WHY??")
    st.write("E-waste is becoming a global issue🌏")
    st.write("The more electrical and electronic equipment are being produced, the more E-waste need to be disposed or managed properly.")
    st.write("If e-waste is discarded without implementing environmentally sound manner such as into the river, landfill, burning or sent to informal sector, e-waste may endanger our life, affecting human health and causing deterioration of environmental quality.")
    st.write("Therefore, we should properly managed e-waste in environmentally sound manner!")
with col2:
    st_lottie(lottie_questionman, height=200, key="questionman")
# FAQ5------------------------------------------------------------------------------------------------------
