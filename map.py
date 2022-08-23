import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

# # getting user's location when he/she allows 
# def getuserlocation():
#     latitude = result.get("GET_LOCATION")['lat']
#     longitude = result.get("GET_LOCATION")['lon']
#     return latitude,longitude

def permissionbutton():
    loc_button = Button(label="Get Location")
    loc_button.js_on_event("button_click", CustomJS(code="""
        navigator.geolocation.getCurrentPosition(
            (loc) => {
                document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
            }
        )
        """))

    result = streamlit_bokeh_events(
        loc_button,
        events="GET_LOCATION",
        key="get_location",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)
# # result = permissionbutton()
# # if result:
# #     latitude,longitude = getuserlocation()

# # draw a map
# # Stamen Terrain, Stamen Toner, Stamen Water Color, CartoDB Positron
# map = folium.Map(location=(latitude,longitude), tiles='OpenStreetMap',zoom_start = 300)

# # user's location as centre on map
# folium.Marker([latitude,longitude], popup=f"Your location:{latitude},{longitude}").add_to(map)

# # user's location above is wrong, for me it's this...
# folium.Marker([2.9920513,101.7830867], popup="My home",color='red').add_to(map)

# #read recycling centre data 
# centres = pd.read_csv('centredata2.csv')

# # loop all centres in csv file and plot locations on map
# for _, centre in centres.iterrows():
#     folium.Marker(
#         location=[centre['Latitude'], centre['Longitude']],
#         popup=centre['CompanyName'],
#         tooltip=centre['CompanyName'],
#         icon=folium.Icon(color='darkgreen', icon_color='white',prefix='fa', icon='circle')
#     ).add_to(map)

# st_folium(map)

