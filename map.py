import streamlit as st
import folium
import pandas as pd
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_folium import st_folium

# create a button to access user's location
def permissionbutton():
    loc_button = Button(label="Yes")
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
    return result


# getting user's location when he/she allows 
def getuserlocation(result):
    latitude = result.get("GET_LOCATION")['lat']
    longitude = result.get("GET_LOCATION")['lon']
    return latitude,longitude

# plot location on map
def drawmap(latitude,longitude):
    map = folium.Map(location=(latitude,longitude), tiles='Stamen Toner',zoom_start = 300)

    # user's location as centre on map
    folium.Marker([latitude,longitude], popup = f"Your location:{latitude},{longitude}").add_to(map)

    # user's location above is wrong, for me it's this...
    folium.Marker([2.9920513,101.7830867], popup="My home",color='red').add_to(map)
    return map

#read recycling centre data 
def centredata(map):
    centres = pd.read_csv('centredata2.csv')

    # loop all centres in csv file and plot locations on map
    for _, centre in centres.iterrows():
        folium.Marker(
            location=[centre['Latitude'], centre['Longitude']],
            popup=centre['CompanyName'],
            tooltip=centre['CompanyName'],
            icon=folium.Icon(color='darkgreen', icon_color='white',prefix='fa', icon='circle')
        ).add_to(map)

def map(bbox_count):
    if bbox_count > 0:
        st.text(f'{bbox_count} device(s) founded, would you like to recycle?')
        result = permissionbutton()
        if result:
            latitude,longitude = getuserlocation(result)
            map = drawmap(latitude,longitude)
            centredata(map)
            st_folium(map)
