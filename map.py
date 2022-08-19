import streamlit as st
from streamlit_folium import st_folium
import folium

import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

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

if result:
    if "GET_LOCATION" in result:
        latitude = result.get("GET_LOCATION")['lat']
        longitude = result.get("GET_LOCATION")['lon']

# big map
map = folium.Map(location=(latitude, longitude), zooom_start = 30)

folium.Marker([latitude, longitude], popup="Your location").add_to(map)

st_folium(map)