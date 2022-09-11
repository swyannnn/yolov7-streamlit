import streamlit as st
import folium
import pandas as pd
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_folium import st_folium
import haversine as hs

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# before getting location access permission from user, plot all centre locations on map
def centredata():
    centres = pd.read_csv('centredata2.csv')

    map = folium.Map(location=[centres.Latitude.mean(), centres.Longitude.mean()], zoom_start=7, tiles='OpenStreetMap')

    # loop all centres in csv file and plot locations on map
    for _, centre in centres.iterrows():
        folium.Marker(
            location=[centre['Latitude'], centre['Longitude']],
            popup=centre['CompanyName'],
            tooltip=centre['CompanyName'],
            icon=folium.Icon(color='darkgreen', icon_color='white',prefix='fa', icon='circle')
        ).add_to(map)
    
    st_folium(map)
    
# create a button to access user's location
def permissionbutton():
    loc_button = Button(label="Click to allow location access")
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

# plot user's location on map
def plotuserlocation(latitude,longitude):
    map = folium.Map(location=(latitude,longitude),zoom_start = 12)

    # user's location as centre on map
    folium.Marker([latitude,longitude], popup = f"Your location:{latitude},{longitude}",
    tooltip=f"Your location:{latitude},{longitude}").add_to(map)

    # user's location above is wrong, for me it's this...
    folium.Marker([2.9920513,101.7830867], popup="My home", tooltip="My home",color='red').add_to(map)
    return map

# find and plot nearest centre from user
def findnearestcentre(map,latitude,longitude):
    # read csv file
    centre_loc=pd.read_csv('centredata2.csv')

    # zip data for each column
    centre_loc['coor'] = list(zip(centre_loc.Latitude, centre_loc.Longitude))

    # function to obtain distance between user's and centre's locations
    def distance_from(loc1,loc2): 
        distance=hs.haversine(loc1,loc2)
        return round(distance,1)

    # make a list to record the distances
    distance = list()
    for _,row in centre_loc.iterrows():
        distance.append(distance_from(row.coor,(latitude,longitude)))

    # assigning data in list to each columns
    centre_loc['distance']=distance

    centre_loc = centre_loc.sort_values(by=['distance'])
    return centre_loc

# plotting the 5 nearest centre from user's location on map
def nearestcentre(centre_loc):
    # read csv file
    centre_loc=pd.read_csv('centredata2.csv')

    # zip data for each column
    centre_loc['coor'] = list(zip(centre_loc.Latitude, centre_loc.Longitude))

    # function to obtain distance between user's and centre's locations
    def distance_from(loc1,loc2): 
        distance=hs.haversine(loc1,loc2)
        return round(distance,1)

    # make a list to record the distances
    distance = list()
    for _,row in centre_loc.iterrows():
        distance.append(distance_from(row.coor,(latitude,longitude)))

    # assigning data in list to each columns
    centre_loc['distance']=distance

    centre_loc = centre_loc.sort_values(by=['distance'])

    # plotting the 5 nearest centre from user's location on map
    x = 1
    for index, row in centre_loc.iterrows(): 
        if x <= 5:
            folium.Marker(
                location= [row['Latitude'],row['Longitude']],
                radius=5,
                popup= f"{row['CompanyName']}({row['distance']}km)",
                tooltip=f"{row['CompanyName']}({row['distance']}km)",
                color='red',
                fill=True,
                fill_color='red',
                icon=folium.Icon(color='darkgreen', icon_color='white',prefix='fa', icon='circle')
                ).add_to(map)
            x+=1
    return centre_loc




# listing the 5 nearest centre from user's location on map
def listnearestcentre(centre_loc):
    x = 1
    for index, row in centre_loc.iterrows(): 
        if x <= 5:
            st.write(f"""
            {x}) {row["CompanyName"]} -- {row['distance']} km\n
            📍 {row["Address"]}\n
            :telephone_receiver: {row['TelNum']}\n
            """)
            x+=1
    link = 'Click [HERE](https://ewaste.doe.gov.my/index.php/about/list-of-collectors/) to know all the government proved recycling centre in Malaysia'
    st.markdown(link,unsafe_allow_html=True)

# Final algorithm
result = permissionbutton()
if result:
    latitude,longitude = getuserlocation(result)
    map = plotuserlocation(latitude,longitude)
    centre_loc = findnearestcentre(map,latitude,longitude)
    st.subheader('Top 5 nearest recycle centre from your current location')
    plotnearestcentre(centre_loc)
    st_folium(map)
    listnearestcentre(centre_loc)
else:
    st.subheader('All government proved recycling centre in Malaysia')
    centredata()
