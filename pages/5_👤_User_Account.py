# import streamlit as st
# import plotly.figure_factory as ff
# import numpy as np

# st.set_page_config(
#     layout="wide",
#     page_title="E-waste",
#     page_icon="♻️",
# )

# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components
# from matplotlib import animation
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import seaborn as sns

# # create worldwide e-waste generation each year as anomated bar chart
# @st.cache(show_spinner=False)
# def ewastebar():
#     fig = plt.figure(figsize=(3,1))
#     axes = fig.add_subplot(1,1,1)
#     axes.set_ylim(0, 70)
#     plt.style.use("seaborn")

#     x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = [], [], [], [], [], [], [], [], [], [], [], []


#     lst1=[i if i<33.8 else 33.8 for i in np.arange(0,60.0,0.1)]
#     lst2=[i if i<35.8 else 35.8 for i in np.arange(0,60.0,0.1)]
#     lst3=[i if i<37.8 else 37.8 for i in np.arange(0,60.0,0.1)]
#     lst4=[i if i<39.8 else 39.8 for i in np.arange(0,60.0,0.1)]
#     lst5=[i if i<44.4 else 44.4 for i in np.arange(0,60.0,0.1)]
#     lst6=[i if i<46.4 else 46.4 for i in np.arange(0,60.0,0.1)]
#     lst7=[i if i<48.2 else 48.2 for i in np.arange(0,60.0,0.1)]
#     lst8=[i if i<50 else 50 for i in np.arange(0,60.0,0.1)]
#     lst9=[i if i<51.8 else 51.8 for i in np.arange(0,60.0,0.1)]
#     lst10=[i if i<53.6 else 53.6 for i in np.arange(0,60.0,0.1)]
#     lst11=[i if i<57.4 else 57.4 for i in np.arange(0,60.0,0.1)]
#     palette = list(reversed(sns.color_palette("afmhot", 11).as_hex()))

#     def animate(i):
#         y1=lst1[i]
#         y2=lst2[i]
#         y3=lst3[i]
#         y4=lst4[i]
#         y5=lst5[i]
#         y6=lst6[i]
#         y7=lst7[i]
#         y8=lst8[i]
#         y9=lst9[i]
#         y10=lst10[i]
#         y11=lst11[i]
        
#         plt.bar(range(11), sorted([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11]), color=palette)
#         tick_lst=["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#         plt.xticks(range(11), tick_lst)

#     plt.title("Some Title, Year: {} ".format(5000), color=("blue"))
#     ani = FuncAnimation(fig, animate, interval=10)

#     with open("ewastebar.html","w") as f:
#         print(ani.to_jshtml(), file=f)


# HtmlFile = open("ewastebar.html", "r")
# source_code = HtmlFile.read() 
# components.html(source_code, height = 900,width=900)






# from streamlit_lottie import st_lottie
# from streamlit_lottie import st_lottie_spinner
# import requests, json

# def load_lottiefile(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)


# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# lottie_recycle = load_lottiefile("./lottiefiles/54940-recycle-icon-animation.json")
# col1,col2=st.columns((2,8))
# with col1:
#     st_lottie(lottie_recycle, height=100, key="recycle")
# with col2:
#     st.write('Hi')


import json
import time

import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_progress = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_TsKMbf.json")
lottie_success = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_3ykigvxc.json")
lottie_error = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_LlRvIg.json")

st.set_page_config(page_title="Streamlit Lottie Demo", page_icon=":tada:", initial_sidebar_state='collapsed')

st.title("Hello Lottie!")
st.markdown(
    """
[Lottie](https://airbnb.io/lottie) is a library that parses [Adobe After Effects](http://www.adobe.com/products/aftereffects.html) animations 
exported as json with [Bodymovin](https://github.com/airbnb/lottie-web) and renders them natively on mobile and on the web!
Go look at the [awesome animations](https://lottiefiles.com/) to spice your Streamlit app!
"""
)

with st.sidebar:
    st.header("Animation parameters")
    speed = st.slider("Select speed", 0.1, 2.0, 1.0)
    reverse = st.checkbox("Reverse direction", False)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
        unsafe_allow_html=True
    )

c_col1, colx, c_col2, coly = st.columns((1, 0.1, 0.25, 1))
if c_col1.button("Run some heavy computation...for 5 seconds!"):
    with c_col2.empty():
        with st_lottie_spinner(lottie_progress, loop=True, key="progress"):
            time.sleep(5)
        st_lottie(lottie_success, loop=False, key="success")

st.markdown("---")
st.header("Try it yourself!")
st.markdown(
    "Choose a Lottie from [the website](https://lottiefiles.com/) and paste its 'Lottie Animation URL'"
)
lottie_url = st.text_input(
    "URL", value="https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
)
downloaded_url = load_lottieurl(lottie_url)

if downloaded_url is None:
    col1, col2 = st.columns((2, 1))
    col1.warning(f"URL {lottie_url} does not seem like a valid lottie JSON file")
    with col2:
        st_lottie(lottie_error, height=100, key="error")
else:
    with st.echo("above"):
        st_lottie(downloaded_url, key="user")






















fig = go.Figure(
    data=[go.Scatter(x=[0, 1], y=[0, 1])],
    layout=go.Layout(
        xaxis=dict(range=[2010, 2022], autorange=False),
        yaxis=dict(range=[0, 60], autorange=False),
        title="E-waste generated every year",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(data=[go.Scatter(x=[0,2010], y=[0, 33.8])]),
            go.Frame(data=[go.Scatter(x=[2010,2011], y=[33.8,35.8])]),
            go.Frame(data=[go.Scatter(x=[2011,2012], y=[35.8, 37.8])]),
            go.Frame(data=[go.Scatter(x=[2012,2013], y=[37.8, 39.8])]),
            go.Frame(data=[go.Scatter(x=[2013,2014], y=[39.8, 44.4])]),
            go.Frame(data=[go.Scatter(x=[2014,2015], y=[44.4, 46.4])]),
            go.Frame(data=[go.Scatter(x=[2015,2016], y=[46.4, 48.2])]),
            go.Frame(data=[go.Scatter(x=[2016,2017], y=[48.2, 50])]),
            go.Frame(data=[go.Scatter(x=[2017,2018], y=[50, 51.8])]),
            go.Frame(data=[go.Scatter(x=[2018,2019], y=[51.8, 53.6])]),
            go.Frame(data=[go.Scatter(x=[2019,2020], y=[53.6, 57.4])],
            layout=go.Layout(title_text="End Title"))]
)


# Plot!
st.plotly_chart(fig, use_container_width=True)