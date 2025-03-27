import requests
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import matplotlib.pyplot as plt
import folium 
from streamlit_folium import st_folium
import geopandas as gpd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from folium.plugins import MarkerCluster, MeasureControl, MiniMap
import branca
import matplotlib.colors as mcolors


############################################################
############################################################

# Streamlit instellingen
pd.set_option('display.max_columns', None)
st.set_page_config(layout='wide')

# mardowns toevoegen
st.markdown(
    """
    <style>
        /* Witte achtergrond voor de hele pagina */
        body {
            background-color: #FFFFFF;  
        }
        .stApp {
            background-color: #FFFFFF;  /* Witte achtergrond voor de app */
        }

        /* Titels in KLM blauw */
        h1, h2, h3, h4 {
            color: #00A1E4 !important;  /* KLM-blauw */
        }

        /* Transparante achtergrond voor de widgets */
        .stSelectbox, .stPlotlyChart, .stButton {
            background-color: rgba(255, 255, 255, 0) !important; /* Transparant voor widgets */
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Functie om afbeeldingen correct te laden
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# KLM-logo rechts onderin plaatsen
KLM_logo = get_image_base64("KLM_logo.png")
if KLM_logo:
    st.markdown(
        f"""
        <div style="position: fixed; bottom: 10px; right: 10px;">
            <img src="data:image/png;base64,{KLM_logo}" style="width: 120px;" />
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.write("Fout bij het laden van de afbeelding.")

# KLM kleuren schema voor de plots:
klm_colors = ['#1C65A3', '#00A9E0', '#A7A8AA', '#E0E5E8', '#FF6A13']

######################################################### Dataset inlezen #########################################################################3

gemiddelde_sel_periode = pd.read_csv("sensornet_data.csv")
arrivals_vs_departure = pd.read_csv("arrival_vs_departure.csv")
capa_filtered = pd.read_csv("capa_data_filtered.csv")
geodata_geluid = pd.read_csv("geodata_geluid.csv")
qatarA350 = pd.read_csv('QR276_39967286.csv')
TK1952 = pd.read_csv('TK1952_39965e7d.csv')

#inladen Aalsmeer data

# Cache de functie die de data ophaalt en verwerkt
@st.cache_data
def load_aalsmeer_data(start_date, end_date):
    # Maak de request naar de API
    response = requests.get(f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_date}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_date}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags')
    
    # Verkrijg de kolomnamen en data
    colnames = pd.DataFrame(response.json()['metadata'])
    aalsmeer_df = pd.DataFrame(response.json()['rows'])
    aalsmeer_df.columns = colnames.headers
    
    # Zet de tijd om naar een datetime formaat
    aalsmeer_df['time'] = pd.to_datetime(aalsmeer_df['time'], unit='s')
    
    # Voeg een kolom toe die aangeeft dat de data uit Aalsmeer komt
    aalsmeer_df['bron'] = 'Aalsmeer'
    
    return aalsmeer_df

# Gebruik de functie en geef het de juiste datums
start_date = int(pd.to_datetime('2025-03-23').timestamp())
end_date = int(pd.to_datetime('2025-03-26').timestamp())
aalsmeer_df = load_aalsmeer_data(start_date, end_date)

geluids_df = aalsmeer_df.copy()

# Data voor Aalsmeer
aalsmeer_data = {
    'location_long': [
        'Uitersweg', 'Aalsmeerderweg', 'Hornweg', 
        'Blaauwstraat', 'Kudelstraatseweg', 
        'Darwinstraat', 'Copierstraat'
    ],
    'Latitude': [52.264, 52.272, 52.268, 52.264, 52.235, 52.234, 52.229],
    'Longitude': [4.733, 4.771, 4.780, 4.775, 4.741, 4.759, 4.739],
    'Plaats': ['Aalsmeer'] * 7
}

# Data voor Amstelveen
amstelveen_data = {
    'location_long': [
        'Catharina van Clevepark', 'Jeanne d\'Arclaan', 'Burgemeester Haspelslaan',
        'Noorddammerweg', 'Sportlaan', 'Schweitzerlaan', 
        'Langs de Werf', 'Pastoor van Zantenlaan', 'De Wijde Blik'
    ],
    'Latitude': [52.320, 52.315, 52.306, 52.293, 52.290, 52.285, 52.282, 52.261, 52.259],
    'Longitude': [4.862, 4.880, 4.868, 4.834, 4.852, 4.822, 4.836, 4.872, 4.869],
    'Plaats': ['Amstelveen'] * 9
}

# Maak DataFrames
df_aalsmeer = pd.DataFrame(aalsmeer_data)
df_amstelveen = pd.DataFrame(amstelveen_data)

geodata_geluid_yuri = pd.concat([df_aalsmeer, df_amstelveen], ignore_index=True)

######################################################## Tabs aanmaken #######################################################################################
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Sensor data", "A330 VS. A350","Conclusions"])

# Homepagina met KLM-verhaal
with tab1:
    st.title("✈️ Soundcheck A350: Vliegt Schiphol de Stilte Tegemoet?")
    st.write("KLM gaat de Airbus A330 vervangen door de stillere A350. Dit dashboard heeft als doel om vlucht- en sensordata te analyseren en te vergelijken of de A350 daadwerkelijk stiller is dan de A330, en of de inwoners van Aalsmeer en Amstelveen minder last van het vliegtuiggeluid zullen hebben.")
    st.write("")
    st.write("**Het dashboard is opgedeeld in de volgende secties:**")
    st.write("")
    st.header("**A330 VS. A350: Vliegtuigen Vergelijken**")
    st.write("Hier kun je de specificaties en geluidskenmerken van de A330 en A350 vergelijken, en zie je de belangrijkste verschillen tussen de twee vliegtuigen.")
    st.header("**Sensor Data: Geluid Monitoring** ")
    st.write("In dit gedeelte wordt een kaart weergegeven met de locaties van de geluidsensoren rondom de luchthaven. Je kunt hier ook de geluidsniveaus zien die door verschillende vliegtuigen tijdens hun vlucht worden geproduceerd.")
    st.header("**Vluchtpad Heatmap**")
    st.write("Een heatmap toont de vluchtpaden van inkomende vliegtuigen naar Schiphol, zodat je inzicht krijgt in hoe vliegtuigen de luchthaven naderen en welk geluid ze onderweg genereren.")
    st.header("**Conclusie**")
    st.write("Op basis van de verzamelde data zal dit gedeelte een overzicht geven van de bevindingen en een definitieve conclusie of de A350 inderdaad stiller is dan de A330.")


################################ TAB 2 BOUWEN #########################################
with tab2:


    # Combineer beide DataFrames
    gefilterdQTR276 = geluids_df[geluids_df['callsign']=='QTR276']
    gefilterdQTR276 = gefilterdQTR276.merge(geodata_geluid_yuri, on='location_long')
    aalsmeerderweg = gefilterdQTR276[gefilterdQTR276['location_long']=='Aalsmeerderweg']
    blauwstraat = gefilterdQTR276[gefilterdQTR276['location_long']=='Blaauwstraat']
    hornweg = gefilterdQTR276[gefilterdQTR276['location_long']=='Hornweg']

    # Haal mu op als een enkel getal
    mu = aalsmeerderweg.loc[0, 'duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(aalsmeerderweg.loc[0, 'time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(aalsmeerderweg.loc[0, 'time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = aalsmeerderweg.loc[0, 'lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})
    geluid_aalsmeerderweg = pd.merge_asof(df,aalsmeerderweg,on='time',direction='nearest')

    # Haal mu op als een enkel getal
    mu = blauwstraat.iloc[0]['duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(blauwstraat.iloc[0]['time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(blauwstraat.iloc[0]['time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = blauwstraat.iloc[0]['lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})

    geluid_blauwstraat = pd.merge_asof(df,blauwstraat,on='time',direction='nearest')

    # Haal mu op als een enkel getal
    mu = hornweg.iloc[0]['duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(hornweg.iloc[0]['time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(hornweg.iloc[0]['time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = hornweg.iloc[0]['lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})

    geluid_hornweg = pd.merge_asof(df,hornweg,on='time',direction='nearest')

    qatarA350 = pd.read_csv('QR276_39967286.csv')
    qatarA350["time"] = pd.to_datetime(qatarA350["UTC"]).dt.tz_localize(None)
    # 'Position' kolom splitsen in 'Latitude' en 'Longitude'
    qatarA350[["Latitude", "Longitude"]] = qatarA350["Position"].str.split(",", expand=True)

    # Omzetten naar numerieke waarden
    qatarA350["Latitude"] = qatarA350["Latitude"].astype(float)
    qatarA350["Longitude"] = qatarA350["Longitude"].astype(float)

    # 'Position' kolom verwijderen
    qatarA350.drop(columns=["Position"], inplace=True)

    qatarA350 = qatarA350.head(130)
    qatarA350 = qatarA350.tail(61).reset_index(drop=True)

    # Specificeer de start- en eindtijd
    start_time = pd.to_datetime("2025-03-23 12:01:27")
    end_time = pd.to_datetime("2025-03-23 12:03:30")

    # Maak een lijst van datetimes met een interval van 10 seconden
    time_list = pd.DataFrame(pd.date_range(start=start_time, end=end_time, freq="1s"))
    time_list.columns = ['time']

    vluchtenmerge = pd.merge_ordered(time_list,qatarA350,how='left', on='time')
    vluchtenmerge.interpolate(method="linear", inplace=True)

    df = vluchtenmerge.copy()

    times = sorted(set(df["time"].unique().tolist()))

    # Create the figure
    figa350 = go.Figure()



    # Add geodata locations as a separate trace
    figa350.add_trace(go.Scattermap(
        lat=geodata_geluid_yuri['Latitude'],
        lon=geodata_geluid_yuri['Longitude'],
        mode='markers+text',
        marker=dict(size=9, color='blue', opacity=0.7),
        text=geodata_geluid_yuri['location_long'],
        name="Geodata Locations"
    ))


    # Create initial traces (for the first time step)
    figa350.add_trace(go.Scattermap(
        lat=df[df["time"] == times[0]]["Latitude"],
        lon=df[df["time"] == times[0]]["Longitude"],
        mode='markers',
        # marker=dict(size=9, color='red', opacity=0.7),
        text=df[df["time"] == times[0]]["Callsign"]
    ))


    # Set up the mapbox layout
    figa350.update_layout(
        map=dict(
            style="open-street-map",  # You can replace this with a Mapbox style if you have a token
            center={"lat": 52.2790, "lon": 4.7809},  # Center the map on your data (around Schiphol)
            zoom=12  # Adjust the zoom level as needed
        ),
        hovermode="closest",
        height=800,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.0,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Prepare frames for each time step
    frames = []
    for time in times:
        frame = {
            "data": [
                go.Scattermap(
                    lat=df[df["time"] == time]["Latitude"],
                    lon=df[df["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=11, color='darkred', opacity=1),
                    name="Qatar 276",
                ),
                go.Scattermap(
                    lat=geluid_aalsmeerderweg[geluid_aalsmeerderweg["time"] == time]["Latitude"],
                    lon=geluid_aalsmeerderweg[geluid_aalsmeerderweg["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=(geluid_aalsmeerderweg[geluid_aalsmeerderweg["time"] == time]['decibel']**6)/1500000000, 
                                color=geluid_aalsmeerderweg[geluid_aalsmeerderweg["time"] == time]['decibel'], 
                                opacity=0.3),

                    name="Aalsmeerderweg",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_aalsmeerderweg[geluid_aalsmeerderweg["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),      
                go.Scattermap(
                    lat=geluid_blauwstraat[geluid_blauwstraat["time"] == time]["Latitude"],
                    lon=geluid_blauwstraat[geluid_blauwstraat["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=(geluid_blauwstraat[geluid_blauwstraat["time"] == time]['decibel']**6)/1500000000, 
                                color=geluid_blauwstraat[geluid_blauwstraat["time"] == time]['decibel'], 
                                opacity=0.3),

                    name="Blauwstraat",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_blauwstraat[geluid_blauwstraat["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),   
                go.Scattermap(
                    lat=geodata_geluid['Latitude'],
                    lon=geodata_geluid['Longitude'],
                    mode='markers+text',
                    marker=dict(size=9, color='blue', opacity=0.5),
                    text=geodata_geluid['location_long'],
                    name="Geodata Locations"
                ),
                go.Scattermap(
                    lat=geluid_hornweg[geluid_hornweg["time"] == time]["Latitude"],
                    lon=geluid_hornweg[geluid_hornweg["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=(geluid_hornweg[geluid_hornweg["time"] == time]['decibel']**6)/1500000000, 
                                color=geluid_hornweg[geluid_hornweg["time"] == time]['decibel'], 
                                opacity=0.3),

                    name="Hornweg",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_hornweg[geluid_hornweg["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),    

                                
            ],
            "name": str(time)  # Use the time as the frame name
        }
        frames.append(frame)

    figa350.frames = frames

    # Add lines connecting the points for tui_vlucht (this will animate along with the points)
    figa350.add_trace(go.Scattermap(
        lat=df["Latitude"],
        lon=df["Longitude"],
        mode='lines',  # Only draw the line
        line=dict(color='red', width=1),
        name="TUI Vliegtuig Route",
        opacity=0.7
    ))

    # Add geodata locations as a separate trace
    figa350.add_trace(go.Scattermap(
        lat=geodata_geluid_yuri['Latitude'],
        lon=geodata_geluid_yuri['Longitude'],
        mode='markers+text',
        marker=dict(size=9, color='blue', opacity=0.5),
        text=geodata_geluid_yuri['location_long'],
        name="Geodata Locations"
    ))

    figa350.add_trace(go.Scattermap(
        lat=geluid_hornweg[geluid_hornweg["time"] == times[0]]["Latitude"],
        lon=geluid_hornweg[geluid_hornweg["time"] == times[0]]["Longitude"],
        # mode='markers',
        # marker=dict(size=geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel'], 
        #             color=geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel'], 
        #             opacity=0.3),

        name="Hornweg",))



    # Add the slider
    sliders_dict = {
        "steps": [
            {
                "args": [
                    [str(time)],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }
                ],
                "label": str(time),
                "method": "animate"
            }
            for time in times
        ],
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 15},
            "prefix": "Time: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "bounce-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
    }

    figa350.update_layout(sliders=[sliders_dict],showlegend = False)

    # Show the map
    # fig.show()


    #############
    # Tweede kaart
    #############


    gefilterdKL423 = geluids_df.loc[geluids_df['callsign'] == 'THY5VU']
    gefilterdKL423 = gefilterdKL423.merge(geodata_geluid_yuri, on='location_long')
    aalsmeerderweg2 = gefilterdKL423[gefilterdKL423['location_long']=='Aalsmeerderweg']
    blauwstraat2 = gefilterdKL423[gefilterdKL423['location_long']=='Blaauwstraat']
    hornweg2 = gefilterdKL423[gefilterdKL423['location_long']=='Hornweg']

    # Haal mu op als een enkel getal
    mu = aalsmeerderweg2.loc[0, 'duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(aalsmeerderweg2.loc[0, 'time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(aalsmeerderweg2.loc[0, 'time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = aalsmeerderweg2.loc[0, 'lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})

    geluid_aalsmeerderweg2 = pd.merge_asof(df,aalsmeerderweg2,on='time',direction='nearest')


    # Haal mu op als een enkel getal
    mu = blauwstraat2.iloc[0]['duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(blauwstraat2.iloc[0]['time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(blauwstraat2.iloc[0]['time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = blauwstraat2.iloc[0]['lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})

    geluid_blauwstraat2 = pd.merge_asof(df,blauwstraat2,on='time',direction='nearest')

    # Haal mu op als een enkel getal
    mu = hornweg2.iloc[0]['duration'] / 3  # Midden van de tijdsperiode
    mu = mu.astype('int')
    sigma = 10  # Spreiding van de normaalverdeling

    # Start- en eindtijd berekenen als enkele waardes
    start_time = pd.to_datetime(hornweg2.iloc[0]['time']) - pd.Timedelta(seconds=2*mu)
    end_time = pd.to_datetime(hornweg2.iloc[0]['time']) + pd.Timedelta(seconds=2*mu)

    # Tijdreeks genereren met 1-seconde interval
    time = pd.date_range(start=start_time, end=end_time, freq="1s")
    time_seconds = np.linspace(-mu, mu, len(time))  # Zet tijd om naar numerieke as rond mu

    # Normaalverdeling genereren
    peak_db = hornweg2.iloc[0]['lasmax_dB']  # Haal de piekwaarde op
    db_values = peak_db * np.exp(-(time_seconds**2) / (2 * sigma**2))

    # Dataframe maken
    df = pd.DataFrame({"time": time, "decibel": db_values})

    geluid_hornweg2 = pd.merge_asof(df,hornweg2,on='time',direction='nearest')

    TK1952 = pd.read_csv('TK1952_39965e7d.csv')
    TK1952["time"] = pd.to_datetime(TK1952["UTC"]).dt.tz_localize(None)
    # 'Position' kolom splitsen in 'Latitude' en 'Longitude'
    TK1952[["Latitude", "Longitude"]] = TK1952["Position"].str.split(",", expand=True)

    # Omzetten naar numerieke waarden
    TK1952["Latitude"] = TK1952["Latitude"].astype(float)
    TK1952["Longitude"] = TK1952["Longitude"].astype(float)

    # 'Position' kolom verwijderen
    TK1952.drop(columns=["Position"], inplace=True)

    TK1952 = TK1952.head(100)
    TK1952 = TK1952.tail(41).reset_index(drop=True)

    # Specificeer de start- en eindtijd
    start_time2 = pd.to_datetime(TK1952['time'].iloc[0])
    end_time2 = pd.to_datetime(TK1952['time'].iloc[-1])

    # Maak een lijst van datetimes met een interval van 10 seconden
    time_list2 = pd.DataFrame(pd.date_range(start=start_time2, end=end_time2, freq="1s"))
    time_list2.columns = ['time']

    TK1952merge = pd.merge_ordered(time_list2,TK1952,how='left', on='time')
    TK1952merge.interpolate(method="linear", inplace=True)

    df = TK1952merge.copy()

    times = sorted(set(df["time"].unique().tolist()))

    # Create the figure
    fig2 = go.Figure()

    # Add geodata locations as a separate trace
    fig2.add_trace(go.Scattermap(
        lat=geodata_geluid_yuri['Latitude'],
        lon=geodata_geluid_yuri['Longitude'],
        mode='markers+text',
        marker=dict(size=9, color='blue', opacity=0.7),
        text=geodata_geluid_yuri['location_long'],
        name="Geodata Locations"
    ))


    # Create initial traces (for the first time step)
    fig2.add_trace(go.Scattermap(
        lat=df[df["time"] == times[0]]["Latitude"],
        lon=df[df["time"] == times[0]]["Longitude"],
        mode='markers',
        # marker=dict(size=9, color='red', opacity=0.7),
        text=df[df["time"] == times[0]]["Callsign"]
    ))


    # Set up the mapbox layout
    fig2.update_layout(
        map=dict(
            style="open-street-map",  # You can replace this with a Mapbox style if you have a token
            center={"lat": 52.2790, "lon": 4.7809},  # Center the map on your data (around Schiphol)
            zoom=12  # Adjust the zoom level as needed
        ),
        hovermode="closest",
        height=800,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.0,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Prepare frames for each time step
    frames = []
    for time in times:
        frame = {
            "data": [
                go.Scattermap(
                    lat=df[df["time"] == time]["Latitude"],
                    lon=df[df["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=11, color='darkred', opacity=1),
                    name="TK 1952 (A 330)",
                ),
                go.Scattermap(
                    lat=geluid_aalsmeerderweg2[geluid_aalsmeerderweg2["time"] == time]["Latitude"],
                    lon=geluid_aalsmeerderweg2[geluid_aalsmeerderweg2["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=(geluid_aalsmeerderweg2[geluid_aalsmeerderweg2["time"] == time]['decibel']**6)/1500000000, 
                                color=geluid_aalsmeerderweg2[geluid_aalsmeerderweg2["time"] == time]['decibel'], 
                                opacity=0.3),

                    name="Aalsmeerderweg",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_aalsmeerderweg2[geluid_aalsmeerderweg2["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),      
                go.Scattermap(
                    lat=geluid_blauwstraat2[geluid_blauwstraat2["time"] == time]["Latitude"],
                    lon=geluid_blauwstraat2[geluid_blauwstraat2["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(size=(geluid_blauwstraat2[geluid_blauwstraat2["time"] == time]['decibel']**6)/1500000000, 
                                color=geluid_blauwstraat2[geluid_blauwstraat2["time"] == time]['decibel'], 
                                opacity=0.3),

                    name="Blauwstraat",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_blauwstraat2[geluid_blauwstraat2["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),   
                go.Scattermap(
                    lat=geodata_geluid['Latitude'],
                    lon=geodata_geluid['Longitude'],
                    mode='markers+text',
                    marker=dict(size=9, color='blue', opacity=0.5),
                    text=geodata_geluid['location_long'],
                    name="Geodata Locations"
                ),
                go.Scattermap(
                    lat=geluid_hornweg2[geluid_hornweg2["time"] == time]["Latitude"],
                    lon=geluid_hornweg2[geluid_hornweg2["time"] == time]["Longitude"],
                    mode='markers',
                    marker=dict(
                        size=(geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel']**6) / 1500000000, 
                        color=geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel'], 
                        opacity=0.3
                    ),
                    name="Hornweg",
                    text=[f"<span style='font-size:16px;'><b>Geluid: {round(db,1)} dB</b></span>" 
                        for db in geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel']],                 
                    hoverinfo="text"
                ),

                                
            ],
            "name": str(time)  # Use the time as the frame name
        }
        frames.append(frame)

    fig2.frames = frames

    # Add lines connecting the points for tui_vlucht (this will animate along with the points)
    fig2.add_trace(go.Scattermap(
        lat=df["Latitude"],
        lon=df["Longitude"],
        mode='lines',  # Only draw the line
        line=dict(color='red', width=1),
        name="TUI Vliegtuig Route",
        opacity=0.7
    ))

    # Add geodata locations as a separate trace
    fig2.add_trace(go.Scattermap(
        lat=geodata_geluid_yuri['Latitude'],
        lon=geodata_geluid['Longitude'],
        mode='markers+text',
        marker=dict(size=9, color='blue', opacity=0.5),
        text=geodata_geluid_yuri['location_long'],
        name="Geodata Locations"
    ))

    fig2.add_trace(go.Scattermap(
        lat=geluid_hornweg2[geluid_hornweg2["time"] == times[0]]["Latitude"],
        lon=geluid_hornweg2[geluid_hornweg2["time"] == times[0]]["Longitude"],
        # mode='markers',
        # marker=dict(size=geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel'], 
        #             color=geluid_hornweg2[geluid_hornweg2["time"] == time]['decibel'], 
        #             opacity=0.3),

        name="Hornweg",))

    # Add the slider
    sliders_dict = {
        "steps": [
            {
                "args": [
                    [str(time)],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }
                ],
                "label": str(time),
                "method": "animate"
            }
            for time in times
        ],
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 15},
            "prefix": "Time: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "bounce-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
    }

    fig2.update_layout(sliders=[sliders_dict],showlegend = False)

    # Show the map
    # fig2.show()


    col1, col2 = st.columns([1 ,1],border = True)
    with col1:  
        col4, col5, col6, col7, col8 = st.columns([1, 10, 10, 10, 10])
        with col5:
            st.write("### Airbus 330")
        with col6:
            st.write("### TK 1952")
        with col7:
            st.write("### 23 maart 2025")
        with col8:
            st.write("### 11 : 32 : 00")

        st.markdown("""---""")

        col4, col5, col6, col7 = st.columns([1, 13, 13, 13])
        with col5:
            st.metric(label="Speed at peak measurement (knts)", value='173') 
            st.metric(label="Altitude at peak measurement (m):", value='420',delta='-32')
        with col6:
            st.metric(label="Windspeed (knts)", value='3')
            st.metric(label="Distance at peak measurement (m)", value='587',delta='-49')    
        with col7:  
            st.metric(label="Engine", value='RR Trent 700')      
            st.metric(label="Winddirection (degrees)", value='167', delta= '6',delta_color='off')     


        st.plotly_chart(fig2)
    with col2:   
        col4, col5, col6, col7, col8 = st.columns([1, 10, 10, 10, 10])
        with col5:
            st.write("### Airbus 350")
        with col6:
            st.write("### QR 276")
        with col7:
            st.write("### 23 maart 2025")
        with col8:
            st.write("### 12 : 01 : 40")

        st.markdown("""---""")

        col4, col5, col6, col7 = st.columns([1, 13, 13, 13])
        with col5:
            st.metric(label="Speed at peak measurement (knts)", value='173')   
            st.metric(label="Altitude at peak measurement (m):", value='452',delta='32')
        with col6:
            st.metric(label="Windspeed (knts)", value='3')
            st.metric(label="Distance at peak measurement (m)", value='636',delta='49')    
        with col7: 
            st.metric(label="Engine", value='RR Trent XWB',delta="")        
            st.metric(label="Winddirection (degrees)", value='161' ,delta= '-6',delta_color='off')     

        st.plotly_chart(figa350)



    ###########################################################################################################################
    ####################################     EINDE VAN YURI ZIJN KAARTEN            ###########################################
    ###########################################################################################################################

    st.markdown('''---''')
    ########################### Maak een 2-koloms lay-out: legenda links, kaart rechts ########################################

    col1, col2, col3 = st.columns([2,6,5])  # 1/4 voor legenda, 3/4 voor de kaart

    with col1:
        st.markdown("""
            <div style="background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; width: 100%; text-align: left;">
                <b>SEL Color Legend</b><br>
                <span style="display: inline-block; width: 20px; height: 20px; background-color: green; margin-right: 10px;"></span> < 10,000,000<br>
                <span style="display: inline-block; width: 20px; height: 20px; background-color: yellow; margin-right: 10px;"></span> 10,000,000 - 50,000,000<br>
                <span style="display: inline-block; width: 20px; height: 20px; background-color: orange; margin-right: 10px;"></span> 50,000,000 - 100,000,000<br>
                <span style="display: inline-block; width: 20px; height: 20px; background-color: red; margin-right: 10px;"></span> > 100,000,000
            </div>
        """, unsafe_allow_html=True)

        # Selecteer vliegtuigtype met radio buttons
        selected_vliegtuig = st.radio("Selecteer vliegtuigcategorie:", ["A330", "A350", "Other"], horizontal=True, key="vliegtuig_radio")

    ###### Folium kaart met stations geplot #######
    # Coördinaten van Schiphol
    schiphol_coords = [52.2804, 4.7743]

    # Maak een kaart gecentreerd op Schiphol 
    m = folium.Map(location=schiphol_coords, zoom_start=11)

    # Functie om kleur op basis van SEL te bepalen
    def get_color(sel_value):
        if pd.notna(sel_value):
            if sel_value < 10_000_000:
                return 'green'
            elif sel_value < 50_000_000:
                return 'yellow'
            elif sel_value < 100_000_000:
                return 'orange'
            else:
                return 'red'
        else:
            return 'gray'  # Voor rijen zonder SEL-waarde

    # Filter de dataset
    filtered_data = geodata_geluid[geodata_geluid["vliegtuig_categorie"] == selected_vliegtuig]

    # Voeg markers en cirkels toe aan de kaart
    for _, row in filtered_data.iterrows():
        color = get_color(row['gemiddelde_SEL'])  

        marker = folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=(f"Straat: {row['location_long']}<br>"
                f"Plaats: {row['Plaats']}<br>"
                f"Vliegtuigcategorie: {row['vliegtuig_categorie']}<br>"
                f"Gemiddelde SEL: {row['gemiddelde_SEL'] if pd.notna(row['gemiddelde_SEL']) else 'Geen data'} J<br>"
                f"Afstand: {row['gemiddelde_distance'] if pd.notna(row['gemiddelde_distance']) else 'Geen data'} m"),
            icon=folium.Icon(color=color)
        ).add_to(m)  # 🔹 Voeg de marker toe aan de kaart

        # Pas de radius aan op basis van de SEL_dB waarde
        radius = row['gemiddelde_SEL_dB'] ** 5.9/206500000 if pd.notna(row['gemiddelde_SEL_dB']) else 100

        circle = folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.1
        ).add_to(m)  # 🔹 Voeg de cirkel toe aan de kaart

    with col2:
        # Toon de kaart
        st_folium(m, width=800, height=500)

    with col3:
        st.write("### Sensor data")
        st.write("Voordat er verder wordt ingegaan op deze data is het van belang om goed te begrijpen waar het eigenlijk over gaat:")
        st.write("### Wat is SEL?")
        st.write("Ter verduidelijking; Sound Exposure Level (SEL) is een maat voor de totale geluidsenergie van een geluidsevenement, zoals een overvliegend vliegtuig. SEL wordt uitgedrukt in decibels (dB) en corrigeert voor de duur van het geluid, zodat korte en lange geluiden eerlijk vergeleken kunnen worden.")
        st.write("")
        st.write("### Decibels en de logaritmische schaal")
        st.write("Decibels werken met een logaritmische schaal, wat betekent dat een kleine toename in dB een grote impact heeft op de waargenomen geluidsintensiteit. Een stijging van 3 dB verdubbelt de geluidsenergie, terwijl een stijging van 10 dB het geluid 10 keer zo intens maakt. Dit verklaart waarom kleine dB-verschillen in de praktijk goed hoorbaar zijn. **Kort gezegd:** SEL helpt om verschillende geluiden objectief te vergelijken, terwijl de logaritmische schaal van dB laat zien hoe snel geluid in intensiteit toeneemt.")

    st.markdown('''---''')
    st.write('Bronvermelding')
    col1, col2,col3 = st.columns([1 ,1,1])
    with col1: 
        st.write('https://sensornet.nl/project/aalsmeer/bov')
    with col2: 
        st.write('https://www.flightradar24.com/data/flights/qr276')
    with col3:
        st.write('https://www.flightradar24.com/data/flights/tk1952')
    st.markdown('''---''')

################################## TAB 3 BOUWEN ###########################################
with tab3:

    st.title("A330 VS. A350: Vliegtuigen Vergelijken :")
    st.write("Hier word de vergelijking gemaakt tussen de A330 VS. de A350. Aan de hand van meerdere plots wordt er gekeken welke er meer geluid produceert.")
    st.write("")

    # KPI tabs
    st.header("Alles snel op een rijtje")
    # KPI-kaarten naast elkaar zetten
    col1, col2 = st.columns(2)

    # gemiddelden berekenen voor de KPI tabs
    gemiddelden = capa_filtered.groupby('vliegtuig_categorie')[['Aircraft Age', 'Total Seats', 'Maximum Takeoff Weight', 'SEL', 'SEL_dB']].mean().round(0)
    # Bereken SEL per passagier (SEL_dB gedeeld door Total Seats)
    gemiddelden['SEL per passagier'] = (gemiddelden['SEL'] / gemiddelden['Total Seats']).round(0)
    # bereken SEL_dB per passagier (SEL_dB gedeeld door Total Seats)
    gemiddelden['SEL_dB per passagier'] = (gemiddelden['SEL_dB'] / gemiddelden['Total Seats']).round(2)

    # Bereken het verschil tussen A350 en A330 voor de 'SEL per passagier'
    a330_sel_per_passagier = gemiddelden.loc['A330', 'SEL per passagier']
    a350_sel_per_passagier = gemiddelden.loc['A350', 'SEL per passagier']

    a330_seldb_per_passagier = gemiddelden.loc['A330', 'SEL_dB per passagier']
    a350_seldb_per_passagier = gemiddelden.loc['A350', 'SEL_dB per passagier']

    # verschil uitrekenen voor cool pijltje
    verschil = a350_sel_per_passagier - a330_sel_per_passagier
    verschil_percentage = ((verschil / a330_sel_per_passagier) * 100).round(2)
    verschil_percentage.astype(int)

    verschil_2 = a350_seldb_per_passagier - a330_seldb_per_passagier
    verschil_2_percentage = ((verschil_2 / a330_seldb_per_passagier) * 100).round(2)
    verschil_2_percentage.astype(int)

    # A330 KPI's
    with col1:
        st.subheader("✈️ Airbus A330")
        st.metric(label="Gemiddeld Aantal Stoelen", value=int(gemiddelden.loc['A330', 'Total Seats']))
        st.metric(label="Maximale Startmassa (MTOW)", value=int(gemiddelden.loc['A330', 'Maximum Takeoff Weight']))
        st.metric(label="dB per passagier", value=float(gemiddelden.loc['A330', 'SEL_dB per passagier']))
        st.metric(label="SEL per passagier", value=f"{gemiddelden.loc['A330', 'SEL per passagier']} J")

    # A350 KPI's
    with col2:
        st.subheader("✈️ Airbus A350")
        st.metric(label="Gemiddeld Aantal Stoelen", value=int(gemiddelden.loc['A350', 'Total Seats']))
        st.metric(label="Maximale Startmassa (MTOW)", value=int(gemiddelden.loc['A350', 'Maximum Takeoff Weight']))
        st.metric(label="dB per passagier", value=float(gemiddelden.loc['A350', 'SEL_dB per passagier']), delta=f"{verschil_2_percentage}%", delta_color='inverse')
        st.metric(label="SEL per passagier", value=f"{gemiddelden.loc['A350', 'SEL per passagier']} J", delta=f"{verschil_percentage}%", delta_color='inverse')

    ########### plotje met geluid van 330 vs 350 ####################

    # Plot maken met automatische kleuren
    fig = px.bar(
        gemiddelde_sel_periode, 
        x="vlucht_periode", 
        y="SEL_dB",  # Default waarde
        color="vliegtuig_categorie",  
        barmode="group",  
        labels={"SEL_dB": "SEL (dB)", "vlucht_periode": "Vluchtperiode", "vliegtuig_categorie": "Vliegtuigtype"},
        template="plotly_white",
        color_discrete_sequence=klm_colors
    )

    # Stel de y-as range in vanaf het begin, zodat deze begint bij 65 dB
    fig.update_yaxes(range=[65, None])

    # Stel de y-as range in vanaf het begin, zodat deze begint bij 60 dB
    fig.update_layout(
        yaxis=dict(
            title="SEL (dB)", 
            range=[65, None]  # Y-as start bij 60 dB
        )
    )

    # Dropdown toevoegen om te wisselen tussen SEL_dB en SEL
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "label": "SEL_dB", 
                    "method": "update", 
                    "args": [
                        {"y": [gemiddelde_sel_periode.loc[gemiddelde_sel_periode["vliegtuig_categorie"] == cat, "SEL_dB"].values for cat in gemiddelde_sel_periode["vliegtuig_categorie"].unique()]}, 
                        {"yaxis": {"title": "SEL_dB (dB)", "range": [60, None]}}
                    ]
                },
                {
                    "label": "SEL", 
                    "method": "update", 
                    "args": [
                        {"y": [gemiddelde_sel_periode.loc[gemiddelde_sel_periode["vliegtuig_categorie"] == cat, "SEL"].values for cat in gemiddelde_sel_periode["vliegtuig_categorie"].unique()]}, 
                        {"yaxis": {"title": "totale SEL"}}
                    ]
                }
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }]
    )

    # Toon de plot in Streamlit
    st.plotly_chart(fig)

    st.write("Niet alleen is het vliegtuigtype van belang, een vliegtuig maakt aanzienlijk meer lawaai als deze opstijgt. In het volgende plot worden de A350 en A330 met elkaar vergeleken tijdens take-off en de landing")

    # arrival vs departure
    # Gemiddelde SEL per vliegtuigtype en vluchtstatus (Arrival/Departure)

    # Splits de data in arrivals en departures
    arrivals = arrivals_vs_departure[arrivals_vs_departure["FlightType"] == "Arrivals"]
    departures = arrivals_vs_departure[arrivals_vs_departure["FlightType"] == "Departures"]

    # Maak een subplot met 1 rij en 2 kolommen
    fig = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True,  # Zelfde schaal voor de y-as
        subplot_titles=("Arrivals", "Departures")
    )

    # Unieke vliegtuigtypes voor kleurconsistentie
    unique_vliegtuigen = arrivals_vs_departure["vliegtuig_categorie"].unique()

    # Voeg arrivals toe aan subplot 1 (linkerkant)
    for i, vliegtuig in enumerate(unique_vliegtuigen):
        subset = arrivals[arrivals["vliegtuig_categorie"] == vliegtuig]
        fig.add_trace(
            go.Bar(
                x=subset["vliegtuig_categorie"], 
                y=subset["SEL_dB"], 
                name=f"{vliegtuig} (Arrivals)", 
                marker_color=klm_colors[i % len(klm_colors)] 
            ),
            row=1, col=1
        )

    # Voeg departures toe aan subplot 2 (rechterkant)
    for i, vliegtuig in enumerate(unique_vliegtuigen):
        subset = departures[departures["vliegtuig_categorie"] == vliegtuig]
        fig.add_trace(
            go.Bar(
                x=subset["vliegtuig_categorie"], 
                y=subset["SEL_dB"], 
                name=f"{vliegtuig} (Departures)", 
                marker_color=klm_colors[i % len(klm_colors)] 
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title="SEL (dB) per vliegtuigtype en vluchtphase",
        yaxis_title="SEL (dB)",
        height=500, width=1000,  # Pas de grootte aan
        template="plotly_white",
        barmode="group",  # Groepeer de bars per vliegtuigtype
        showlegend=False,  # Legenda weghalen (voorkomt dubbele entries)
        yaxis=dict(range=[65,None])
    )

    # Toon de plot
    st.plotly_chart(fig)
   

################################## TAB 5 BOUWEN ##############################################

with tab4 :

    st.title("Conclusions")
    st.write("Na grondige analyses van de geluidsdata van de Airbus A330 en A350, kunnen we concluderen dat de A350 inderdaad stiller is dan de A330. De geluidsniveaus van de A350, gemeten tijdens de verschillende vluchten, liggen consistent lager dan die van de A330. Voor de inwoners van Amstelveen en Aalsmeer betekent dit dat, met de mogelijke komst van de A350 binnen de vloot van KLM, zij in de toekomst waarschijnlijk minder last zullen hebben van geluidsoverlast van vliegende toestellen. Deze conclusie biedt een positief vooruitzicht voor de levenskwaliteit van de bewoners in deze gebieden, aangezien de A350, door zijn lagere geluidsniveaus, minder storend zal zijn in vergelijking met de A330.")
    