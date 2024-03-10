# _io_streamlit.py
# C:\Dropbox\_\work\_io_streamlit.py

from __future__ import division         # DEV_TET_RiskParity
import os
import datetime as dt
import time
import sys
import os
import shutil
import socket
import getpass

import json
import requests
from random import random
import string
import secrets
import re
import collections
from collections import Counter
from difflib import SequenceMatcher

# import matplotlib.pyplot as plt
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)
# import matplotlib.dates as mdates
import requests                 # for web scraping
# from bs4 import BeautifulSoup   # for web scraping
# from shapely.geometry import Polygon
# from matplotlib.patches import Polygon as PolygonPatch

import math
import random
# from scipy.stats import norm
# import plotly.figure_factory as ff


# class TOC = Table of Contents
class TOC:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def h1(self, text):
        self._markdown(text, "h1")

    def h2(self, text):
        self._markdown(text, "h2", " " * 2)

    def h3(self, text):
        self._markdown(text, "h3", " " * 4)

    def h4(self, text):
        self._markdown(text, "h4", " " * 6)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        # key = "".join(filter(str.isalnum, text)).lower()
        key = text.replace(" ", "-").lower()
        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

# Launch .bat file
# C:\Dropbox\_\work\_io_streamlit.bat
# os.system("C:\Dropbox\_\work\_io_streamlit.bat")  # KO = End up in loop

# info
s_dt_T_yyyymmdd_hhmmss_raw = dt.datetime.today().strftime('%Y%m%d %H:%M:%S')
s_dt_T_yyyymmdd_hhmmss = str.replace(str.replace(s_dt_T_yyyymmdd_hhmmss_raw," ","_"),":","")
s_dt_T_yyyymmdd = dt.datetime.today().strftime('%Y%m%d')
s_dt_datetime = str(dt.datetime.now())
s_dt_date = str(dt.datetime.now().date())
s_dt_time = str(dt.datetime.now().time())
s_path = os.path.dirname(sys.executable)
s_this_file_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
s_this_file_drive = s_this_file_p[:2]
s_this_file_ne = os.path.basename(__file__)
s_this_file_pne = s_this_file_p + '\\' + s_this_file_ne
s_this_file_archive_p = s_this_file_p + '\\' + "_archive"
s_this_file_archive_ne = str.replace(s_this_file_ne,".py", "_" + s_dt_T_yyyymmdd_hhmmss + ".py")
s_this_file_archive_pne = s_this_file_archive_p + '\\' + s_this_file_archive_ne
s_hostname = socket.gethostname()
if socket.gethostname() == 'LENOVO2022':
    s_env = 'Home'
    # import libraries custom
    sys.path.append(s_this_file_drive + "/___lib_py")
    sys.path.append(s_this_file_drive + "/___lib_py_bbg")
s_username = getpass.getuser()

# import libraries standard
import streamlit as st
# from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from numpy import *
from numpy.random import rand
# import scipy
# import scipy.stats
# from pylab import *
# import matplotlib
# import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

import time
# import dropbox
# from sqlalchemy import create_engine  # GAMA: create connection to DB
import urllib
import requests

# import ___lib_py.PY_Sys_Dbx as PY_Sys_Dbx
# import ___lib_py.PY_Sys_File_InputOutput as PY_Sys_File_InputOutput
# import ___lib_py.PY_Sys_DataStructures as PY_Sys_DataStructures

# import ___lib_py.PY_Math as PY_Math

# C:\Dropbox\_\work\___lib_py\PY_Fin_Deriv.py
# import ___lib_py.PY_Fin_Deriv as PY_Fin_Deriv

# import ___lib_py_yahoo.ydh as ydh
# import ___lib_py_finnhub.fdh as fdh

# from matplotlib import pyplot as plt    # DEV_TET_RiskParity
from numpy.linalg import inv, pinv      # DEV_TET_RiskParity
import numpy as np                      # DEV_TET_RiskParity
# from scipy.optimize import minimize     # DEV_TET_RiskParity


# import ___lib_py_bbg.bdp as bdp         # bdp.request
# import ___lib_py_bbg.bdh as bdh         # bdh.bbapi_bdh
# import ___lib_py_bbg.bbg_bdp_bdh as bbg_bdp_bdh

# page layout
st.set_page_config(layout="wide")
# st.title("Wylie CHAN")

# st.sidebar.image("_io_streamlit__quantbible_apple-touch-icon_250x250.png", width=100)
sidebar_button_archive = st.sidebar.button("Archive")

if sidebar_button_archive:
    # back up this file
    s_src = s_this_file_pne
    s_dst = s_this_file_archive_pne
    shutil.copy(s_src, s_dst)
    st.sidebar.text("archived at: " + s_dt_T_yyyymmdd_hhmmss)

# https://blog.streamlit.io/session-state-for-streamlit/
# if s_env == 'Home':
#     pass
# elif s_env == 'Work':
#     pass
# else:
#     pass

# SelectBox
list_modes = ['<select>',
              'DEV',
              'X_Process_Simulation',
              'V_FinDeriv',
              'Risk Parity: REC: Data',
              'Risk Parity: REC: Analysis',
              'ECXDRKP',
              'DEV_TET_RiskParity',
              'DEV_TET_Hangman',
              'DEV_Bloomberg',
              'DEV_SQL',
              '_prd/_data_sg_cb_mas/_api_R_r_SG_SGD',
              '_prd/_data_sg_cb_mas/_api_I_Life',
              'PY_Virtual_Env',
              'PY_Template',
              'PY_Template_Lite',
              'Jelly_Belly',
              'PY_Pandas',
              'PY_NumPy',
              'PY_DateTime_Holiday',
              'PY_DateTime',
              'PY_Environment_Virtual',
              'PY_Module',
              'PY_Class_Object__Broker',
              'PY_Class_Object__Constructor',
              'PY_Class_Object__Attribute_Method',
              'PY_Files_Import_py_Library_Package_Module',
              'PY_Files_Import_py',
              'PY_Exceptions_Handling_SaveToFile',
              'PY_Exceptions_Handling',
              'PY_Exceptions',
              'PY_Files',
              'PY_Functions_Dict_IO_Streamlit',
              'PY_Functions_Dict_IO_Console',
              'PY_Functions_Dict_Print',
              'PY_Functions_Parameters_Arguments',
              'PY_Functions',
              'PY_Loops_Nested_Console',
              'PY_Loops',
              'PY_Loops_Type_04',
              'PY_Loops_Type_03',
              'PY_Loops_Type_02',
              'PY_Loops_Type_01',
              'PY_DataStructures_Blog_InvestmentPortfolio',
              'PY_DataStructures',
              'PY_DataStructures_Dictionaries',
              'PY_DataStructures_Tuples',
              'PY_DataStructures_Lists',
              'PY_UserInput',
              'PY_Conditionals',
              'PY_Strings',
              'PY_Math',
              'PY',
              ]
sidebar_selectbox_mode = st.sidebar.selectbox("Mode", list_modes)

#################################################
# Functions to be moved to libraries
#################################################

def graph_pair(x1, x2, b_norm = False):
    color1 = 'black'
    color2 = 'green'
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.autofmt_xdate(rotation=45)
    ax.legend()

    if b_norm==False:
        df_graph = df[[x1, x2]].copy()  # select columns to output
        ax.plot(df_graph.index, df_graph[x1], color=color1)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel(x1 + ' [USD]', color=color1, fontsize=12)
        ax2 = ax.twinx()
        ax2.plot(df_graph.index, df_graph[x2], color=color2)
        ax2.set_ylabel(x2 + ' [USD]', color=color2, fontsize=12)
        plt.title('Close of '+x1+', '+x2+' from {} to {}'.format(dt_start.date(), dt_end.date()))
    else:
        df_graph = df[[x1+'_norm', x2+'_norm']].copy()  # select columns to output
        ax.plot(df_graph.index, df_graph[x1+'_norm'], color=color1)
        ax.plot(df_graph.index, df_graph[x2+'_norm'], color=color2)
        ax.set_xlabel('Date', fontsize=14)
        # ax.set_ylabel(x1+'_norm' + ' [USD]', color=color1, fontsize=12)
        # ax.set_ylabel(x2+'_norm' + ' [USD]', color=color2, fontsize=12)
        plt.title('Close of '+x1+'_norm'+', '+x2+'_norm'+' from {} to {}'.format(dt_start.date(), dt_end.date()))

    dtFmt = mdates.DateFormatter('%Y%m%d')  # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.gca().xaxis.set_minor_formatter(dtFmt)
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.show()
    st.pyplot(fig)

# This algorithm generates Poisson random variables
def Poisson(M,theta):
#M: number of simulations
#theta: average number of jumps

    N = zeros(M)

    for i in range(0,M):
        p = exp(-theta)                     #probability of N = 0
        F = p
        N[i] = 0
        U = rand()
        while U > F:
            N[i] = N[i]+1
            p = p*theta/N[i]
            F= F+p

    return N

#################################################################################
#scipy function to get the inverse distribution
def Poisson_scipy(M,N,theta):
    N = scipy.stats.distributions.poisson.ppf(rand(M,N),theta)
    return N

# Function to scrape data from a website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Replace this with your actual web scraping logic
    # Below is just a dummy example
    # data = {'Category': ['A', 'B', 'C'], 'Value': [10, 20, 30]}
    # st.write(type(soup))

    # st.write(soup.text)
    # st.write(soup.prettify())
    st.subheader("soup.title")
    try:
        st.write(soup.title)
    except:
        st.write("Error")

    st.subheader("soup.title.name")
    st.write(soup.title.name)
    st.subheader("soup.title.string")
    st.write(soup.title.string)
    st.subheader("soup.title.parent.name")
    st.write(soup.title.parent.name)
    st.subheader("soup.p")
    try:
        st.write(soup.p)
    except:
        st.write("Error")
    st.subheader("soup.p['class']")
    st.write(soup.p['class'])




    data = soup
    return pd.DataFrame(data)



if sidebar_selectbox_mode == "DEV":
    sidebar_title = st.sidebar.title("sidebar.title")
    sidebar_text = st.sidebar.text("sidebar.text")

    checkbox_expander = st.sidebar.checkbox("sidebar.expander")
    if checkbox_expander:
        st.header("Expander")
        with st.expander("Click to expand"):
             st.write("""
                HelloWorld!
             """)
             st.image("https://static.streamlit.io/examples/dice.jpg")

    checkbox_info = st.sidebar.checkbox("sidebar.checkbox.info")
    if checkbox_info:
        st.header("Info")
        with st.expander("Click to expand"):
            st.write("checkbox_info")
            st.text('s_dt_T_yyyymmdd_hhmmss_raw    = ' + s_dt_T_yyyymmdd_hhmmss_raw)
            st.text('s_dt_T_yyyymmdd_hhmmss        = ' + s_dt_T_yyyymmdd_hhmmss)
            st.text('s_dt_T_yyyymmdd               = ' + s_dt_T_yyyymmdd)
            st.text('s_dt_datetime                 = ' + s_dt_datetime)
            st.text('s_dt_date                     = ' + s_dt_date)
            st.text('s_dt_time                     = ' + s_dt_time)
            st.text('s_hostname                    = ' + s_hostname)
            st.text('s_username                    = ' + s_username)
            st.text('path                          = ' + s_path)
            st.text('this file drive               = ' + s_this_file_drive)
            st.text('this file path                = ' + s_this_file_p)
            st.text('this file name ext            = ' + s_this_file_ne)
            st.text('this file path name ext       = ' + s_this_file_pne)
            st.text('archive path                  = ' + s_this_file_archive_p)
            st.text('archive name ext              = ' + s_this_file_archive_ne)
            st.text('archive path name ext         = ' + s_this_file_archive_pne)
            # st.text('s_env                         = ' + s_env)

    checkbox_buttons = st.sidebar.checkbox("sidebar.buttons")
    if checkbox_buttons:
        st.header("Buttons")
        with st.sidebar.expander("Buttons"):
            sidebar_button_00000010 = st.sidebar.button("sidebar.button.01", key="sidebar_button_00000010")
            sidebar_button_00000020 = st.sidebar.button("sidebar.button.02", key="sidebar_button_00000020")
        if sidebar_button_00000010:
            st.write("sidebar.button.01")
        if sidebar_button_00000020:
            st.write("sidebar.button.02")

    checkbox_selectbox = st.sidebar.checkbox("sidebar.selectbox")
    if checkbox_selectbox:
        st.header("SelectBox")
        sidebar_selectbox = st.sidebar.selectbox('sidebar.selectbox', ('sidebar.selectbox.value_01', 'sidebar.selectbox.value_02'))
        if sidebar_selectbox == 'sidebar.selectbox.value_01':
            st.write("sidebar.selectbox.value_01")
        if sidebar_selectbox == 'sidebar.selectbox.value_02':
            st.write("sidebar.selectbox.value_02")

        st.header("SelectBox: Values from DataFrame Column")
        df = pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
            })
        option = st.selectbox(
            'Which number do you like best?',
             df['first column'])
        'You selected: ', option

    checkbox_slider = st.sidebar.checkbox("sidebar.slider")
    if checkbox_slider:
        st.header("SideBar Slider")
        sidebar_slider = st.sidebar.slider('sidebar.slider', 0.0, 100.0, (25.0, 75.0))
        st.write(sidebar_slider)
        st.write(sidebar_slider[0])
        st.write(sidebar_slider[1])
        st.header("Slider")
        x = st.slider('x')  # 👈 this is a widget
        st.write(x, 'squared is', x * x)

    def spinner_task():
        # Simulating a task that takes some time
        time.sleep(5)
        st.success("Spinner done!")

    checkbox_spinner = st.sidebar.checkbox("sidebar.spinner")
    if checkbox_spinner:
        st.header("Spinner")

        # Button to trigger the task
        if st.button("Start Spinner [5 sec]"):
            # Display the spinner while the task is running
            with st.spinner("Spinner spinning..."):
                # Simulate a task
                spinner_task()

    checkbox_progress = st.sidebar.checkbox("sidebar.progress")
    if checkbox_progress:
        st.header("Progress Bar")
        sidebar_progressbar = st.sidebar.progress(0)
        sidebar_progressbar_iteration = st.sidebar.empty()
        for i in range(100):
            sidebar_progressbar.progress(i + 1)
            sidebar_progressbar_iteration.text(f'Iteration {i + 1}/100')
            time.sleep(0.01)

        'Starting a long computation...'
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            # Update the progress bar with each iteration.
            bar.progress(i + 1)
            latest_iteration.text(f'Iteration {i + 1}')
            time.sleep(0.02)
        '...and now we\'re done!'

    checkbox_radio = st.sidebar.checkbox("sidebar.radio")
    if checkbox_radio:
        st.header("Radio")
        chosen = st.radio(
            'Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
        st.write(f"You are in {chosen} house!")

    checkbox_columns = st.sidebar.checkbox("sidebar.columns")
    if checkbox_columns:
        st.header("Columns")
        col_01, col_02, col_03 = st.columns(3)
        col_01.text('Name')
        with col_01:
            col_01_chosen = st.radio('Name',("Wylie", "Jaehee", "Isaac", "Isabel"))
        col_02.text('Year of Birth')
        with col_02:
            col_02_chosen = st.radio('Year of Birth',("1979", "1986", "2008", "2013"))
        col_03.text('Age')
        st.write(f"My name is {col_01_chosen} CHAN!")
        st.write(f"I was born in the year {col_02_chosen}!")

        col_01, col_02, col_03 = st.columns(3)
        with col_01:
            st.header("A cat")
            st.image("https://static.streamlit.io/examples/cat.jpg")
        with col_02:
            st.header("A dog")
            st.image("https://static.streamlit.io/examples/dog.jpg")
        with col_03:
            st.header("An owl")
            st.image("https://static.streamlit.io/examples/owl.jpg")

    checkbox_echo = st.sidebar.checkbox("sidebar.echo")
    if checkbox_echo:
        st.header("Echo")
        with st.echo():
            # Your code here
            x = 5
            y = 10
            result = x + y
            st.write(result)

    checkbox_run = st.sidebar.checkbox("sidebar.run")
    if checkbox_run:
        st.header("run")
        st.text("-----" + " " + "streamlit run" + " " + "-" * 100)
        st.text('> streamlit run your_script.py [-- script args]')
        st.text('> $ python -m streamlit run your_script.py')
        st.text('> $ streamlit run your_script.py')
        st.text('> $ streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py')

    checkbox_df = st.sidebar.checkbox("dataframe")
    if checkbox_df:
        st.header("DataFrame")
        df = pd.DataFrame({
          'first column': [1, 2, 3, 4],
          'second column': [10, 20, 30, 40]
        })
        df

        st.header("DataFrame: st.write")
        st.write(df)

        st.header("DataFrame: st.table")
        st.table(df)

        st.header("DataFrame: random")
        dataframe = np.random.randn(10, 20)
        st.dataframe(dataframe)

        st.header("DataFrame: random, column headers")
        st.text("-----" + " " + "st.dataframe + st.table" + " " + "-" * 100)
        dataframe = pd.DataFrame(np.random.randn(10, 20), columns=('col %d' % i for i in range(20)))
        st.table(dataframe)

        st.header("DataFrame: random, column headers, pd.Styler(highlight_max)")
        dataframe = pd.DataFrame(np.random.randn(10, 20), columns=('col %d' % i for i in range(20)))
        st.dataframe(dataframe.style.highlight_max(axis=0)) # In each column, highlight the max

        st.header("DataFrame: random, column headers, line chart")
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        st.line_chart(chart_data)

        st.header("DataFrame: random, column headers, bar chart")
        st.bar_chart(chart_data)
        st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

    checkbox_df_map = st.sidebar.checkbox("dataframe map")
    if checkbox_df_map:
        st.header("DataFrame Map Point: Home")
        map_data = pd.DataFrame([1.2992346724123316, 103.83427736786686] + np.array([[0, 0]]), columns=['lat', 'lon'])
        st.map(map_data)
        st.write(np.array([[0, 0]]))

        st.header("DataFrame Map Scatter: Home")
        map_data = pd.DataFrame([1.2992346724123316, 103.83427736786686] + np.random.randn(10, 2) / [100, 100], columns=['lat', 'lon'])
        st.map(map_data)
        st.write(np.random.randn(10, 2))

    checkbox_text_input = st.sidebar.checkbox("text_input")
    if checkbox_text_input:
        st.header("Text Input")
        st.text("-----" + " " + "widgets: st.textinput + key" + " " + "-" * 100)
        st.text_input("textinput_001", key="textinput_001")
        st.write(st.session_state.textinput_001)




if sidebar_selectbox_mode == "X_Process_Simulation":

    sidebar_radio_process = st.sidebar.radio('Process', (
        "GBM", "GBM_Histogram", "CIR", "Heston", "Jump Diffusion", "Poisson"))

    if sidebar_radio_process == "GBM":
        # Here we have the simulation of a Geometric Brownian Motion
        norminv = scipy.stats.distributions.norm.ppf
        norm = scipy.stats.distributions.norm.cdf

        M = 100  # Number of paths - We are using 100 paths for illustrative purposes - The more paths, the more accurate.
        d = 365  # Number of steps - We are using daily steps
        spot = 3900

        T = 1  # Time to maturity - We are using one year
        delta_t = T / d

        y1 = rand(M, d)
        z1 = norminv(y1)

        initial_spot_asset1 = spot * ones((M, 1))

        ASSET1 = zeros((M, d))
        ASSET1 = append(initial_spot_asset1, ASSET1, axis=1)

        r = -0.0175  # Interest rate
        sigma = 0.12  # Volatility

        # Calculate the paths
        for i in range(0, M):
            for j in range(0, d):
                ASSET1[i, j + 1] = ASSET1[i, j] * exp((r - 0.5 * sigma ** 2) * delta_t + sigma * sqrt(delta_t) * z1[i, j])

        # Plot the paths
        fig, ax = plt.subplots(figsize=(6, 3))                     # streamlit plot: Add this at start
        plt.plot(ASSET1[0:50, :].transpose(), c='silver')
        plt.plot(ASSET1[0:1, :].transpose(), c='darkblue')
        ylabel("Asset's price", fontsize=20)
        xlabel("Time", fontsize=20)
        box(on=None)
        show()
        st.pyplot(fig)                                              # streamlit plot: Add this at end

    if sidebar_radio_process == "GBM_Histogram":
    # https://medium.com/@nikitasinghiitk/monte-carlo-simulations-the-quants-playground-of-chance-5bc16d424d95
        # Define option parameters
        S0 = 100  # Initial stock price
        K = 105  # Strike price
        T = 1  # Time to maturity
        sigma = 0.2  # Volatility
        r = 0.05  # Risk-free rate

        # Define number of simulations
        n_sims = 10000

        # Generate random price paths using geometric Brownian motion
        dt = T / 252  # Discretize time into daily steps
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n_sims, 252))  # Generate random increments
        S = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW).cumsum(axis=1)

        # Calculate option payoffs at maturity
        call_payoff = np.maximum(S[:, -1] - K, 0)

        # Calculate option price using Monte Carlo average
        call_price = np.exp(-r * T) * np.mean(call_payoff)

        # Print option price
        st.write("Call option price:", call_price)

        # Plot distribution of simulated call payoffs
        plt.hist(call_payoff, bins=50, density=True, label="Call Payoff Distribution")
        plt.xlabel("Call Payoff")
        plt.ylabel("Density")
        plt.title("Distribution of Simulated Call Payoffs")
        plt.grid(True)
        plt.show()
        st.pyplot(gcf())

    if sidebar_radio_process == "CIR":

        # Here we have the simulation of a Cox Ingersoll Ross - CIR process

        ncx2inv = scipy.stats.distributions.ncx2.ppf  # Non Central Chi Square inverse function
        M = 10 ** 2  # Number of simulations
        N = 365  # Number of time steps
        T = 1  # Time to maturity
        delta_t = T / N  # Time step
        r0 = 0.05  # Initial r - interest rate
        initial_r = r0 * ones((M, 1))  # Initial r vector
        r = zeros((M, N))  # Declaration of the interest rate matrix (r)
        r = append(initial_r, r, axis=1)

        kappa = 2  # Mean reversion speed
        theta = 0.05  # Long run mean
        sigma = 0.2
        d = 4 * theta * kappa / sigma ** 2  # Degrees of freedom
        x = rand(M, N)
        w = zeros((M, N))
        lamda = zeros((M, N))

        for i in range(0, N):
            coeff1 = 4 * kappa
            coeff2 = (sigma ** 2) * (1 - exp(-kappa * delta_t))
            coeff3 = exp(-kappa * delta_t)

            lamda[:, i] = r[:, i] * (
                        (coeff1 * coeff3) / coeff2)  # Non central parameter according to the notes in Glasserman

            w[:, i] = ncx2inv(x[:, i], d, lamda[:, i])  # We generate a vector of non central chi squares
            # ncx2.ppf(prb, df, nc)-> df:degrees of freedom, nc:Non centrality parameter
            r[:, i + 1] = (coeff2 / coeff1) * w[:, i]

        fig, ax = plt.subplots(figsize=(6, 3))                     # streamlit plot: Add this at start
        plt.plot(r[0:50, :].transpose(), c='silver')
        plt.plot(r[0:1, :].transpose(), c='darkblue')
        ylabel("Interest rate", fontsize=20)
        xlabel("Time", fontsize=20)
        box(on=None)
        show()
        st.pyplot(fig)                                              # streamlit plot: Add this at end

    if sidebar_radio_process == "Heston":

        # Heston process simulation

        norminv = scipy.stats.distributions.norm.ppf
        norm = scipy.stats.distributions.norm.cdf
        M = 100  # Number of paths
        d = 365  # Number of steps

        T = 1  # Time to maturity
        delta_t = T / d

        y1 = rand(M, d)
        y2 = rand(M, d)
        y3 = rand(M, d)
        z1 = norminv(y1)
        z2 = norminv(y2)

        spot = 4700  # Spot
        r = 0.002  # Interest rate
        sigma = 0.29  # Volatility of volatility
        rho = -0.75  # Correlation spot-vol
        sigma1 = rho * sigma  # Relationship to generate two independent brownian motions
        sigma2 = sqrt(sigma ** 2 - sigma1 ** 2)
        kappa = 0.5  # Mean reversion constant for vol process
        theta = 0.0725  # Long run variance
        variance = 0.026  # Initial variance
        # strike = 4700        #strike de la opción
        # barrier = spot*0.8

        initial_spot_asset = spot * ones((M, 1))
        initial_variance_asset = variance * ones((M, 1))

        ASSET = zeros((M, d))
        ASSET = append(initial_spot_asset, ASSET, axis=1)

        ASSET_VARIANCE = zeros((M, d))
        ASSET_VARIANCE = append(initial_variance_asset, ASSET_VARIANCE, axis=1)

        for i in range(0, M):
            for j in range(0, d):
                if y3[i, j] > 0.5:
                    epsilon = delta_t
                else:
                    epsilon = -delta_t

                ASSET[i, j + 1] = ASSET[i, j] * (
                            1 + r * delta_t + sqrt(abs(ASSET_VARIANCE[i, j])) * sqrt(delta_t) * z1[i, j]) \
                                  + 0.5 * r ** 2 * ASSET[i, j] * delta_t ** 2 \
                                  + ((r + 0.25 * (sigma1 - kappa) * ASSET[i, j] * sqrt(abs(ASSET_VARIANCE[i, j]))) + (
                            0.25 * (kappa - theta) - 0.0625 * sigma ** 2) * (
                                                 ASSET[i, j] / sqrt(abs(ASSET_VARIANCE[i, j])))) * sqrt(delta_t) * z1[
                                      i, j] * delta_t \
                                  + 0.5 * ASSET[i, j] * (ASSET_VARIANCE[i, j] + 0.5 * sigma1) * (
                                              delta_t * z1[i, j] ** 2 - delta_t) + 0.25 * sigma2 * ASSET[i, j] * (
                                              sqrt(delta_t) * z2[i, j] * sqrt(delta_t) * z1[i, j] + epsilon)

                ASSET_VARIANCE[i, j + 1] = kappa * theta * delta_t + (1 - kappa * delta_t) * abs(ASSET_VARIANCE[i, j]) \
                                           + sqrt(abs(ASSET_VARIANCE[i, j])) * (
                                                       sigma1 * sqrt(delta_t) * z1[i, j] + sigma2 * sqrt(delta_t) * z2[
                                                   i, j]) - 0.5 * kappa ** 2 * (
                                                       theta - abs(ASSET_VARIANCE[i, j])) * delta_t ** 2 \
                                           + ((0.25 * kappa * theta - 0.0625 * sigma ** 2) / sqrt(
                    abs(ASSET_VARIANCE[i, j])) - 1.5 * kappa * sqrt(abs(ASSET_VARIANCE[i, j]))) * (
                                                       sigma1 * sqrt(delta_t) * z1[i, j] + sigma2 * sqrt(delta_t) * z2[
                                                   i, j]) * delta_t \
                                           + 0.25 * sigma1 ** 2 * (
                                                       delta_t * z1[i, j] ** 2 - delta_t) + 0.25 * sigma2 ** 2 * (
                                                       delta_t * z2[i, j] ** 2 - delta_t) + 0.5 * sigma1 * sigma2 * sqrt(
                    delta_t) * z1[i, j] * z2[i, j]

        # Plot the paths
        fig, ax = plt.subplots(figsize=(6, 3))                     # streamlit plot: Add this at start
        plt.plot(ASSET[0:50, :].transpose(), c='silver')
        plt.plot(ASSET[0:1, :].transpose(), c='darkblue')
        ylabel("Asset's price", fontsize=20)
        xlabel("Time", fontsize=20)
        box(on=None)
        show()
        st.pyplot(fig)                                              # streamlit plot: Add this at end

    if sidebar_radio_process == "Jump Diffusion":

        # Here we have the simulation of a Jump Difussion Process
        # It can be in two different ways. The first one is with the size of the jumps distributed lognormally.
        # The second one is more general. The example is shown for an exponential or a uniform distribution

        ########################################################################################################################################3
        # First method: Yi have a lognormal distribution with parameters (a,b)
        if True:
            lamda = 0.1  # Average number of jumps
            T = 1  # Time to maturity
            d = 365  # Number of steps
            delta_t = T / d  # Step size
            M = 10 ** 2  # Number of paths
            a = 0.25  # Average jump's size
            b = 0.05  # Jump's standard deviation
            m = exp(
                a + 0.5 * b ** 2) - 1  # m = Jump's average - 1. Since Yi is lognormal distributed, the average is exp(a+0.5b^2)

            r = 0.175  # Interest rate
            sigma = 0.12  # Volatilidty
            S0 = 3650  # Spot price

            # Random variables generation
            z1 = randn(M, d)  # Random variable to generate brownian motion
            z2 = randn(M, d)  # Random variable to generate jumps

            N = Poisson(M, lamda * delta_t)  # Generate Poisson random variable
            for i in range(0, d - 1):
                N = append(N, Poisson(M, lamda * delta_t))

            N = resize(N, (M, d))

            J = a * N + b * sqrt(N) * z2  # Sum of the jump's logarithm. These are lognormal distributed
            x = zeros((M, d + 1))
            x[:, 0] = log(S0)

            # Generacion del path
            for j in range(0, d):
                x[:, j + 1] = x[:, j] + (r - lamda * m - 0.5 * sigma ** 2) * delta_t + sigma * sqrt(delta_t) * z1[:, j] + J[
                                                                                                                          :,
                                                                                                                          j]

            path = exp(x)
            if True:
                # Plot first 50 paths
                fig, ax = plt.subplots(figsize=(6, 3))  # streamlit plot: Add this at start
                plt.plot(path[0:50, :].transpose(), c='silver')
                plt.plot(path[0:1, :].transpose(), c='darkblue')
                ylabel("Asset's price", fontsize=20)
                xlabel("Time", fontsize=20)
                box(on=None)
                show()
                st.pyplot(fig)                                              # streamlit plot: Add this at end

        ###########################################################################################################################################################

        if False:
            # Second method - much more general. Yi is not necessarily lognormal distributed. In this example we are using
            # jumps with uniform or exponential distribution. Recall that Yi can be interpreted as the return after the jump

            lamda = 0.1
            T = 1
            d = 365
            delta_t = T / d
            M = 10 ** 2
            a = 5  # If it is exponential: Average jump size, If it is uniform: Lower interval range
            b = 0  # If it is exponential: 0, If it is uniform: Upper interval range
            dist = 'exponential'

            if dist == 'exponential':
                m = 1 / a  # Average. Depends on the exponencial or uniform distribution
            elif dist == 'uniform':
                m = 0.5 * (a + b) - 1

            r = 0.0175
            sigma = 0.12
            S0 = 3650
            K = 100

            # Random variable generation
            z1 = randn(M, d)

            N = PoissonGenerator.Poisson(M, lamda * delta_t)
            for i in range(0, d - 1):
                N = append(N, PoissonGenerator.Poisson(M, lamda * delta_t))

            N = resize(N, (M, d))
            J = zeros((M, d))

            # Loop to get J, which is the sum of the jumps
            for i in range(0, M):
                for j in range(0, d):
                    if N[i, j] == 0:
                        J[i, j] = 0
                    else:
                        suma = 0
                        for k in range(0, int(N[i, j])):
                            if dist == 'exponential':
                                suma = suma + (-log(rand()) / a)
                            elif dist == 'uniform':
                                suma = suma + log((a + rand() * (b - a)))  # generate uniform random variable in [a,b]
                        J[i, j] = suma

            x = zeros((M, d + 1))

            # Generate paths
            for j in range(0, d):
                x[:, j + 1] = x[:, j] + (r - lamda * m - 0.5 * sigma ** 2) * delta_t + sigma * sqrt(delta_t) * z1[:, j] + J[
                                                                                                                          :,
                                                                                                                          j]

            path = S0 * exp(x)
            if True:
                # Plot first 50 paths
                fig, ax = plt.subplots(figsize=(6, 3))                      # streamlit plot: Add this at start
                plt.plot(path[0:50, :].transpose(), c='silver')
                plt.plot(path[0:1, :].transpose(), c='darkblue')
                ylabel("Asset's price", fontsize=20)
                xlabel("Time", fontsize=20)
                box(on=None)
                show()
                st.pyplot(fig)                                              # streamlit plot: Add this at end

    if sidebar_radio_process == "Poisson":
        # This is an algorithm to generate Poisson distributed random variables
        def Poisson(M, theta):
            # M is the number of simulations
            # Theta is the mean number of jumps

            N = zeros(M)

            for i in xrange(0, M):
                p = exp(
                    -theta)  # This is probability of N = 0. It is basically replacing N=0 in the formula for Poisson distribution
                F = p  # Make the cumulative probability equal to p(N=0).
                N[i] = 0
                U = rand()
                while U > F:
                    N[i] = N[i] + 1
                    p = p * theta / N[i]
                    F = F + p

            return N


        #################################################################################
        # This function is from scipy with the inverse distribution
        def Poisson_scipy(M, N, theta):
            N = scipy.stats.distributions.poisson.ppf(rand(M, N), theta)
            return N


        M = 100
        d = 365
        theta = 0.01

        JUMPS_vector = Poisson_scipy(M, d, theta)  # 100 paths, 365 days, 0.01 jumps per day on average

        # To acumulate the jumps
        JUMPS_vector_cum = zeros((M, d + 1))

        # Calculate the paths
        for i in range(0, M):
            for j in range(0, d):
                JUMPS_vector_cum[i, j + 1] = JUMPS_vector_cum[i, j] + JUMPS_vector[i, j]

        # Plot the paths
        fig, ax = plt.subplots(figsize=(6, 3))  # streamlit plot: Add this at start
        plt.plot(JUMPS_vector_cum[0:50, :].transpose(), c='silver')
        plt.plot(JUMPS_vector_cum[0:1, :].transpose(), c='darkblue')
        ylabel("Cumulative Jumps", fontsize=20)
        xlabel("Time", fontsize=20)
        box(on=None)
        show()
        st.pyplot(fig)  # streamlit plot: Add this at end

# ==============================================================================================================================================
# ===== V_FinDeriv =============================================================================================================================
# ==============================================================================================================================================
if sidebar_selectbox_mode == "V_FinDeriv":
    # create a TOC and put it in the sidebar
    toc = TOC()
    list_section = [
            "_",
            "R_B_ZC",
            "R_B_Perp",
            "R_B_Perp_Annuity",
            "R_B_Perp_Annuity_CPF_LIFE",
            "ECXRK_Opt: Simple",
            "ECXRK_Opt: Matrix",
    ]
    section = st.sidebar.radio("Select", list_section)

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    import sys
    # C:\Dropbox\_\work\___lib_py_github\wylie-chan-phillips-com-sg\pages\_io_streamlit.py
    # C:\Dropbox\_\work\___lib_py_github\wylie-chan-phillips-com-sg\pages\Fin_Deriv\R_B_ZC_Insur\R_B_ZC_Insur.py
    # from Fin_Deriv.R_B_ZC_Insur import R_B_ZC_Insur
    import Fin_Deriv.R_B_ZC_Insur.R_B_ZC_Insur as R


    s = "R_B_ZC"
    if section == s:
        toc.h1(s)

        # Create the Object using Standard Parameters
        s_ticker = r"R_B_ZC_Insur___NKSgd=020-200_Prem1_T=3y_r=3.62\%_Endow_Par0_LIC_WealthPlus3.1"
        R_B_ZC_Insur_obj = R.R_B_ZC_Insur(
            asset_class = "R",
            security_type = "B_ZC",
            ticker = s_ticker
        )
        R_B_ZC_Insur_obj.n = 3

        st.write(R_B_ZC_Insur_obj)

        c_display, c_text, c_input, c_formula = st.columns([3,2,1,1])
        with c_display:
            st.latex(R_B_ZC_Insur_obj.s_display_formula)
        with c_text:
            st.latex(R_B_ZC_Insur_obj.s_text_fields)
        with c_input:
            try:

                n = R_B_ZC_Insur_obj.n
                s_n = st.text_input(':green[Input: Time Horizon [year]:]',str(n))
                m = 1/n
                nm = n * m
                per = 1/m

                st.text('\n')
                s_m = st.text_input('Output: Coupon Freq [1/year]:',m)
                s_nm = st.text_input('Output: Total Num Coupons [period]:',str(nm))
                st.text('\n')
                s_per = st.text_input('Output: Period Length [year]:',str(per))
                st.text('\n')
                st.text('\n')

                # https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx
                # r_R_B_Govt_SG_3y     = 3.16   <==   2y = 3.24, 5y = 2.99
                r_R_B_Govt_SG_2y = 3.24
                r_R_B_Govt_SG_5y = 2.99
                r_R_B_Govt_SG_3y = r_R_B_Govt_SG_2y + 1 / 3 * (r_R_B_Govt_SG_5y - r_R_B_Govt_SG_2y) # = 3.15
                r_R_B_Bank_SG_1y_BOC = 3.10   # roll to year 2 and 3, rate not guaranteed
                r_R_B_Bank_SG_1y_RHB = 3.25   #  roll to year 2 and 3, rate not guaranteed
                r_R_B_SG_Best = r_R_B_Bank_SG_1y_RHB

                # st.markdown("""<style>.st-eb {background-color: black;}</style>""", unsafe_allow_html=True)
                s_r = st.text_input(':green[Input: Risk-Free Rate [%/year]:]', r_R_B_SG_Best)
                r = float(s_r)/100

                r_per = r/m
                s_r_per = st.text_input('Output: Risk-Free Rate [%/period]:', str(r_per * 100))
                # :color[green]

                st.text('\n')
                st.text('\n')
                s_c = st.text_input(':green[Input: Coupon Rate [%/year]:]', 3.62)
                c = float(s_c)/100
                c_per = c/m
                s_c_per = st.text_input('Output: Coupon Rate [%/period]:',str(c_per * 100))

                s_N = st.text_input(':green[Input: Notional(t=0) [SGD]:]', "{:,.2f}".format(20000))
                N = float(s_N.replace(",",""))

                # CF = c * N
                # s_CF = st.text_input('Output: CF [SGD/y]:', "{:,.2f}".format(CF))
                st.text('\n')
                st.text('\n')
                CF_per = N * (1+c_per)
                s_CF_per = st.text_input(':blue[Output: CF_per [SGD/per]:]', "{:,.2f}".format(CF_per))

                R_B_ZC = 1/(1+r_per) * CF_per
                s_R_B_ZC = st.text_input(':blue[Output: PV of Zero-Coupon [SGD]:]', "{:,.2f}".format(R_B_ZC))

                st.text('\n')
                st.text('\n')
                f_comm_FA_pct = R_B_ZC_Insur_obj.f_comm_pct_prem_plus_gst_to_FA / 100
                f_comm_FA_amt = f_comm_FA_pct * N
                s_f_comm_FA_amt = st.text_input(':blue[Output: Commission to FA [SGD]:]', "{:,.2f}".format(f_comm_FA_amt))

            except Exception as e:
                st.write(e)
                pass
        with c_formula:
            st.write('\n')
            st.write('\n')
            st.write(n)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(m)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(nm)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(r)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(r_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(c)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(c_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(N)
            # st.write(CF)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(CF_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(R_B_ZC)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(f_comm_FA_amt)



        with c_display:

            c_field, c_data = st.columns([1,1])
            with c_field:
                st.write("Date of Launch")
            with c_data:
                st.write(R_B_ZC_Insur_obj.dt_launch)

            with c_field:
                st.write("Policy Term [years]")
            with c_data:
                st.write(R_B_ZC_Insur_obj.i_policy_term_in_years)
            with c_field:
                st.write("Time Horizon Min [years]")
            with c_data:
                st.write(R_B_ZC_Insur_obj.i_time_horizon_years_min)
            with c_field:
                st.write("Time Horizon Max [years]")
            with c_data:
                st.write(R_B_ZC_Insur_obj.i_time_horizon_years_max)

            with c_field:
                st.write("Return Guaranteed")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_return_guaranteed)

            with c_field:
                st.write("Issuer")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_issuer)
            with c_field:
                st.write("Product or Plan")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_product_plan)
            with c_field:
                st.write("Premium Type")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_premium_type)
            with c_field:
                st.write("Premium Min ["+R_B_ZC_Insur_obj.s_currency+"]")
            with c_data:
                st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_min))
            with c_field:
                st.write("Premium Incr ["+R_B_ZC_Insur_obj.s_currency+"]")
            with c_data:
                st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_step))
            with c_field:
                st.write("Premium Max ["+R_B_ZC_Insur_obj.s_currency+"]")
            with c_data:
                st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_max))
            with c_field:
                st.write("Participation")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_par)
            with c_field:
                st.write("s_eligibility_age_min")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_eligibility_age_min)
            with c_field:
                st.write("s_eligibility_age_max")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_eligibility_age_max)
            with c_field:
                st.write("s_eligibility_residency_SG_Citizen")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_Citizen)
            with c_field:
                st.write("s_eligibility_residency_SG_PR")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_PR)
            with c_field:
                st.write("s_eligibility_residency_SG_EP")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_EP)
            with c_field:
                st.write("s_issuer_contact_number")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_issuer_contact_number)
            with c_field:
                st.write("s_issuer_contact_email")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_issuer_contact_email)


            with c_field:
                st.write("Product Description")
            with c_data:
                st.write(R_B_ZC_Insur_obj.s_product_desc)


    # s_product_benefit_maturity = (
    #     "If the insured survives at the end of the policy term and this policy has not ended,  \n"
    #     "we will pay the guaranteed maturity benefit at the end of the policy term.  \n"
    #     "This policy will end when we make this payment.")
    # s_product_benefit_death = "In the event of death of the life assured, the single premium is returned with interest, using the guaranteed simple interest rate."
    # s_product_benefit_death_accidental = ("In the event of accidental death in the first year of the policy,  \n"
    #                                       "subject to the policyholder being under age 70,  \n"
    #                                       "an additional 10% of the single premium will be payable.")
    # s_product_benefit_TPD = (
    #     "Upon diagnosis of TPD before age 65, the single premium is returned with interest, using the guaranteed simple interest rate.")
    # s_product_benefit_exclusions = ("Death due to suicide within one year from the date of issue of Policy."








    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    s = "R_B_Perp"
    if section == s:
        toc.h1(s)

        c_display, c_text, c_input, c_formula = st.columns([3,2,1,1])
        with c_display:
            st.latex(
                # r'\Huge'
                r'\huge'
                r'\begin{array}{lcl} '
                r'\\\\'
                r'     &&R\_B\_Perp(t,t,\infty) &= &\left(\frac{CF\_per}{r\_per}\right) & & \\'
                r'\\\\'
                r'\end{array}'
            )
        with c_text:
            st.latex(
                r'\begin{array}{lcl} '
                r'n [y] &= &\text{Time Horizon} \\'
                r'       &  &\text{(in years)} \\'
                r'\\'
                r'm [1/y] &= &\text{Coupon Frequency} \\'
                r'       &  &\text{(per year)} \\'
                r'\\'
                r'p [1] &= &\text{Total Num Coupons} \\'
                r'       &  &\text{(periods)} \\'
                r'\\\\'
                r'r [\%] &= &\text{risk-free rate} \\'
                r'       &  &\text{(per year)} \\'
                r'\\'
                r'r\_per [\%] &= &\text{risk-free rate} \\'
                r'       &  &\text{(per period)} \\'
                r'\\\\\\'
                r'c [\%] &= &\text{coupon rate} \\'
                r'       &  &\text{(per year)} \\'
                r'\\'
                r'c\_per [\%] &= &\text{coupon rate} \\'
                r'       &  &\text{(per period)} \\'
                r'\\\\'
                r'N [SGD] &= &\text{Notional} \\'
                r'       &  &\text{(in SGD)} \\'
                r'\\\\'
                r'CF [SGD/y] &= &\text{Cashflow} \\'
                r'         &  &\text{(in SGD per year)} \\'
                r'\\'
                r'CF\_per [SGD/per] &= &\text{Cashflow} \\'
                r'                  &  &\text{(in SGD per period)} \\'
                r'\\\\'
                r'R\_B\_Perp(0,0,\infty) [SGD] &= &\text{Present Value} \\'
                r'                             &  &\text{of Perpetuity} \\'
                r'                             &  &\text{(in SGD)} \\'
                r'\end{array}'
            )
        with c_input:
            try:
                n = float('inf')
                s_n = st.text_input('Input: Time Horizon [year]:',str(n))
                s_m = st.text_input('Input: Coupon Frequency [1/year]:',12)
                m = int(s_m)
                p = n * m
                s_p = st.text_input('Output: Total Num Coupons [period]:',str(p))

                s_r = st.text_input('Input: Risk-Free Rate [%/year]:',4.25)
                r = float(s_r)/100
                r_per = r/m
                s_r_per = st.text_input('Output: Risk-Free Rate [%/period]:',str(r_per * 100))

                s_c = st.text_input('Input: Coupon Rate [%/year]:', 4.25)
                c = float(s_c)/100
                c_per = c/m
                s_c_per = st.text_input('Output: Coupon Rate [%/period]:',str(c_per * 100))

                s_N = st.text_input('Input: Notional [SGD]:', "{:,.2f}".format(1000000))
                N = float(s_N.replace(",",""))

                CF = c * N
                s_CF = st.text_input('Output: CF [SGD/y]:', "{:,.2f}".format(CF))
                CF_per = c/m * N
                s_CF_per = st.text_input('Output: CF_per [SGD/per]:', "{:,.2f}".format(CF_per))

                R_B_Perp = CF_per/r_per
                s_R_B_Perp = st.text_input('Output Present Value of Perpetuity [SGD]:', "{:,.2f}".format(R_B_Perp))

            except Exception as e:
                st.write(e)
                pass
        with c_formula:
            st.write('\n')
            st.write('\n')
            st.write(n)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(m)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(p)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(r)
            st.write('\n')
            st.write('\n')
            st.write(r_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(c)
            st.write('\n')
            st.write('\n')
            st.write(c_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(N)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(CF)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(CF_per)
            st.write('\n')
            st.write('\n')
            st.write(R_B_Perp)

        cols = st.columns(1)
        with cols[0]:
            st.latex(
                r'\begin{align*} '
                r'\\\\'
                r'\\\\'
                r'\\\\'
                r'  &&R\_B\_Perp(0,0,\infty) &= & & &\sum_{i=1}^{\inf}\ &D(0,i)                         &\cdot &CF\_per  \\'
                r'  &&R\_B\_Perp(0,0,\infty) &= & & &\sum_{i=1}^{\inf}\ &\left(\frac{1}{1+r\_per}\right)^{i} &\cdot &CF\_per  '
                r'                                                                                      &&\ \ \ \Longrightarrow\text{ [1]} \\'
                r'\\\\'
                r'  \left(\frac{1}{1+r\_per}\right) '
                r'  &\cdot '
                r'   &R\_B\_Perp(0,0,\infty) &= &\left(\frac{1}{1+r\_per}\right) & \cdot'
                r'                                  &\sum_{i=1}^{\inf}\ &\left(\frac{1}{1+r\_per}\right)^{i}   &\cdot &CF\_per'
                r'                                                                                        &,\text{where we discounted both sides by 1 period} \\'
                r'  \left(\frac{1}{1+r\_per}\right) '
                r'  &\cdot '
                r'   &R\_B\_Perp(0,0,\infty) &= & & &\sum_{i=2}^{\inf}\ &\left(\frac{1}{1+r\_per}\right)^{i} &\cdot &CF\_per '
                r'                                                                                          &,\text{where infinity allows us to start index at 2 instead of 1}'
                r'                                                                                          &\ \ \ \Longrightarrow\text{ [2]} \\'

                r'\\\\'
                r'  +\left(1\right) '
                r'  &\cdot '
                r'   &R\_B\_Perp(0,0,\infty) &= &+ & &\sum_{i=1}^{\inf}\ &\left(\frac{1}{1+r\_per}\right)^{i} &\cdot &CF\_per '
                r'                                                                                          &'
                r'                                                                                          &\ \ \ \Longrightarrow\text{[1] - [2]} \\'
                r'  -\left(\frac{1}{1+r\_per}\right) '
                r'  &\cdot '
                r'   &R\_B\_Perp(0,0,\infty) &  &- & &\sum_{i=2}^{\inf}\ &\left(\frac{1}{1+r\_per}\right)^{i} &\cdot &CF\_per & &  \\'

                r'\\\\'
                r'  \left[1-\left(\frac{1}{1+r\_per}\right)\right] &\cdot &R\_B\_Perp(0,0,\infty) '
                r'     &= & & & &\left(\frac{1}{1+r\_per}\right) &\cdot &CF\_per & & \\'
                r'  \left[(1+r\_per)-1\right]                      &\cdot &R\_B\_Perp(0,0,\infty) '
                r'     &= & & & &                           &      &CF\_per & & \\'
                r'  r                                         &\cdot &R\_B\_Perp(0,0,\infty) '
                r'     &= & & & &                           &      &CF\_per & & \\'
                r'                                            &      &R\_B\_Perp(0,0,\infty) '
                r'     &= & & & &                           &      &\left(\frac{CF\_per}{r\_per}\right) & & \\'

                r'\\\\'
                r'     &&R\_B\_Perp(0,0,\infty) &= &\left(\frac{CF\_per}{r\_per}\right) & & \\'
                r'\\\\'
                r'     &&R\_B\_Perp(t,t,\infty) &= &\left(\frac{CF\_per}{r\_per}\right) & & \\'
                r'\\\\'
                r'\\\\'
                r'\end{align*}'
            )


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    s = "R_B_Perp_Annuity"
    if section == s:
        toc.h1(s)

        c_display, c_text, c_input, c_formula = st.columns([3,2,1,1])
        with c_display:
            st.latex(
                r'\begin{align*} '
                r'\\\\'
                r'  &&R\_B\_Annuity(0,t,T) &= & &  & & &R\_B\_Perp(0,t,\infty) \\'
                r'  &&                     &  & &  & & &R\_B\_Perp(0,T,\infty) \\'
                r'\\\\'
                r'  &&R\_B\_Annuity(0,t,T) &= & &+ &D(0,t) &\cdot &R\_B\_Perp(t,t,\infty) \\'
                r'  &&                     &  & &- &D(0,T) &\cdot &R\_B\_Perp(T,T,\infty) \\'
                r'\\\\'
                r'  &&R\_B\_Annuity(0,t,T) &= & &+ &\left(\frac{1}{1+r\_per}\right)^{tm} &\cdot &\left(\frac{CF\_per}{r\_per}\right) \\'
                r'  &&                     &  & &- &\left(\frac{1}{1+r\_per}\right)^{Tm} &\cdot &\left(\frac{CF\_per}{r\_per}\right) \\'
                r'\\\\'
                r'  &&R\_B\_Annuity(0,t,T) &= &\Huge{[} &+ &\left(\frac{1}{1+r\_per}\right)^{tm}'
                r'                             \Huge{]} &\cdot &\left(\frac{CF\_per}{r\_per}\right) \\'
                r'  &&                     &  &\Huge{[} &- &\left(\frac{1}{1+r\_per}\right)^{Tm}\Huge{]} & & \\'
                r'\\\\'
                r'\end{align*}'
            )

        with c_text:
            st.latex(
                r'\begin{array}{lcl} '
                r'dt\_Price &= &\text{Date Pricing} \\ '
                r'[dt]     &  &                    \\ '
                r'\\\\\\'
                r'm     &= &\text{Coupon Frequency} \\ '
                r'[1/y] &  &\text{(per year)} \\ '
                r'\\\\\\'
                r'T   &= &\text{Time Horizon} \\ '
                r'[y] &  &\text{(in years)} \\ '
                r'\\'
                r'Tm  &= &\text{Time Horizon Periods} \\ '
                r'[1] &  &\text{(in periods)} \\ '
                r'\\'
                r'dt\_End &= &\text{Date End} \\ '
                r'[dt]     &  &                    \\ '
                r'\\\\\\\\\\'
                r't   &= &\text{Time} \\ '
                r'[y] &  &\text{to Forward Start} \\ '
                r'    &  &\text{(in years)} \\ '
                r'\\'
                r'tm  &= &\text{Time Periods} \\ '
                r'[1] &  &\text{to Forward Start} \\ '
                r'    &  &\text{(in periods)} \\ '
                r'\\'
                r'dt\_Start &= &\text{Date Start} \\ '

                r'\\\\\\\\'
                r'r    &= &\text{risk-free rate} \\ '
                r'[\%] &  &\text{(per year)} \\ '
                r'\\'
                r'r\_per &= &\text{risk-free rate} \\ '
                r'[\%]   &  &\text{(per period)} \\ '
                r'\\\\\\'
                r'c    &= &\text{coupon rate} \\ '
                r'[\%] &  &\text{(per year)} \\ '
                r'\\'
                r'c\_per &= &\text{coupon rate} \\ '
                r'[\%]   &  &\text{(per period)} \\ '
                r'\\\\\\\\'
                r'N     &= &\text{Notional} \\ '
                r'[SGD] &  &\text{(in SGD)} \\ '
                r'\\\\\\'
                r'CF      &= &\text{Cashflow} \\ '
                r'[SGD/y] &  &\text{(in SGD per year)} \\ '
                r'\\'
                r'CF\_per   &= &\text{Cashflow} \\ '
                r'[SGD/per] &  &\text{(in SGD per period)} \\ '
                r'\\\\\\\\'
                r'R\_B\_Annuity(0,t,T) &= &\text{Present Value} \\ '
                r'[SGD]                &  &\text{of Annuity} \\ '
                r'                     &  &\text{(in SGD)} \\ '
                r'\end{array}'
            )
        with c_input:
            try:
                dt_Now = dt.datetime.now()
                next_month = dt_Now + relativedelta(months=1)
                start_of_next_month = dt.datetime(next_month.year, next_month.month, 1)
                dt_Pricing = start_of_next_month
                s_dt_Pricing = st.text_input('Input: Date Pricing [date]:', dt_Pricing)

                st.text('\n')
                st.text('\n')
                st.text('\n')
                s_m = st.text_input('Input: Coupon Frequency [1/year]:',12)
                m = int(s_m)

                st.text('\n')
                st.text('\n')
                T = 20
                s_T = st.text_input('Input: Time Horizon [year]:',str(T))
                Tm = T * m
                s_Tm = st.text_input('Output: Time Horizon Periods [period]:',str(Tm))
                from dateutil.relativedelta import relativedelta
                dt_End = dt_Pricing + relativedelta(months=Tm)
                s_dt_End = st.text_input('Output: Date End [date]:', dt_End)

                st.text('\n')
                st.text('\n')
                st.text('\n')
                t = 10
                s_t = st.text_input('Input: Time to Forward Start [year]:',str(t))
                tm = t * m
                s_tm = st.text_input('Output: Time Periods to Forward Start [period]:',str(tm))
                from dateutil.relativedelta import relativedelta
                dt_Start = dt_Pricing + relativedelta(months=tm)
                s_dt_Start = st.text_input('Output: Date Start [date]:', dt_Start)

                st.text('\n')
                st.text('\n')
                s_r = st.text_input('Input: Risk-Free Rate [%/year]:',4.25)
                r = float(s_r)/100
                r_per = r/m
                s_r_per = st.text_input('Output: Risk-Free Rate [%/period]:',str(r_per * 100))

                st.text('\n')
                st.text('\n')
                s_c = st.text_input('Input: Coupon Rate [%/year]:', 4.25)
                c = float(s_c)/100
                c_per = c/m
                s_c_per = st.text_input('Output: Coupon Rate [%/period]:',str(c_per * 100))

                st.text('\n')
                st.text('\n')
                s_N = st.text_input('Input: Notional [SGD]:', "{:,.2f}".format(1000000))
                N = float(s_N.replace(",",""))

                st.text('\n')
                st.text('\n')
                CF = c * N
                s_CF = st.text_input('Output: CF [SGD/y]:', "{:,.2f}".format(CF))
                CF_per = c/m * N
                s_CF_per = st.text_input('Output: CF_per [SGD/per]:', "{:,.2f}".format(CF_per))

                st.text('\n')
                st.text('\n')
                R_B_Perp_Annuity_cont = ( exp(-r_per * tm)  - exp(-r_per * Tm)  ) * (CF_per/r_per)
                R_B_Perp_Annuity_disc = ( (1/(1+r_per))**tm - (1/(1+r_per))**Tm ) * (CF_per/r_per)
                # R_B_Perp_Annuity = R_B_Perp_Annuity_cont
                R_B_Perp_Annuity = R_B_Perp_Annuity_disc
                s_R_B_Perp_Annuity = st.text_input('Output: Present Value of Annuity [SGD]:', "{:,.2f}".format(R_B_Perp_Annuity))

            except Exception as e:
                st.write(e)
                pass

        with c_formula:

            st.write('\n')
            st.write('\n')
            st.write(dt_Pricing)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(m)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(T)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(Tm)

            st.write('\n')
            st.write('\n')
            st.write(dt_End)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(t)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(tm)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(dt_Start)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(r)

            st.write('\n')
            st.write('\n')
            st.write(r_per)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(c)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(c_per)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(N)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(CF)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(CF_per)
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write('\n')
            st.write(R_B_Perp_Annuity)
            st.write('\n')

        with c_display:

            dt_range = pd.date_range(start=dt_Pricing, end=dt_End, freq='MS')
            df = pd.DataFrame(index=dt_range)
            df.loc[df.index == dt_Pricing,'DateTime_Name'] = 'dt_Pricing'
            df.loc[df.index == dt_Start,'DateTime_Name'] = 'dt_Start'
            df.loc[df.index == dt_End,'DateTime_Name'] = 'dt_End'
            df['i'] = range(len(df))
            df['DF_i'] = (1/(1+r_per)) ** df['i']

            df.index.name = 'DateTime'
            df['CF_SGD'] = 0
            df.loc[(df.i >= tm+1) & (df.i <= Tm), 'CF_SGD'] = CF_per   # dt_Start + 1 = first coupon
            df['PVCF_SGD'] = df['DF_i'] * df['CF_SGD']

            R_B_Perp_Annuity_disc_from_df = df['PVCF_SGD'].sum()
            df['CF_SGD_Priced'] = df['CF_SGD'].copy()
            df.loc[(df.i == 0), 'CF_SGD_Priced'] = - R_B_Perp_Annuity_disc_from_df   # dt_Pricing payment negative cashflow

            # Plot a bar graph
            fig = plt.figure(figsize=(20, 10))
            plt.bar(df.index, df['CF_SGD_Priced'], width=8, color=df['CF_SGD_Priced'].apply(lambda x: 'green' if x > 0 else 'red'))
            plt.title('Cashflows [SGD] Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cashflow [SGD]]')
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.patch.set_facecolor('grey')
            st.pyplot(plt)

            # Data
            st.write(df.head(3))
            st.write(df[df['i'].isin([tm-1,tm,tm+1])])
            st.write(df.tail(3))

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    s = "R_B_Perp_Annuity_CPF_LIFE"
    if section == s:
        toc.h1(s)

    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    s = "ECXRK_Opt: Simple"
    if section == s:
        toc.h1(s)
        # C:\Dropbox\_\work\___lib_py\PY_Fin_Deriv.py
        import ___lib_py.PY_Fin_Deriv as PY_Fin_Deriv
        st.header("Simple: Price, Risk")
        col_in, col_out_V, col_out_V_ddX, col_out_V_ddsigma, col_out_V_ddXdsigma, col_out_V_ddt, col_out_V_ddXsRet = st.columns([1.5,1,1,1,1,1,1])
        with col_in:
            st.subheader("Inputs")
            with st.echo():
                i_Model_N0_LogN1 = 1
                b_Disc = 1
                s_Exer_Euro_Amer = "E"
                s_Prod_x_p_c = "P"
                s_Payoff_Digi_Anal = "Anal"
                d_TimeToMat = 1.00
                d_RiskFreeRate = 0.06
                d_Strike = 100.0
                d_Under = 100.0
                d_UnderRetMean = 0.06
                d_UnderRetStdDev = 0.20
        with col_out_V:
            st.subheader("Output: V")
            def st_write_result(s_Req):
                d_Res = PY_Fin_Deriv.PY_Fin_Deriv_BS_Price_Risk(
                    s_Req, i_Model_N0_LogN1, b_Disc, s_Exer_Euro_Amer, s_Prod_x_p_c, s_Payoff_Digi_Anal,
                    d_TimeToMat, d_RiskFreeRate, d_Strike, d_Under, d_UnderRetMean, d_UnderRetStdDev)
                st.write(s_Req)
                st.write(d_Res)
            st_write_result("Price")
        with col_out_V_ddX:
            st.subheader("Output: V: d dX")
            st_write_result("Delta")
            st_write_result("Gamma")
            st_write_result("Speed")
        with col_out_V_ddsigma:
            st.subheader("Output: V: d dsigma")
            st_write_result("Vega")
            st_write_result("Volga")
        with col_out_V_ddXdsigma:
            st.subheader("Output: V: d dX dsigma")
            st_write_result("Vanna")
        with col_out_V_ddt:
            st.subheader("Output: V: d dt")
            st_write_result("Theta")
        with col_out_V_ddXsRet:
            st.subheader("Output: V: d dXsRet")
            st_write_result("ppmu")
            st_write_result("ppr")
            st_write_result("Rho")

        st.header("Simple: Implied Volatility")
        col_in, col_in_V, col_out_sigma_implied, col_04, col_05, col_06, col_07 = st.columns([1.5,1.5,1,1,1,1,0.5])
        with col_in:
            st.subheader("Inputs")
            with st.echo():
                i_Model_N0_LogN1 = 1
                b_Disc = 1
                s_Exer_Euro_Amer = "E"
                s_Prod_x_p_c = "P"
                s_Payoff_Digi_Anal = "Anal"
                d_TimeToMat = 1.00
                d_RiskFreeRate = 0.06
                d_Strike = 100.0
                d_Under = 100.0
                d_UnderRetMean = 0.06
                # d_UnderRetStdDev = 0.20 # to be found
        with col_in_V:
            st.subheader("Input: V")
            with st.echo():
                d_OptPriceObsMkt = 5.16600251105086
                v_UnderRetStdDev_Init = 0.5
                v_Tolerance = 0.0001
                v_MaxIter = 10000

        with col_out_sigma_implied:
            st.subheader("Output: sigma Implied")
            d_Res = PY_Fin_Deriv.PY_Fin_Deriv_BS_Implied_Volatility( i_Model_N0_LogN1, s_Exer_Euro_Amer, s_Prod_x_p_c,
                        s_Payoff_Digi_Anal,
                        d_TimeToMat, d_RiskFreeRate,
                        d_Strike, d_Under,
                        d_UnderRetMean,
                        d_OptPriceObsMkt,
                        v_UnderRetStdDev_Init,
                        v_Tolerance,
                        v_MaxIter
                        )
            st.write("Implied Volatility")
            st.write(d_Res)


    s = "ECXRK_Opt: Matrix"
    if section == s:
        toc.h1(s)
        with st.echo():
            # C:\Dropbox\_\work\___lib_py\PY_Fin_Deriv.py
            import ___lib_py.PY_Fin_Deriv as PY_Fin_Deriv


if sidebar_selectbox_mode == "Risk Parity: REC: Data":

    # Initialize date parameters
    dt_start = dt.datetime(2014, 1, 1)
    dt_end = dt.datetime.now()

    # Read a .csv list of Bloomberg Tickers into
    # Dataframe of Tickers (Bloomberg) with the Country and Asset Class Suffixes
    df_csv_tickers_bbg = pd.read_csv('.\___lib_py_yahoo\ydh_in___ECXRK.csv', header=None)
    df_csv_tickers_bbg.columns = ['Ticker_Bbg']
    st.write(df_csv_tickers_bbg)

    # Dataframe of Tickers without the Country and Asset Class Suffixes
    df_csv_tickers = df_csv_tickers_bbg.copy(deep=True)
    df_csv_tickers.columns = ['Ticker']  # Method 1 of 2
    # df_csv_tickers.rename(columns={'Ticker_Bbg': 'Ticker'}, inplace=True)     # Method 2 of 2
    df_csv_tickers = df_csv_tickers.astype(str).replace("Equity", "", regex=True)
    df_csv_tickers = df_csv_tickers.astype(str).replace(" US ", " ", regex=True)
    df_csv_tickers = df_csv_tickers.astype(str).replace(" ", "", regex=True)
    st.write(df_csv_tickers)

    # Request data from yahoo finance
    # df_USD_SGD = ydh.yhapi_ydh('SGD=x', dt_start, dt_end)
    dfx_R_USD = ydh.yhapi_ydh('TLT', dt_start, dt_end, "M")
    dfx_E_USD = ydh.yhapi_ydh('SPY', dt_start, dt_end, "M")
    dfx_C_USD = ydh.yhapi_ydh('GLD', dt_start, dt_end, "M")
    # st.write(df_E_USD)
    # st.write(df_C_USD)
    # st.write(df_R_USD)

    # dfx = dataframe for price x
    dfx = dfx_R_USD
    dfx = dfx.join(dfx_E_USD)
    dfx = dfx.join(dfx_C_USD)

    # Rename the index
    dfx.index.names = ['Date']

    # Export
    dfx_R_USD.to_csv('./_prd/_data_yahoo/dfx_R_USD.csv', index=True)
    dfx_E_USD.to_csv('./_prd/_data_yahoo/dfx_E_USD.csv', index=True)
    dfx_C_USD.to_csv('./_prd/_data_yahoo/dfx_C_USD.csv', index=True)
    dfx.to_csv('./_prd/_data_yahoo/dfx.csv', index=True)

    st.write(dfx)

if sidebar_selectbox_mode == "Risk Parity: REC: Analysis":

    # Initialize date parameters
    dt_start = dt.datetime(2014, 1, 1)
    dt_end = dt.datetime.now()

    # Import
    dfx = pd.read_csv('./_prd/_data_yahoo/dfx.csv', index_col=0)  # assuming the index is in the first column
    dfx.index = pd.to_datetime(dfx.index)

    # Prices
    x_fig, x_ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
    plt.plot(dfx.TLT, marker='', linestyle='-', color='red', label='R_TLT = iShares 20 Plus Year Treasury Bond ETF')
    plt.plot(dfx.SPY, marker='', linestyle='-', color='orange', label='E_SPY = SPDR S&P 500 ETF Trust')
    plt.plot(dfx.GLD, marker='', linestyle='-', color='yellow', label='C_GLD = SPDR Gold Trust')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Asset Class Prices over Last 10 Years')
    plt.legend()
    x_ax.set_facecolor('#333333')  # Set the background color to dark gray
    plt.show()
    st.pyplot(x_fig)  # streamlit plot: Add this at end

    # Prices, Normalized
    dfxn = dfx.divide(dfx.iloc[0]) * 100
    xn_fig, xn_ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
    plt.plot(dfxn.TLT, marker='', linestyle='-', color='red', label='R_TLT = iShares 20 Plus Year Treasury Bond ETF')
    plt.plot(dfxn.SPY, marker='', linestyle='-', color='orange', label='E_SPY = SPDR S&P 500 ETF Trust')
    plt.plot(dfxn.GLD, marker='', linestyle='-', color='yellow', label='C_GLD = SPDR Gold Trust')
    plt.xlabel('Date')
    plt.ylabel('Price Normalized')
    plt.title('Asset Class Prices Normalized over Last 10 Years')
    plt.legend()
    xn_ax.set_facecolor('#333333')  # Set the background color to dark gray
    plt.show()
    st.pyplot(xn_fig)  # streamlit plot: Add this at end

    # Returns Monthly
    dfxr = dfx.pct_change()
    xr_fig, xr_ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
    plt.plot(dfxr.TLT, marker='', linestyle='-', color='red', label='R_TLT = iShares 20 Plus Year Treasury Bond ETF')
    plt.plot(dfxr.SPY, marker='', linestyle='-', color='orange', label='E_SPY = SPDR S&P 500 ETF Trust')
    plt.plot(dfxr.GLD, marker='', linestyle='-', color='yellow', label='C_GLD = SPDR Gold Trust')
    plt.xlabel('Date')
    plt.ylabel('Monthly Returns')
    plt.title('Asset Class Monthly Returns over Last 10 Years')
    plt.legend()
    xr_ax.set_facecolor('#333333')  # Set the background color to dark gray
    plt.show()
    st.pyplot(xr_fig)  # streamlit plot: Add this at end

    # Returns Monthly, Histogram
    # Plot histograms for each time series
    xrh_fig, xrh_ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
    colors = ['red', 'orange', 'yellow']  # Specify colors for each series
    for col, color in zip(dfxr.columns[0:], colors):
        # Plot histogram
        xrh_ax.hist(dfxr[col], bins=20, alpha=0.6, edgecolor='black', color=color, label=col)

    plt.title('Overlayed Distribution of Monthly Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    # xrh_ax.set_facecolor('#333333')  # Set the background color to dark gray
    plt.show()
    st.pyplot(xrh_fig)  # streamlit plot: Add this at end

    # Returns Monthly - Statistics
    dfxr_stats = pd.DataFrame()
    dfxr_stats['Mean'] = dfxr.mean() * 12
    dfxr_stats['StdDev'] = dfxr.std() * (12**0.5)

    # Returns Monthly - Correlation
    dfxr_corr = dfxr.corr()

    # Display
    col_stats, col_correl = st.columns(2)
    with col_stats:
        st.write('dfxr_stats')
        st.write(dfxr_stats)
        st.write(dfxr_stats.applymap(lambda x: f'{x * 100:.2f}%'))
    with col_correl:
        st.write('dfxr_corr')
        st.write(dfxr_corr)
        st.write(dfxr_corr.applymap(lambda x: f'{x*100:.2f}%'))

    # index asset class
    asset_classes = ['R', 'E', 'C']

    # df_w = Weights Vector (n-by-1)
    w_REC_BaseCase = {
        'Weights': [0.5, 0.3, 0.2]
    }
    w_REC_RiskParity = {
        'Weights': [0.325, 0.345, 0.33]
    }
    df_w0 = pd.DataFrame(w_REC_BaseCase, index=asset_classes)
    df_w1 = pd.DataFrame(w_REC_RiskParity, index=asset_classes)

    selected_case = st.sidebar.radio("Select a case:", ["Base-Case", "Risk-Parity"])
    if selected_case == "Base-Case":
        df_w = df_w0
    elif selected_case == "Risk-Parity":
        df_w = df_w1

    # df_sigma = Standard Deviation Matrix (n-by-n)
    sigma = {
        'R': [dfxr_stats['StdDev']['TLT'], 0, 0],
        'E': [0, dfxr_stats['StdDev']['SPY'], 0],
        'C': [0, 0, dfxr_stats['StdDev']['GLD']]
    }
    df_sigma = pd.DataFrame(sigma, index=['R', 'E', 'C'])
    df_sigma.columns = asset_classes

    # df_rho = Correlation Matrix (n-by-n)
    df_rho = dfxr_corr.copy()
    df_rho.index = asset_classes
    df_rho.columns = asset_classes

    col_w, col_sigma, col_rho, col_4 = st.columns(4)
    with col_w:
        st.write('df_w')
        st.write(df_w)
    with col_sigma:
        st.write('df_sigma')
        st.write(df_sigma)
    with col_rho:
        st.write('df_rho')
        st.write(df_rho)

    # Variance Decomposition by Asset Class
    df_covar = df_sigma.dot(df_rho).dot(df_sigma)
    df_covar_R = df_covar.copy()
    df_covar_R[['E','C']] = 0
    df_covar_E = df_covar.copy()
    df_covar_E[['R','C']] = 0
    df_covar_C = df_covar.copy()
    df_covar_C[['R','E']] = 0

    col_REC, col_R, col_E, col_C = st.columns(4)
    with col_REC:
        st.write('df_covar')
        st.write(df_covar)
    with col_R:
        st.write('df_covar_R')
        st.write(df_covar_R)
    with col_E:
        st.write('df_covar_E')
        st.write(df_covar_E)
    with col_C:
        st.write('df_covar_C')
        st.write(df_covar_C)

    # df_port_var = Portfolio Variance by Asset Class
    port_var   = float(df_w.transpose().dot(df_covar).dot(df_w).iloc[0,0])
    port_var_R = float(df_w.transpose().dot(df_covar_R).dot(df_w).iloc[0,0])
    port_var_E = float(df_w.transpose().dot(df_covar_E).dot(df_w).iloc[0,0])
    port_var_C = float(df_w.transpose().dot(df_covar_C).dot(df_w).iloc[0,0])
    df_port_var = pd.DataFrame(index=asset_classes, columns=['Variance'])
    df_port_var['Variance']['R'] = port_var_R
    df_port_var['Variance']['E'] = port_var_E
    df_port_var['Variance']['C'] = port_var_C
    st.write('port_var')
    st.write(port_var)

    # df_port_rc = Portfolio Risk Contribution
    df_port_rc = pd.DataFrame(index=asset_classes, columns=['RiskContrib'])
    df_port_rc['RiskContrib']['R'] = port_var_R / port_var
    df_port_rc['RiskContrib']['E'] = port_var_E / port_var
    df_port_rc['RiskContrib']['C'] = port_var_C / port_var

    # df_port_std = Portfolio Risk
    df_port_std = pd.DataFrame(index=asset_classes, columns=['StdDev'])
    df_port_std['StdDev'] = df_port_var['Variance'] ** 0.5

    df_stats = df_w.copy()
    df_stats = df_stats.join(df_port_var)
    df_stats = df_stats.join(df_port_rc)
    df_stats = df_stats.join(df_port_std)
    st.write('df_stats')
    st.write(df_stats)

    df_stats_total = pd.DataFrame(index=['Portfolio'], columns=df_stats.columns)
    df_stats_total['Weights'] = df_stats['Weights'].sum()
    df_stats_total['Variance'] = df_stats['Variance'].sum()
    df_stats_total['RiskContrib'] = df_stats['RiskContrib'].sum()
    df_stats_total['StdDev'] = (df_stats['StdDev']**2).sum()**0.5
    st.write(df_stats_total)

    # Display in %
    st.write('df_stats_display')
    st.write(df_stats.applymap(lambda x: f'{x * 100:.2f}%'))
    st.write(df_stats_total.applymap(lambda x: f'{x * 100:.2f}%'))

    # Back-Test
    df_r = dfxr[['TLT', 'SPY', 'GLD']].copy()
    df_r.columns = ['R','E','C']
    st.write(df_r)

    # dfpr = Portfolio Returns
    dfpr0 = df_r @ df_w0[['Weights']]
    dfpr1 = df_r @ df_w1[['Weights']]
    dfpr0.columns = ['PortfolioReturns']
    dfpr1.columns = ['PortfolioReturns']
    st.write(f"{df_r.shape} * {df_w0[['Weights']].shape} = {dfpr0.shape}")
    st.write(f"{df_r.shape} * {df_w1[['Weights']].shape} = {dfpr1.shape}")

    dfpr0['PortfolioIndex_BaseCase'] = (1 + dfpr0['PortfolioReturns']).cumprod() * 100
    dfpr0['PortfolioIndex_BaseCase'][0] = 100
    dfpr1['PortfolioIndex_RiskParity'] = (1 + dfpr1['PortfolioReturns']).cumprod() * 100
    dfpr1['PortfolioIndex_RiskParity'][0] = 100

    dfpi = dfpr0[['PortfolioIndex_BaseCase']].copy()
    dfpi = dfpi.join(dfpr1[['PortfolioIndex_RiskParity']])
    dfpir = dfpi.pct_change()
    dfpir_mu = pd.DataFrame(dfpir.mean()) * 12
    dfpir_mu.columns = ['Expected Return']
    dfpir_sigma = pd.DataFrame(dfpir.std()) * (12**0.5)
    dfpir_sigma.columns = ['Volatilty']
    dfpir_inforatio = pd.DataFrame(dfpir_mu['Expected Return'] / dfpir_sigma['Volatilty'])
    dfpir_inforatio.columns = ['InfoRatio']

    dfpir_stats = dfpir_mu.copy()
    dfpir_stats = dfpir_stats.join(dfpir_sigma)
    dfpir_stats = dfpir_stats.join(dfpir_inforatio)
    st.write(dfpir_stats)

    # Portfolio Index
    pi_fig, pi_ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
    plt.plot(dfpi.PortfolioIndex_BaseCase, marker='', linestyle='-', color='black', label='Base-Case Portfolio')
    plt.plot(dfpi.PortfolioIndex_RiskParity, marker='', linestyle='-', color='green', label='Risk-Parity Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Index')
    plt.title('Portfolio Index Back-Test')
    plt.legend()
    pi_ax.set_facecolor('#333333')  # Set the background color to dark gray
    plt.show()
    st.pyplot(pi_fig)  # streamlit plot: Add this at end


if sidebar_selectbox_mode == "ECXDRKP":

    sidebar_checkbox_secutype_E_Idx_US = st.sidebar.checkbox("E_Idx_US")
    if sidebar_checkbox_secutype_E_Idx_US:

        # Initialize date parameters
        dt_start = dt.datetime(2023, 1, 1)
        dt_end = dt.datetime.now()

        # Read a .csv list of Bloomberg Tickers into
        # Dataframe of Tickers (Bloomberg) with the Country and Asset Class Suffixes
        # E_Idx_US = SPX Index = S&P 500, PCA applied to narrow down to 30 Tickers
        df_csv_tickers_bbg = pd.read_csv('.\___lib_py_yahoo\ydh_in___E_Idx_US_SPY_505_PCA_030.csv', header=None)
        df_csv_tickers_bbg.columns = ['Ticker_Bbg']
        print(df_csv_tickers_bbg)

        # Dataframe of Tickers without the Country and Asset Class Suffixes
        df_csv_tickers = df_csv_tickers_bbg.copy(deep=True)
        df_csv_tickers.columns = ['Ticker']  # Method 1 of 2
        # df_csv_tickers.rename(columns={'Ticker_Bbg': 'Ticker'}, inplace=True)     # Method 2 of 2
        df_csv_tickers = df_csv_tickers.astype(str).replace("Equity", "", regex=True)
        df_csv_tickers = df_csv_tickers.astype(str).replace(" US ", " ", regex=True)
        df_csv_tickers = df_csv_tickers.astype(str).replace(" ", "", regex=True)
        print(df_csv_tickers)

        col_sidebar, col_content = st.columns([1,4])
        with col_sidebar:
            toggle_show_tickers = st.toggle('Show Tickers')
            if toggle_show_tickers:
                # print in streamlit in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.write(df_csv_tickers_bbg)
                with col2:
                    st.write(df_csv_tickers)

            # checkbox
            cb_SPY = st.checkbox('SPY', value=True)
            cb_QQQ = st.checkbox('QQQ', value=True)
            cb_NVDA = st.checkbox('NVDA')
            cb_TSLA = st.checkbox('TSLA')

            toggle_req_data = st.toggle('Request Data')
            if toggle_req_data:
                with st.spinner():
                    # Request data from finnhub
                    # df_SPY_USD = fdh.fhapi_fdh('SPY', dt_start, dt_end)
                    # df_QQQ_USD = fdh.fhapi_fdh('QQQ', dt_start, dt_end)

                    # Request data from yahoo finance
                    # df_USD_SGD = ydh.yhapi_ydh('SGD=x', dt_start, dt_end)
                    df_SPY_USD = ydh.yhapi_ydh('SPY', dt_start, dt_end)
                    df_QQQ_USD = ydh.yhapi_ydh('QQQ', dt_start, dt_end)

                    df_NVDA_USD = ydh.yhapi_ydh('NVDA', dt_start, dt_end)
                    df_WFC_USD = ydh.yhapi_ydh('WFC', dt_start, dt_end)
                    df_TSLA_USD = ydh.yhapi_ydh('TSLA', dt_start, dt_end)
                    df_SCHW_USD = ydh.yhapi_ydh('SCHW', dt_start, dt_end)
                    df_LCRX_USD = ydh.yhapi_ydh('LRCX', dt_start, dt_end)
                    df_ISRG_USD = ydh.yhapi_ydh('ISRG', dt_start, dt_end)
                    df_USB_USD = ydh.yhapi_ydh('USB', dt_start, dt_end)
                    df_CDNS_USD = ydh.yhapi_ydh('CDNS', dt_start, dt_end)
                    df_PYPL_USD = ydh.yhapi_ydh('PYPL', dt_start, dt_end)
                    df_TFC_USD = ydh.yhapi_ydh('TFC', dt_start, dt_end)
                    df_ANET_USD = ydh.yhapi_ydh('ANET', dt_start, dt_end)
                    df_DXCM_USD = ydh.yhapi_ydh('DXCM', dt_start, dt_end)
                    df_HAL_USD = ydh.yhapi_ydh('HAL', dt_start, dt_end)
                    df_FTNT_USD = ydh.yhapi_ydh('FTNT', dt_start, dt_end)
                    df_WBD_USD = ydh.yhapi_ydh('WBD', dt_start, dt_end)
                    df_DFS_USD = ydh.yhapi_ydh('DFS', dt_start, dt_end)
                    df_ILMN_USD = ydh.yhapi_ydh('ILMN', dt_start, dt_end)
                    df_MPWR_USD = ydh.yhapi_ydh('MPWR', dt_start, dt_end)
                    df_FITB_USD = ydh.yhapi_ydh('FITB', dt_start, dt_end)
                    df_ENPH_USD = ydh.yhapi_ydh('ENPH', dt_start, dt_end)
                    df_CCL_USD = ydh.yhapi_ydh('CCL', dt_start, dt_end)
                    df_CFG_USD = ydh.yhapi_ydh('CFG', dt_start, dt_end)
                    df_TER_USD = ydh.yhapi_ydh('TER', dt_start, dt_end)
                    df_STLD_USD = ydh.yhapi_ydh('STLD', dt_start, dt_end)
                    df_KEY_USD = ydh.yhapi_ydh('KEY', dt_start, dt_end)
                    df_EPAM_USD = ydh.yhapi_ydh('EPAM', dt_start, dt_end)
                    df_APA_USD = ydh.yhapi_ydh('APA', dt_start, dt_end)
                    df_ZION_USD = ydh.yhapi_ydh('ZION', dt_start, dt_end)
                    df_GNRC_USD = ydh.yhapi_ydh('GNRC', dt_start, dt_end)
                    df_CTLT_USD = ydh.yhapi_ydh('CTLT', dt_start, dt_end)

        with col_content:

            if toggle_req_data:

                # Merge into 1 dataframe
                df = df_SPY_USD
                if cb_QQQ:
                    df = df.join(df_QQQ_USD)
                if cb_NVDA:
                    df = df.join(df_NVDA_USD)
                if cb_TSLA:
                    df = df.join(df_TSLA_USD)

                # Rename the index
                df.index.names = ['Date']

                # Add Columns of Ticker in SGD
                # df['SPY_SGD'] = df['SPY'] * df['SGD=x']

                x1 = 'SPY'
                x2 = 'QQQ'

                # Normalization: Standard
                # df[x1+'_norm'] = (df[x1]-df[x1].mean()) / df[x1].std()
                # df[x2+'_norm'] = (df[x2]-df[x2].mean()) / df[x2].std()

                # Normalization: Max-Min
                df[x1+'_norm'] = (df[x1]-df[x1].mean()) / (df[x1].max()-df[x1].min())
                df[x2+'_norm'] = (df[x2]-df[x2].mean()) / (df[x2].max()-df[x2].min())
                df[x2+'_norm'+'-'+x1+'_norm'] = df[x2+'_norm']-df[x1+'_norm']

                # Output the dataframe to csv
                df.to_csv('ydh_out.csv', encoding='utf-8')
                ydh.print_df(df)
                st.write(df)

                # Graph 2 Curves on 2 Axes
                graph_pair(x1,x2)
                graph_pair(x1,x2,True)
                graph_pair(x2+'_norm'+'-'+x1+'_norm',x2+'_norm'+'-'+x1+'_norm')

                # MT4 Data
                s1 = "SPX500"
                s2 = "NDAQ100"

                df_csv_MT4_Ticker_1_M1 = pd.read_csv(".\___lib_api_mt4\_data\\" + s1 + "1.csv", header=None)
                df_csv_MT4_Ticker_1_M1.columns = ['Date','Time',s1+'_O',s1+'_H',s1+'_L',s1+'_C',s1+'_V']
                df_csv_MT4_Ticker_1_M1['DateTime'] = df_csv_MT4_Ticker_1_M1['Date'] + ' ' + df_csv_MT4_Ticker_1_M1['Time']
                col_order_1_new = ['DateTime','Date','Time',s1+'_O',s1+'_H',s1+'_L',s1+'_C',s1+'_V']
                df_csv_MT4_Ticker_1_M1 = df_csv_MT4_Ticker_1_M1.reindex(columns=col_order_1_new)
                df_csv_MT4_Ticker_1_M1 = df_csv_MT4_Ticker_1_M1.drop('Date', axis=1)
                df_csv_MT4_Ticker_1_M1 = df_csv_MT4_Ticker_1_M1.drop('Time', axis=1)
                df_csv_MT4_Ticker_1_M1.reset_index()
                df_csv_MT4_Ticker_1_M1.set_index('DateTime',inplace=True)

                df_csv_MT4_Ticker_2_M1 = pd.read_csv(".\___lib_api_mt4\_data\\" + s2 + "1.csv", header=None)
                df_csv_MT4_Ticker_2_M1.columns = ['Date','Time',s2+'_O',s2+'_H',s2+'_L',s2+'_C',s2+'_V']
                df_csv_MT4_Ticker_2_M1['DateTime'] = df_csv_MT4_Ticker_2_M1['Date'] + ' ' + df_csv_MT4_Ticker_2_M1['Time']
                col_order_2_new = ['DateTime','Date','Time',s2+'_O',s2+'_H',s2+'_L',s2+'_C',s2+'_V']
                df_csv_MT4_Ticker_2_M1 = df_csv_MT4_Ticker_2_M1.reindex(columns=col_order_2_new)
                df_csv_MT4_Ticker_2_M1 = df_csv_MT4_Ticker_2_M1.drop('Date', axis=1)
                df_csv_MT4_Ticker_2_M1 = df_csv_MT4_Ticker_2_M1.drop('Time', axis=1)
                df_csv_MT4_Ticker_2_M1.reset_index()
                df_csv_MT4_Ticker_2_M1.set_index('DateTime',inplace=True)

                df_csv_MT4 = df_csv_MT4_Ticker_1_M1.join(df_csv_MT4_Ticker_2_M1, on='DateTime')

                # print in streamlit in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.write(df_csv_MT4_Ticker_1_M1)
                with col2:
                    st.write(df_csv_MT4_Ticker_2_M1)
                st.write(df_csv_MT4)

    sidebar_checkbox_secutype_C_Fut_Opt_Energy_NatGas_US_USD_HenryHub = st.sidebar.checkbox("C_Fut_Opt_Energy_NatGas_US_USD_HenryHub")
    if sidebar_checkbox_secutype_C_Fut_Opt_Energy_NatGas_US_USD_HenryHub:
        st.header("C_Fut_Opt_Energy_NatGas_US_USD_HenryHub")
        C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Tickers = [
            'NGU22 Comdty',
            'NGU2P 10 Comdty', 'NGU2P 11 Comdty',
            'BOP000BN5MS Corp', 'BOP000BN5NG Corp',
            'BOP000BN75V Corp',
        ]
        C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Fields_Request = \
            ['market_sector_des',
             'id_cusip',
             'name',
             'country',
             'crncy',
             'direction',
             'settle_dt',
             'fut_roll_dt',
             'last_tradeable_dt',
             'opt_valuation_dt',
             'opt_expire_dt',
             'opt_days_expire',
             'opt_strike_px',
             'fxopt_notional',
             'fxopt_notional_ccy',
             'fxopt_commodity_ccy',
             'fxopt_price_ccy',
             'px_last',
             'quote_units',
             'fut_val_pt',
             'fut_px_val_bp',
             'fut_cont_size',
             'fut_trading_units',
             'contract_value',
             'opt_delta',
             'delta',
             '',
             '',
             '',
             ]
        st.write(C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Tickers)
        st.write(C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Fields_Request)

        try:
            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_bdp = bdp.BDP()
        except:
            st.write("Required: BBG")
        else:
            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_bdp.start_session()
            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_bdp.request(
                securities=C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Tickers,
                field_id=C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_Fields_Request
            )
            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df = C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_bdp.results
            # C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df

            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df_T = C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df.T
            C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df_T

            # AgGrid(C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df_T)
            AgGrid(C_Fut_Opt_Energy_NatGas_US_USD_HenryHub_df_T, height=500, fit_columns_on_grid_load=True)
        finally:
            st.write("Done.")

    sidebar_checkbox_secutype_XV = st.sidebar.checkbox("XV")
    if sidebar_checkbox_secutype_XV:
        st.header("XV")
        st.subheader("XV Live: bdp")
        st.checkbox_XV_Live_bdp = st.checkbox("XV Live: bdp")
        if st.checkbox_XV_Live_bdp:
            XV_Tickers = ['EURUSDV1Y Curncy', 'USDJPYV1Y Curncy']
            XV_Fields_Request = \
                ['name',
                 'px_last',
                 '',
                 ]
            XV_bdp = bdp.BDP()
            XV_bdp.start_session()
            XV_bdp.request(securities=XV_Tickers, field_id=XV_Fields_Request)
            XV_df = XV_bdp.results
            # XV_df

            XV_df_T = XV_df.T
            # XV_df_T

            # remio = Remember Index Old
            XV_df_T_ri_remio = PY_Sys_DataStructures.df_Reset_Index_Columns(XV_df_T, True, False, True, 'Fields')

            # AgGrid(XV_df_T)
            AgGrid(XV_df_T_ri_remio, height=100, fit_columns_on_grid_load=False)

        # Bloomberg: bdh()
        st.subheader("XV Hist: bdh")
        st.checkbox_XV_Hist_bdp = st.checkbox("XV Hist: bdh")
        if st.checkbox_XV_Hist_bdp:

            XV_Tickers = ['EURUSDV1Y Curncy', 'USDJPYV1Y Curncy']

            req = {
                'securities': XV_Tickers,
                'field_id': 'PX_LAST',
                'refdataservice': "HistoricalDataRequest",
                'periodicity': "DAILY",
                'start_date': (dt.datetime.today() - relativedelta(days=28)).strftime("%Y%m%d"),
                'end_date': dt.datetime.today().strftime("%Y%m%d")}
            all_req = []
            all_req.append(req)

            for req in all_req:
                df_bdh = bdh.bbapi_bdh(securities=req['securities'],
                                       field_id=req['field_id'],
                                       refdataservice=req["refdataservice"],
                                       periodicity=req["periodicity"],
                                       start_date=req['start_date'],
                                       end_date=req['end_date'])

                if len(df_bdh.index.tolist()) < 1:
                    df_bdh = pd.DataFrame(['Error'], columns=['Error'])

            # Plot XV1, XV2
            st.line_chart(df_bdh)
            df_bdh

            x = df_bdh.index
            x_labels = list(df_bdh.index)
            fig = plt.figure()
            fig.patch.set_facecolor('grey')
            plt.plot(df_bdh)
            plt.gca().set_facecolor('xkcd:grey')
            plt.xticks(x, x_labels, rotation=45)
            st.pyplot(fig)

            # df_bdh

            df_bdh_ri_remio_date = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh, True, False, True, 'date')
            AgGrid(df_bdh_ri_remio_date, height=700, fit_columns_on_grid_load=False)

            # Set col 'date' as index, then plot the line chart
            # df_bdh_ri_remio_date_index = df_bdh_ri_remio_date.rename(columns={'date':'index'}).set_index('index')
            # st.line_chart(df_bdh_ri_remio_date_index)

            # Calc XVS = XV1-XV2

            # Plot XVS

    sidebar_checkbox_secutype_D = st.sidebar.checkbox("D")
    if sidebar_checkbox_secutype_D:
        st.header("D = Digital")
        checkbox_D = st.checkbox("D = Digital")
        if checkbox_D:
            sidebar_radio_api = st.sidebar.radio("API", (["CoinGecko", "Others"]))

            st.text("-----" + " " + "D: AUM" + " " + "-" * 100)
            AUM = pd.read_csv(
                r'C:\Dropbox\_\work\___lib_api_coingecko\input_aum.csv').values[
                0, 0]
            st.write(AUM)

            st.text("-----" + " " + "D: Portfolio" + " " + "-" * 100)
            Portfolio = pd.read_csv(
                r'C:\Dropbox\_\work\___lib_api_coingecko\input_portfolio.csv')
            st.write(Portfolio)

            PortfolioSymbolsList = []

            if sidebar_radio_api == "CoinGecko":
                st.text("-----" + " " + "D Load Mkt Data from CoinGecko" + " " + "-" * 100)
                API_URL = 'https://api.coingecko.com/api/v3'
                RequestMarketList = requests.get(API_URL + '/coins/list', params={"vs_currency": "usd"})
                st.write(RequestMarketList)
                MarketListJson = RequestMarketList.json()
                st.write(MarketListJson)
                MarketList = pd.DataFrame(MarketListJson)
                TimeStampStart = int(time.time()) - 730 * 24 * 60 * 60
                TimeStampEnd = int(time.time())

    sidebar_checkbox_secutype_R_B_Corp = st.sidebar.checkbox("R_B_Corp")
    if sidebar_checkbox_secutype_R_B_Corp:
        st.header("R_B_Corp")
        checkbox_R_B_Corp_Live_bdp = st.checkbox("R_B_Corp Live: bdp")
        if checkbox_R_B_Corp_Live_bdp:
            st.subheader("R_B_Corp Live: bdp")
            R_B_Corp_Tickers = [
                'T 2.875 05/15/32 Govt',
                'AYRWF 12.5 12/10/24 Corp',
            ]
            R_B_Corp_Fields_Request = \
                ['name',
                 'px_last',
                 'dur_adj_mid',
                 'risk_ask',
                 'risk_mid',
                 'risk_bid',
                 ]
            R_B_Corp_bdp = bdp.BDP()
            R_B_Corp_bdp.start_session()
            R_B_Corp_bdp.request(securities=R_B_Corp_Tickers, field_id=R_B_Corp_Fields_Request)
            R_B_Corp_df = R_B_Corp_bdp.results
            R_B_Corp_df_T = R_B_Corp_df.T
            # R_B_Corp_df_T

            # remio = Remember Index Old
            R_B_Corp_df_T_ri_remio = PY_Sys_DataStructures.df_Reset_Index_Columns(R_B_Corp_df_T, True, False, True,
                                                                                  'Fields')
            R_B_Corp_df_T_ri_remio

            list_col_names = list(R_B_Corp_df_T_ri_remio)
            df_analytics = pd.DataFrame(np.array([['_', 0, 0]]), columns=list_col_names)

            B_01 = float(PY_Sys_DataStructures.df_VLookUp('PX_LAST', 'Fields', R_B_Corp_df_T_ri_remio,
                                                          R_B_Corp_Tickers[1 - 1].upper()))
            B_02 = float(PY_Sys_DataStructures.df_VLookUp('PX_LAST', 'Fields', R_B_Corp_df_T_ri_remio,
                                                          R_B_Corp_Tickers[2 - 1].upper()))
            df_row_to_insert = pd.DataFrame(np.array([['B', B_01, B_02]]), columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            D_mod_01 = float(PY_Sys_DataStructures.df_VLookUp('DUR_ADJ_MID', 'Fields', R_B_Corp_df_T_ri_remio,
                                                              R_B_Corp_Tickers[1 - 1].upper()))
            D_mod_02 = float(PY_Sys_DataStructures.df_VLookUp('DUR_ADJ_MID', 'Fields', R_B_Corp_df_T_ri_remio,
                                                              R_B_Corp_Tickers[2 - 1].upper()))
            df_row_to_insert = pd.DataFrame(np.array([['D_mod', D_mod_01, D_mod_02]]), columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            D_mod_10y = D_mod_01

            Pos_q_01 = 1000000
            Pos_q_02 = 39060000
            df_row_to_insert = pd.DataFrame(np.array([['Pos_q', Pos_q_01, Pos_q_02]]), columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            # R_DV01 = - 8.8 / 10k * B_10yEqv
            Pos_R_DV01_01 = - D_mod_01 / 10000 * (B_01 / 100) * Pos_q_01
            Pos_R_DV01_02 = - D_mod_02 / 10000 * (B_02 / 100) * Pos_q_02
            df_row_to_insert = pd.DataFrame(np.array([['Pos_R_DV01', Pos_R_DV01_01, Pos_R_DV01_02]]),
                                            columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            Pos_R_MV_01 = (B_01 / 100) * Pos_q_01
            Pos_R_MV_02 = (B_02 / 100) * Pos_q_02
            df_row_to_insert = pd.DataFrame(np.array([['Pos_R_MV', Pos_R_MV_01, Pos_R_MV_02]]), columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            Pos_R_MV_10yEqv_01 = - Pos_R_DV01_01 * 10000 / D_mod_10y
            Pos_R_MV_10yEqv_02 = - Pos_R_DV01_02 * 10000 / D_mod_10y
            df_row_to_insert = pd.DataFrame(np.array([['Pos_R_MV_10yEqv', Pos_R_MV_10yEqv_01, Pos_R_MV_10yEqv_02]]),
                                            columns=list_col_names)
            df_analytics = df_analytics.append(df_row_to_insert)

            df_analytics

            # AgGrid(R_B_Corp_df_T)
            # AgGrid(R_B_Corp_df_T_ri_remio, height=200, fit_columns_on_grid_load=False)
            AgGrid(df_analytics, height=300, fit_columns_on_grid_load=False)

    sidebar_checkbox_secutype_ERK_B_Cnvrt = st.sidebar.checkbox("ERK_B_Cnvrt")
    if sidebar_checkbox_secutype_ERK_B_Cnvrt:
        st.header("ERK_B_Cnvrt")
        ERK_B_Cnvrt_Tickers = ['KINSF 0.625 04/29/25 Corp', ' 3888 HK Equity']
        ERK_B_Cnvrt_Fields_Request = \
            ['name',
             'cur_mkt_cap',
             'rsk_bb_implied_cds_spread',
             'bb_5y_default_prob',
             'cds_recovery_cv_model',
             'bond_recovery_amt_cv_model',
             'announce_dt',
             'issue_dt',
             'first_settle_dt',
             'first_cpn_dt',
             '',
             ]
        st.write(ERK_B_Cnvrt_Tickers)
        st.write(ERK_B_Cnvrt_Fields_Request)
        try:
            ERK_B_Cnvrt_bdp = bdp.BDP()
        except:
            st.write("Required: BBG")
        else:
            ERK_B_Cnvrt_bdp.start_session()
            ERK_B_Cnvrt_bdp.request(securities=ERK_B_Cnvrt_Tickers, field_id=ERK_B_Cnvrt_Fields_Request)
            ERK_B_Cnvrt_df = ERK_B_Cnvrt_bdp.results
            ERK_B_Cnvrt_df_T = ERK_B_Cnvrt_df.T
            ERK_B_Cnvrt_df_T

            # AgGrid(ERK_B_Cnvrt_df_T)
            AgGrid(ERK_B_Cnvrt_df_T, height=500, fit_columns_on_grid_load=True)
        finally:
            st.write("Done.")
        # ERK_B_Cnvrt_df

    sidebar_checkbox_secutype_P = st.sidebar.checkbox("P")
    if sidebar_checkbox_secutype_P:
        st.header("P = Property")
        st.write("P_Fwd = Property Forward: Fwd Price = Cash Grown - Rent")



if sidebar_selectbox_mode == "DEV_TET_RiskParity":
    """ USING THE MATRIX NOTATION DEFINED IN
    APPENDIX OF "A CRITIQUE OF THE ASSET PRICING THEORY'S TESTS" ROLL (1977) 
    """

    # SIGMA = V = Covariance Matrix for SPY, GLD, TLT
    V = np.matrix(
        '0.035664981	0.010065823	0.005273817; 0.006216739 0.022027001 0.009224742; 0.005605385 0.015875274 0.037907257')  # covariance
    R = np.matrix('0.104298559; 0.094064988; -0.20860014')  # return
    rf = 3 / 100  # risk free

    # define the
    I = np.ones((len(R), 1))
    SD = np.sqrt(np.diag(V))

    # Variable for Efficient frontier (SEE (A.9) ROLL 1977 page 160)
    C = I.T * pinv(V) * I
    B = R.T * pinv(V) * I
    A = R.T * pinv(V) * R
    D = A * C - B ** 2

    ########################################################
    # EFFICIENT FRONTIER
    ########################################################
    # Efficient Frontier Range for Return
    mu = np.arange(-max(R), max(R) * 5, max(R) / 100);
    # mu = np.arange(-max(R), max(R) * 5, max(R));
    st.write('mu')
    st.write(mu)

    # Plot Efficient Frontier
    minvar = (A - 2 * B * mu + (C * mu ** 2)) / D;
    minstd = np.sqrt(minvar)[0];
    minstd = np.squeeze(np.asarray(minstd))
    fig, ax = plt.subplots(figsize=(6, 3))                     # streamlit plot: Add this at start
    plt.plot(minstd, mu, SD, R, '*')
    plt.title('Efficient Frontier', fontsize=10)
    plt.ylabel('Expected Return [%]', fontsize=8)
    plt.xlabel('Risk [%]', fontsize=8)
    plt.show()
    st.pyplot(fig)                                              # streamlit plot: Add this at end

    ########################################################
    # MVP
    ########################################################
    # Mean and Variance of Global Minimum Variance Portfolio
    mu_g = B / C
    var_g = 1 / C
    std_g = np.sqrt(var_g)

    # Minimum Variance Portfolio Weights
    w_g = (pinv(V) * I) / C
    st.write('-' * 100)
    st.write('w_g')
    st.write(w_g)

    # Plot Efficient Frontier
    minvar = (A - 2 * B * mu + (C * mu ** 2)) / D;
    minstd = np.sqrt(minvar)[0];
    minstd = np.squeeze(np.asarray(minstd))
    fig, ax = plt.subplots(figsize=(6, 3))                     # streamlit plot: Add this at start
    plt.plot(minstd, mu, SD, R, '*', std_g, mu_g, 'ro')
    plt.title
    plt.title("Efficient Frontier", fontsize=10)
    plt.ylabel('Expected Return [%]', fontsize=8)
    plt.xlabel('Risk [%]', fontsize=8)
    plt.show()
    st.pyplot(fig)                                              # streamlit plot: Add this at end

    ########################################################
    # TANGENT PORTFOLIO
    ########################################################
    # Expected Return of Tangency Portfolio
    mu_tan = (B * rf - A) / (C * rf - B);

    # Variance and Standard Deviation of Tangency Portfolio
    vartan = (A - 2 * rf * B + (rf ** 2 * C)) / ((B - C * rf) ** 2);
    stdtan = np.sqrt(vartan);

    # Weights for Tangency Portfolio
    w_tan = (pinv(V) * (R - rf * I)) / (B - C * rf)
    st.write('-' * 100)
    st.write('w_tan')
    st.write(w_tan)

    # Tangency Line
    m_tan = mu[mu >= rf];
    minvar_rf = (m_tan - rf) ** 2 / (A - 2 * rf * B + C * rf ** 2);
    minstd_rf = np.sqrt(minvar_rf);
    minstd_rf = np.squeeze(np.asarray(minstd_rf))

    # Plot with tangency portfolio
    plt.plot(minstd, mu, SD, R, '*', minstd_rf, m_tan, 'r', std_g, mu_g, 'bo', stdtan, mu_tan, 'go')
    plt.title('Efficient Frontier', fontsize=18)
    plt.ylabel('Expected Return [%]', fontsize=12)
    plt.xlabel('Risk [%]', fontsize=12)
    plt.text(0.5, rf, 'rf', fontsize=12)
    plt.title
    plt.text(0.5 + std_g, mu_g, 'Global MVP', fontsize=12);
    plt.text(0.5 + stdtan, mu_tan, 'TP', fontsize=12);
    plt.show()


    # risk budgeting optimization
    def calculate_portfolio_var(w, V):
        # function that calculates portfolio variance = portfolio risk
        w = np.matrix(w)
        return (w * V * w.T)[0, 0]


    def calculate_risk_contribution(w, V):
        # function that calculates asset contribution to total risk
        w = np.matrix(w)
        sigma = np.sqrt(calculate_portfolio_var(w, V))
        # Marginal Risk Contribution
        MRC = V * w.T
        # Risk Contribution
        RC = np.multiply(MRC, w.T) / sigma
        st.write('RC')
        st.write(RC)
        return RC


    def risk_budget_objective(x, pars):
        # calculate portfolio risk
        V = pars[0]  # covariance table
        x_t = pars[1]  # risk target in percent of portfolio risk
        sig_p = np.sqrt(calculate_portfolio_var(x, V))  # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p, x_t))
        asset_RC = calculate_risk_contribution(x, V)
        J = sum(np.square(asset_RC - risk_target.T))[0, 0]  # sum of squared error
        return J


    def total_weight_constraint(x):
        return np.sum(x) - 1.0


    def long_only_constraint(x):
        return x


    # Risk budget [% of Total Portfolio Risk]
    # Assume and equal weighted portfolio
    x_t = [0.333, 0.333, 0.333]

    cons = (
        {'type': 'eq', 'fun': total_weight_constraint},
        {'type': 'ineq', 'fun': long_only_constraint}
    )

    # Set initial weights to weights of the tangency portfolio
    w0 = [0.333, 0.333, 0.333]

    res = minimize(risk_budget_objective, w0, args=[V, x_t], method='SLSQP', constraints=cons, options={'disp': True})
    w_rb = np.asmatrix(res.x)
    print('-' * 100)
    print('w_rb')
    print(w_rb)

if sidebar_selectbox_mode == "DEV_TET_Hangman":
    st.write("DEV_TET_Hangman")

    def list_from_text(s_pne_txt):
        txt = open(s_pne_txt, "r")
        lst = txt.read().splitlines()
        txt.close()
        return lst

    list_failed = list_from_text("C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_failed.txt")
    st.write(list_failed)

    sidebar_radio_steps = st.sidebar.radio("Steps", (["010_Train","020_Combine","021_SortFailed","022_CountFailed","030_API_Upload","040_Practice_1","050_Play"]))

    if sidebar_radio_steps == "010_Train":
        st.write("010Train")

        # Take the orginal training dictionary of about 250,000 words
        # Makes list of prefixes of length 2 to length 20
        s_train = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\words_250000_train.txt"

        def build_dictionary(dictionary_file_location):
            text_file = open(dictionary_file_location, "r")
            full_dictionary = text_file.read().splitlines()
            text_file.close()
            return full_dictionary

        list_dict_train = build_dictionary(s_train)
        list_to_scan = []
        # list_to_scan = list_dict_train[0:1000]
        # list_to_scan = list_dict_train[1000:2000]
        # list_to_scan = list_dict_train[2000:10000]
        # list_to_scan = list_dict_train[10000:100000]
        # list_to_scan = list_dict_train[100000:200000]
        # list_to_scan = list_dict_train[200000:250000] # 227300
        st.write(list_to_scan)

        list_subwords_02 = []
        list_subwords_03 = []
        list_subwords_04 = []
        list_subwords_05 = []
        list_subwords_06 = []
        list_subwords_07 = []
        list_subwords_08 = []
        list_subwords_09 = []
        list_subwords_10 = []
        list_subwords_11 = []
        list_subwords_12 = []
        list_subwords_13 = []
        list_subwords_14 = []
        list_subwords_15 = []
        list_subwords_16 = []
        list_subwords_17 = []
        list_subwords_18 = []
        list_subwords_19 = []
        list_subwords_20 = []
        list_subwords_21 = []
        list_subwords_22 = []
        list_subwords_23 = []

        def get_list_subwords_from_list_dict(list_dict):
            st.write("Num words in list_to_scan = " + str(len(list_dict)))
            count = 0
            for i in list_to_scan:
                count = count + 1
                st.write(count)

                s1 = i
                if i != list_to_scan[-1]:
                    s2 = list_to_scan[list_to_scan.index(i)+1]    # the next item
                    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
                    s_match = s1[match.a:match.a + match.size]
                    # st.write(s1 + "+" + s2 + "=" + s_match)

                    match len(s_match):
                        case 0:
                            pass
                        case 1:
                            pass
                        case 2:
                            if s_match not in list_subwords_02:
                                list_subwords_02.append(s_match)
                            # st.write(list_subwords_02)
                        case 3:
                            if s_match not in list_subwords_03:
                                list_subwords_03.append(s_match)
                            # st.write(list_subwords_03)
                        case 4:
                            if s_match not in list_subwords_04:
                                list_subwords_04.append(s_match)
                            # st.write(list_subwords_04)
                        case 5:
                            if s_match not in list_subwords_05:
                                list_subwords_05.append(s_match)
                            # st.write(list_subwords_05)
                        case 6:
                            if s_match not in list_subwords_06:
                                list_subwords_06.append(s_match)
                            # st.write(list_subwords_06)
                        case 7:
                            if s_match not in list_subwords_07:
                                list_subwords_07.append(s_match)
                            # st.write(list_subwords_07)
                        case 8:
                            if s_match not in list_subwords_08:
                                list_subwords_08.append(s_match)
                            # st.write(list_subwords_08)
                        case 9:
                            if s_match not in list_subwords_09:
                                list_subwords_09.append(s_match)
                            # st.write(list_subwords_09)
                        case 10:
                            if s_match not in list_subwords_10:
                                list_subwords_10.append(s_match)
                            # st.write(list_subwords_10)
                        case 11:
                            if s_match not in list_subwords_11:
                                list_subwords_11.append(s_match)
                            # st.write(list_subwords_11)
                        case 12:
                            if s_match not in list_subwords_12:
                                list_subwords_12.append(s_match)
                            # st.write(list_subwords_12)
                        case 13:
                            if s_match not in list_subwords_13:
                                list_subwords_13.append(s_match)
                            # st.write(list_subwords_13)
                        case 14:
                            if s_match not in list_subwords_14:
                                list_subwords_14.append(s_match)
                            # st.write(list_subwords_14)
                        case 15:
                            if s_match not in list_subwords_15:
                                list_subwords_15.append(s_match)
                            # st.write(list_subwords_15)
                        case 16:
                            if s_match not in list_subwords_16:
                                list_subwords_16.append(s_match)
                            # st.write(list_subwords_16)
                        case 17:
                            if s_match not in list_subwords_17:
                                list_subwords_17.append(s_match)
                            # st.write(list_subwords_17)
                        case 18:
                            if s_match not in list_subwords_18:
                                list_subwords_18.append(s_match)
                            # st.write(list_subwords_18)
                        case 19:
                            if s_match not in list_subwords_19:
                                list_subwords_18.append(s_match)
                            # st.write(list_subwords_19)
                        case 20:
                            if s_match not in list_subwords_20:
                                list_subwords_20.append(s_match)
                            # st.write(list_subwords_20)

                        # If an exact match is not confirmed, this last case will be used if provided
                        case _:
                            st.write("Create list_subwords_" + str(len(s_match)))


        get_list_subwords_from_list_dict(list_to_scan)

        s_path = "C:/Dropbox/_/work/___lib_py/_tutorial_trexquant/project"

        def list_write(lst,i):
            st.write(lst)
            with open(s_path + "/" + "list_" + str(i) + ".txt", 'a') as outfile:
                outfile.write('\n'.join(str(i) for i in lst))
                outfile.write('\n')

        list_write(list_subwords_02, 2)
        list_write(list_subwords_03, 3)
        list_write(list_subwords_04, 4)
        list_write(list_subwords_05, 5)
        list_write(list_subwords_06, 6)
        list_write(list_subwords_07, 7)
        list_write(list_subwords_08, 8)
        list_write(list_subwords_09, 9)
        list_write(list_subwords_10, 10)
        list_write(list_subwords_11, 11)
        list_write(list_subwords_12, 12)
        list_write(list_subwords_13, 13)
        list_write(list_subwords_14, 14)
        list_write(list_subwords_15, 15)
        list_write(list_subwords_16, 16)
        list_write(list_subwords_17, 17)
        list_write(list_subwords_18, 18)
        list_write(list_subwords_19, 19)
        list_write(list_subwords_20, 20)

    if sidebar_radio_steps == "020_Combine":
        st.write("020_Combine")
        def list_from_text(s_pne_txt):
            txt = open(s_pne_txt, "r")
            lst = txt.read().splitlines()
            txt.close()
            return lst

        def list_combine(lst,i):
            st.write(lst)
            with open(s_path + "/" + "list_prefix_02_to_04.txt", 'a') as outfile:       # <-- set the filename
                outfile.write('\n'.join(str(i) for i in lst))
                outfile.write('\n')

        s_path = "C:/Dropbox/_/work/___lib_py/_tutorial_trexquant/project"
        # s_list_dst = s_path + '/' + 'list_prefix.txt'
        list_prefix = []

        for i in range(2, 5):                                                  # <-- set the range from 4 letters to 8 letters
            s_pne = s_path + '/' + 'list_' + str(i) + '.txt'
            st.write(s_pne)
            lst = list_from_text(s_pne)
            # list_combine(lst, i)                                              # <-- uncomment to run and write file

    if sidebar_radio_steps == "021_SortFailed":
        st.write("021_SortFailed")
        s_list_failed = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_failed.txt"
        lst_failed = list_from_text(s_list_failed)
        # sort list by length
        # lst_failed.sort(lambda x, y: cmp(len(x), len(y)))
        lst_failed_sorted = sorted(lst_failed, key=len)
        with open(s_list_failed, 'w') as outfile:  # <-- set the filename
            outfile.write('\n'.join(str(i) for i in lst_failed_sorted))
            outfile.write('\n')

        st.write(lst_failed_sorted)

    if sidebar_radio_steps == "022_CountFailed":
        st.write("022_CountFailed")

        def count_and_sort_letters(strings):
            # Concatenate all strings into one
            all_text = ''.join(strings)
            # Use Counter to count occurrences of each letter
            letter_counts = Counter(all_text)
            # Sort the counts by letter
            sorted_letter_counts = sorted(letter_counts.items())
            return sorted_letter_counts

        s_list_failed = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_failed.txt"
        lst_failed = list_from_text(s_list_failed)
        lst_failed_count = count_and_sort_letters(lst_failed)
        df_lst_failed_count = pd.DataFrame(lst_failed_count, columns=['letter','count'])
        df_lst_failed_count_sorted = df_lst_failed_count.sort_values(by='count', ascending=True)
        st.write(df_lst_failed_count_sorted)
        st.write(''.join(df_lst_failed_count_sorted['letter']))

    if sidebar_radio_steps == "030_API_Upload":
        st.write("030_API_Upload")
        s_path = "C:/Dropbox/_/work/___lib_py/_tutorial_trexquant/project"

        def list_from_text(s_pne_txt):
            txt = open(s_pne_txt, "r")
            lst = txt.read().splitlines()
            txt.close()
            return lst

        def list_to_text(lst):
            st.write(lst)
            with open(s_path + "/" + "list_failed.txt", 'w') as outfile:
                outfile.write("\n".join(map(str, lst)))
                outfile.write("\n")

        try:
            from urllib.parse import parse_qs, urlencode, urlparse
        except ImportError:
            from urlparse import parse_qs, urlparse
            from urllib import urlencode

        from requests.packages.urllib3.exceptions import InsecureRequestWarning

        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        # === Code: HangmanAPI ===
        class HangmanAPI(object):

            # ==== Code: HangmanAPI: def __init__ ====
            def __init__(self, access_token=None, session=None, timeout=None):
                self.hangman_url = self.determine_hangman_url()
                self.access_token = access_token
                self.session = session or requests.Session()
                self.timeout = timeout

                # ===== Code: HangmanAPI: def __init__: guessed_letters =====
                self.guessed_letters = []

                # ===== Code: HangmanAPI: def __init__: full_dictionary = the training list =====
                st.write('=' * 55)
                st.write('[010]: .txt file = training dictionary')
                # full_dictionary_location = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\words_250000_train.txt"
                full_dictionary_location = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_prefix.txt"
                st.write(full_dictionary_location)

                st.write('=' * 55)
                st.write('[020]: list from .txt. file = dictionary')
                self.full_dictionary = self.build_dictionary(full_dictionary_location)
                # st.write(type(self.full_dictionary))      # <class 'list'>
                st.write(self.full_dictionary[:5])        # ['aaa', 'aaaaaa', 'aaas', 'aachen', 'aaee']

                st.write('=' * 55)
                st.write('[030]: list of tuples = full_dictionary_common_letter_sorted')
                self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
                # st.write(type(self.full_dictionary_common_letter_sorted))
                st.write(self.full_dictionary_common_letter_sorted)
                # st.write(type(self.full_dictionary_common_letter_sorted[0]))
                st.write(self.full_dictionary_common_letter_sorted[0])

                st.write('=' * 55)
                st.write('[040]: a_string from list of tuples = eairontlsucpdhmbgfwvykzxjq')
                self.a_string = ''.join(item[0] for item in self.full_dictionary_common_letter_sorted)
                st.write(self.a_string)
                st.write('[041]: z_string from list of tuples = qjxzkyvwfgbmhdpcusltnoriae')
                self.z_string = self.a_string[::-1]
                st.write(self.z_string)

                # ===== Code: HangmanAPI: def __init__: full_dictionary_XX_to_XX = the training list based on =====
                s_dict_02_to_04 = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_prefix_02_to_04.txt"
                self.full_dictionary_02_to_04 = self.build_dictionary(s_dict_02_to_04)
                self.full_dictionary_common_letter_sorted_02_to_04 = collections.Counter("".join(self.full_dictionary_02_to_04)).most_common()

                s_dict_04_to_08 = "C:\Dropbox\_\work\___lib_py\_tutorial_trexquant\project\list_prefix_04_to_08.txt"
                self.full_dictionary_04_to_08 = self.build_dictionary(s_dict_04_to_08)
                self.full_dictionary_common_letter_sorted_04_to_08 = collections.Counter("".join(self.full_dictionary_04_to_08)).most_common()

                self.current_dictionary = []

            @staticmethod
            def determine_hangman_url():
                links = ['https://trexsim.com', 'https://sg.trexsim.com']

                data = {link: 0 for link in links}

                for link in links:

                    requests.get(link)

                    for i in range(10):
                        s = time.time()
                        requests.get(link)
                        data[link] = time.time() - s

                link = sorted(data.items(), key=lambda x: x[1])[0][0]
                link += '/trexsim/hangman'
                return link

            b_wrong_guess_1 = False
            b_wrong_guess_2 = False

            def guess(self, word):  # word input example: "_ p p _ e "
                ###############################################
                # Replace with your own "guess" function here #
                ###############################################

                # clean the word so that we strip away the space characters
                # replace "_" with "." as "." indicates any character in regular expressions
                clean_word = word[::2].replace("_", ".")
                # print("clean_word=" + clean_word)       # .pp.e = apple
                # st.write("clean_word=" + clean_word)

                # find length of passed word
                len_word = len(clean_word)

                # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
                current_dictionary = self.current_dictionary
                new_dictionary = []

                # iterate through all of the words in the old plausible dictionary
                for dict_word in current_dictionary:
                    # continue if the word is not of the appropriate length
                    if len(dict_word) != len_word:
                        continue

                    # if dictionary word is a possible match then add it to the current dictionary
                    if re.match(clean_word, dict_word):
                        new_dictionary.append(dict_word)

                # overwrite old possible words dictionary with updated version
                self.current_dictionary = new_dictionary

                # count occurrence of all characters in possible word matches
                full_dict_string = "".join(new_dictionary)

                # st.write('full_dict_string')
                # st.write(full_dict_string)

                c = collections.Counter(full_dict_string)
                sorted_letter_count = c.most_common()

                st.write('-' * 555)
                st.write('sorted_letter_count')
                st.write(sorted_letter_count)

                guess_letter = '!'

                # return most frequently occurring letter in all possible words that hasn't been guessed yet
                # for letter, instance_count in sorted_letter_count:
                #     if letter not in self.guessed_letters:
                #         guess_letter = letter
                #         break

                # if no word matches in training dictionary, default back to ordering of full dictionary
                if guess_letter == '!':

                    if len_word <= 4:
                        sorted_letter_count = self.full_dictionary_common_letter_sorted_02_to_04
                    else:
                        sorted_letter_count = self.full_dictionary_common_letter_sorted_04_to_08
                        # sorted_letter_count = self.full_dictionary_common_letter_sorted

                    for letter, instance_count in sorted_letter_count:
                        if (letter in self.not_guessed_letters) and (letter not in self.guessed_letters):
                            self.not_guessed_letters.remove(letter) # remove from not_guessed list
                            self.guessed_letters.append(letter)     # add to guessed list
                            guess_letter = letter
                            break

                return guess_letter

            ##########################################################
            # You'll likely not need to modify any of the code below #
            ##########################################################

            def build_dictionary(self, dictionary_file_location):
                text_file = open(dictionary_file_location, "r")
                full_dictionary = text_file.read().splitlines()
                text_file.close()
                return full_dictionary


            def start_game(self, practice=True, verbose=True):
                # reset guessed letters to empty set and current plausible dictionary to the full dictionary
                self.guessed_letters = []
                self.not_guessed_letters = list(string.ascii_lowercase)
                self.current_dictionary = self.full_dictionary
                self.list_failed = []
                self.b_wrong_guess_1 = False
                self.b_wrong_guess_2 = False

                print("not_guessed_letters: " + str(self.not_guessed_letters))

                response = self.request("/new_game", {"practice": practice})
                if response.get('status') == "approved":
                    game_id = response.get('game_id')
                    word = response.get('word')
                    my_clean_word = word[::2].replace("_", ".")
                    print('not_guessed_letters='+ str(self.not_guessed_letters))
                    print('rand='+ choice(self.not_guessed_letters))

                    tries_remains_prev = 6
                    tries_remains = response.get('tries_remains')

                    if verbose:
                        print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
                        st.write("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
                    while tries_remains > 0:

                        st.write("guesses: " + ''.join(self.guessed_letters) + " | " + ''.join(self.not_guessed_letters))

                        # if last two guesses were wrong, and lengthe is longer than 6 letters, then we try the rare list
                        if self.b_wrong_guess_1 and self.b_wrong_guess_2 and len(my_clean_word) > 8:

                            print("..... Last 2 guesses wrong, length more than 8. Current word is " + word)
                            st.write("..... Last 2 guesses wrong, length more than 8. Current word is " + word)

                            z_string = 'vbdgmyhpcusltnroaie_'
                            z_string = 'gdvmbyzqhpcusltnroaie_'
                            z_string = 'zvbdmghypcusltnroaie_'


                            i = 0
                            while z_string[i:i+1] not in self.not_guessed_letters:
                                i=i+1
                            if z_string[i:i+1] in self.not_guessed_letters:
                                guess_letter = z_string[i:i+1]
                            else:
                                # usual method
                                guess_letter = self.guess(word)

                            # usual
                            # guess_letter = self.guess(word)
                            #     # random choice
                            #     # guess_letter = choice(self.not_guessed_letters)

                        else:
                            # get guessed letter from user code

                            # front
                            if my_clean_word[0:3] == "a.r":
                                if 'u' in self.not_guessed_letters:
                                    guess_letter = 'u'
                            elif my_clean_word[0:3] == "e.p":
                                if 'x' in self.not_guessed_letters:
                                    guess_letter = 'x'
                            elif my_clean_word[0:3] == "ie.":
                                if 's' in self.not_guessed_letters:
                                    guess_letter = 's'
                            elif my_clean_word[0:3] == ".e.":
                                if 'r' in self.not_guessed_letters:
                                    guess_letter = 'r'
                                elif 'v' in self.not_guessed_letters:
                                    guess_letter = 'v'
                            elif my_clean_word[0:3] == "re.":
                                if 'v' in self.not_guessed_letters:
                                    guess_letter = 'v'
                            elif my_clean_word[0:4] == ".ra.":
                                if 'f' in self.not_guessed_letters:
                                    guess_letter = 'f'
                            elif my_clean_word[0:4] == "fra.":
                                if 'c' in self.not_guessed_letters:
                                        guess_letter = 'c'
                            elif my_clean_word[0:4] == ".a.a":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'
                            elif my_clean_word[0:4] == "ga.a":
                                if 'l' in self.not_guessed_letters:
                                    guess_letter = 'l'
                            elif my_clean_word[0:4] == ".eli":
                                if 'm' in self.not_guessed_letters:
                                    guess_letter = 'm'
                            elif my_clean_word[0:4] == "o.er":
                                if 'v' in self.not_guessed_letters:
                                    guess_letter = 'v'
                            elif my_clean_word[0:4] == ".eal":
                                if 'm' in self.not_guessed_letters:
                                    guess_letter = 'm'
                            elif 'z' in self.not_guessed_letters:
                                if 'z' in self.not_guessed_letters:
                                    guess_letter = 'z'
                            elif my_clean_word[0:5] == "alon.":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'
                            elif my_clean_word[0:5] == ".eve.":
                                if 'l' in self.not_guessed_letters:
                                    guess_letter = 'l'
                            elif my_clean_word[0:5] == "..oto":
                                if 'p' in self.not_guessed_letters:
                                    guess_letter = 'p'
                            elif my_clean_word[0:5] == "l.tic":
                                if 'y' in self.not_guessed_letters:
                                    guess_letter = 'y'
                            elif my_clean_word[0:5] == "s.oal":
                                if 'h' in self.not_guessed_letters:
                                    guess_letter = 'h'
                            elif my_clean_word[0:5] == "p.oto":
                                if 'h' in self.not_guessed_letters:
                                    guess_letter = 'h'
                            elif my_clean_word[0:5] == "ran..":
                                if 'c' in self.not_guessed_letters:
                                    guess_letter = 'c'
                            elif my_clean_word[0:5] == "ranc.":
                                if 'h' in self.not_guessed_letters:
                                    guess_letter = 'h'
                            elif my_clean_word[0:6] == "cra.on":
                                if 'y' in self.not_guessed_letters:
                                    guess_letter = 'y'
                            elif my_clean_word[0:6] == ".itter":
                                if 'b' in self.not_guessed_letters:
                                    guess_letter = 'b'
                                elif 'j' in self.not_guessed_letters:
                                    guess_letter = 'j'
                            elif my_clean_word[0:6] == "un.oll":
                                if 'p' in self.not_guessed_letters:
                                    guess_letter = 'p'
                            elif my_clean_word[0:6] == "bene.it":
                                if 'f' in self.not_guessed_letters:
                                    guess_letter = 'f'
                            elif my_clean_word[0:10] == ".enetre":      # fenetre = window
                                if 'f' in self.not_guessed_letters:
                                    guess_letter = 'f'
                            elif my_clean_word[0:10] == "photo.raph":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'

                            # middle use regex?

                            # back
                            elif my_clean_word[-2:] == "a.e":
                                if 't' in self.not_guessed_letters:
                                    guess_letter = 't'
                                elif 'v' in self.not_guessed_letters:
                                    guess_letter = 'v'
                                elif 'l' in self.not_guessed_letters:
                                    guess_letter = 'l'
                            elif my_clean_word[-2:] == "p.":
                                if 'h' in self.not_guessed_letters:
                                    guess_letter = 'h'
                            elif my_clean_word[-2:] == "l.":
                                if 'y' in self.not_guessed_letters:
                                    guess_letter = 'y'
                            elif my_clean_word[-3:] == "in.":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'
                            elif my_clean_word[-3:] == "i.g":
                                if 'n' in self.not_guessed_letters:
                                    guess_letter = 'n'
                            elif my_clean_word[-3:] == "iu.":
                                if 'm' in self.not_guessed_letters:
                                    guess_letter = 'm'
                            elif my_clean_word[-3:] == "tr.":
                                if 'y' in self.not_guessed_letters:
                                    guess_letter = 'y'
                            elif my_clean_word[-4:] == "a.le":
                                if 'b' in self.not_guessed_letters:
                                    guess_letter = 'b'
                            elif my_clean_word[-4:] == "in.s":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'
                            elif my_clean_word[-4:] == "oo.e":
                                if 'z' in self.not_guessed_letters:
                                    guess_letter = 'z'
                            elif my_clean_word[-4:] == "la.e":
                                if 'z' in self.not_guessed_letters:
                                    guess_letter = 'z'
                            elif my_clean_word[-5:] == "ousl.":
                                if 'y' in self.not_guessed_letters:
                                    guess_letter = 'y'
                            elif my_clean_word[-5:] == ".laze":
                                if 'b' in self.not_guessed_letters:
                                    guess_letter = 'b'
                            elif my_clean_word[-5:] == ".rap.":
                                if 'g' in self.not_guessed_letters:
                                    guess_letter = 'g'
                            elif my_clean_word[-5:] == "grap.":
                                if 'h' in self.not_guessed_letters:
                                    guess_letter = 'h'
                            elif my_clean_word[-6:] == "ti.ely":
                                if 'v' in self.not_guessed_letters:
                                    guess_letter = 'v'
                            elif my_clean_word[-8:] == "re.ident":
                                if 's' in self.not_guessed_letters:
                                    guess_letter = 's'

                            else:
                                guess_letter = self.guess(word)

                        # append guessed letter to guessed letters field in hangman object
                        if guess_letter not in self.guessed_letters:
                            self.guessed_letters.append(guess_letter)
                        if guess_letter in self.not_guessed_letters:
                            self.not_guessed_letters.remove(guess_letter)

                        if verbose:
                            print("Guessing letter: {0}".format(guess_letter) + "|" + str(self.b_wrong_guess_2) + str(self.b_wrong_guess_1))
                            st.write("Guessing letter: {0}".format(guess_letter) + "|" + str(self.b_wrong_guess_2) + str(self.b_wrong_guess_1))
                        try:
                            res = self.request("/guess_letter",
                                               {"request": "guess_letter", "game_id": game_id, "letter": guess_letter})
                        except HangmanAPIError:
                            print('HangmanAPIError exception caught on request.')
                            st.write('HangmanAPIError exception caught on request.')
                            continue
                        except Exception as e:
                            print('Other exception caught on request.')
                            st.write('Other exception caught on request.')
                            raise e

                        if verbose:
                            print("Sever response: {0}".format(res))
                            st.write("Sever response: {0}".format(res))

                        status = res.get('status')
                        tries_remains = res.get('tries_remains')

                        # store last guess result
                        self.b_wrong_guess_2 = self.b_wrong_guess_1

                        # check last guess
                        if tries_remains < tries_remains_prev:
                            # previous guess was wrong
                            self.b_wrong_guess_1 = True
                        else:
                            self.b_wrong_guess_1 = False

                        if status == "success":
                            if verbose:
                                print("Successfully finished game: {0}".format(game_id))
                                st.write("Successfully finished game: {0}".format(game_id))
                            return True
                        elif status == "failed":
                            # replace the gaps
                            list_failed.append(word.replace(" ",""))
                            # write to txt file the list of failed words
                            list_to_text(list_failed)

                            reason = res.get('reason', '# of tries exceeded!')
                            if verbose:
                                print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                                st.write("Failed game: {0}. Because of: {1}".format(game_id, reason))
                                st.write(word)
                            return False
                        elif status == "ongoing":
                            word = res.get('word')
                            my_clean_word = word[::2].replace("_", ".")

                        # update tries
                        tries_remains_prev = tries_remains

                else:
                    if verbose:
                        print("Failed to start a new game")
                        st.write("Failed to start a new game")
                return status == "success"

            def my_status(self):
                return self.request("/my_status", {})

            def request(
                    self, path, args=None, post_args=None, method=None):
                if args is None:
                    args = dict()
                if post_args is not None:
                    method = "POST"

                # Add `access_token` to post_args or args if it has not already been
                # included.
                if self.access_token:
                    # If post_args exists, we assume that args either does not exists
                    # or it does not need `access_token`.
                    if post_args and "access_token" not in post_args:
                        post_args["access_token"] = self.access_token
                    elif "access_token" not in args:
                        args["access_token"] = self.access_token

                time.sleep(0.2)

                num_retry, time_sleep = 50, 2
                for it in range(num_retry):
                    try:
                        response = self.session.request(
                            method or "GET",
                            self.hangman_url + path,
                            timeout=self.timeout,
                            params=args,
                            data=post_args,
                            verify=False
                        )
                        break
                    except requests.HTTPError as e:
                        response = json.loads(e.read())
                        raise HangmanAPIError(response)
                    except requests.exceptions.SSLError as e:
                        if it + 1 == num_retry:
                            raise
                        time.sleep(time_sleep)

                headers = response.headers
                if 'json' in headers['content-type']:
                    result = response.json()
                elif "access_token" in parse_qs(response.text):
                    query_str = parse_qs(response.text)
                    if "access_token" in query_str:
                        result = {"access_token": query_str["access_token"][0]}
                        if "expires" in query_str:
                            result["expires"] = query_str["expires"][0]
                    else:
                        raise HangmanAPIError(response.json())
                else:
                    raise HangmanAPIError('Maintype was not text, or querystring')

                if result and isinstance(result, dict) and result.get("error"):
                    raise HangmanAPIError(result)
                return result


        class HangmanAPIError(Exception):
            def __init__(self, result):
                self.result = result
                self.code = None
                try:
                    self.type = result["error_code"]
                except (KeyError, TypeError):
                    self.type = ""

                try:
                    self.message = result["error_description"]
                except (KeyError, TypeError):
                    try:
                        self.message = result["error"]["message"]
                        self.code = result["error"].get("code")
                        if not self.type:
                            self.type = result["error"].get("type", "")
                    except (KeyError, TypeError):
                        try:
                            self.message = result["error_msg"]
                        except (KeyError, TypeError):
                            self.message = result

                Exception.__init__(self, self.message)

        if 'api' not in st.session_state:
            st.session_state.api = HangmanAPI(access_token="241fc6ebac4e319e37c73357c8f579", timeout=2000)

        st.write("done")

    if sidebar_radio_steps == "040_Practice_1":
        st.write("040_Practice_1")

        [prev_total_practice_runs, prev_total_recorded_runs,
         prev_total_recorded_successes, prev_total_practice_successes] = st.session_state.api.my_status()  # Get my game stats: (# of tries, # of wins)

        for i in range(1):
            print('Playing ', i, ' th game')
            st.session_state.api.start_game(practice=1, verbose=True)

            # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
            time.sleep(0.5)

        [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = st.session_state.api.my_status()  # Get my game stats: (# of tries, # of wins)
        practice_success_rate = total_practice_successes / total_practice_runs
        success_rate = total_recorded_successes / total_recorded_runs

        this_practice_runs = total_practice_runs - prev_total_practice_runs
        this_recorded_runs = total_recorded_runs - prev_total_recorded_runs
        this_total_recorded_successes = total_recorded_successes - prev_total_recorded_successes
        this_total_practice_successes = total_practice_successes - prev_total_practice_successes
        this_practice_success_rate = this_total_practice_successes / this_practice_runs
        st.write('this_practice_success_rate = ' + str(this_practice_success_rate))

        print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
        st.write('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
        print('run %d recorded games out of an allotted   1,000. recorded success rate so far = %.3f' % (total_recorded_runs, success_rate))
        st.write('run %d recorded games out of an allotted   1,000. recorded success rate so far = %.3f' % (total_recorded_runs, success_rate))
        print('overall success rate = %.3f' % success_rate)
        st.write('overall success rate = %.3f' % success_rate)

        st.sidebar.write('this_practice_success_rate = %.3f' % (this_practice_success_rate))
        st.sidebar.write('practice games = %d' % (total_practice_runs))
        st.sidebar.write('success rate   = %.3f' % (practice_success_rate))

    if sidebar_radio_steps == "050_Play":
        st.write("050_Play")

        [prev_total_practice_runs, prev_total_recorded_runs,
         prev_total_recorded_successes, prev_total_practice_successes] = st.session_state.api.my_status()  # Get my game stats: (# of tries, # of wins)

        for i in range(5):
            print('Playing ', i, ' th game')
            st.session_state.api.start_game(practice=0, verbose=True)
            # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
            time.sleep(0.5)

        [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = st.session_state.api.my_status()  # Get my game stats: (# of tries, # of wins)
        practice_success_rate = total_practice_successes / total_practice_runs
        success_rate = total_recorded_successes / total_recorded_runs

        this_practice_runs = total_practice_runs - prev_total_practice_runs
        this_recorded_runs = total_recorded_runs - prev_total_recorded_runs
        this_total_recorded_successes = total_recorded_successes - prev_total_recorded_successes
        this_total_practice_successes = total_practice_successes - prev_total_practice_successes
        this_recorded_success_rate = this_total_recorded_successes / this_recorded_runs
        st.write('this_recorded_success_rate = ' + str(this_recorded_success_rate))

        print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
        st.write('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
        print('run %d recorded games out of an allotted   1,000. recorded success rate so far = %.3f' % (total_recorded_runs, success_rate))
        st.write('run %d recorded games out of an allotted   1,000. recorded success rate so far = %.3f' % (total_recorded_runs, success_rate))
        print('overall success rate = %.3f' % success_rate)
        st.write('overall success rate = %.3f' % success_rate)





if sidebar_selectbox_mode == "DEV_Bloomberg":
    # Bloomberg: Tickers, Fields: Live
    st.subheader("Bloomberg Tickers Fields Live")
    tickers = ['spx index','es1 index']
    fields_request = ['name','fut_cont_size','px_last']
    # bdp_ticker_fields = bdp.BDP()
    # bdp_ticker_fields.start_session()
    # bdp_ticker_fields.request(securities=tickers, field_id=fields_request)
    # df_bdp_ticker_fields = bdp_ticker_fields.results
    # st.write(df_bdp_ticker_fields.T)

    st.subheader("Bloomberg bdp")
    tickers = ['spx index','es1 index']
    fields_request = ['px_last']
    # bdp = bdp.BDP()
    # bdp.start_session()
    # bdp.request(securities=tickers, field_id=fields_request)
    # df_bdp = bdp.results
    # st.write(df_bdp.T)
    # for a in range(5):
    #     st.write("...")

    st.subheader("Bloomberg bdh")
    # tickers = ['spx index']
    # R_Govt_US_USD_20y_ETF_TLT US Equity
    # K_Corp_US_USD_IG_ETF_LQD US Equity
    # K_Corp_US_USD_HY_ETF_HYG US Equity
    # K_Corp_Swp_CDS_Bskt_EU_EUR_IG_SNRFIN CDSI GEN 5Y Corp
    # K_Corp_Swp_CDS_Bskt_NA_USD_IG_CDX IG CDSI GEN 5Y Corp
    tickers = [ 'ES1 INDEX',
                'NG1 COMDTY',
                # 'TLT US EQUITY',
                # 'LQD US EQUITY',
                # 'HYG US EQUITY',
                # 'SNRFIN CDSI GEN 5Y CORP',
                # 'CDX IG CDSI GEN 5Y CORP',
               ]
    # tickers = ['spx index', 'sx5e index', 'ukx index', 'dax index', 'nifty index', 'set index', 'sti index',
    #            'fbmklci index', 'hsi index', 'shcomp index', 'as51 index', 'kospi index', 'nky index', 'twse index',
    #            'cnymusd index']
    # to do: keep the order of the tickers
    # to do: reverse the dates to latest on top

    req = {
        'securities': tickers,
        'field_id': 'PX_LAST',
        'refdataservice': "HistoricalDataRequest",
        'periodicity': "DAILY",
        'start_date': (dt.datetime.today() - relativedelta(days=28)).strftime("%Y%m%d"),
        'end_date': dt.datetime.today().strftime("%Y%m%d")}
    all_req = []
    all_req.append(req)

    # for req in all_req:
    #     df_bdh = bdh.bbapi_bdh(securities=req['securities'],
    #                        field_id=req['field_id'],
    #                        refdataservice=req["refdataservice"],
    #                        periodicity=req["periodicity"],
    #                        start_date=req['start_date'],
    #                        end_date=req['end_date'])
    #
    #     if len(df_bdh.index.tolist()) < 1:
    #         df_bdh = pd.DataFrame(['Error'], columns=['Error'])

    # x = df_bdh.index
    # x_labels = list(df_bdh.index)
    # y1 = df_bdh["ES1 INDEX"]
    # y2 = df_bdh["NG1 COMDTY"]

    # fig, ax1 = plt.subplots(figsize=(10, 5))
    # plt.title('E.US.ES1 vs C.NG1')
    # plt.xticks(x, x_labels, rotation=45)
    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, color='b', label='E.ES1') # appear in legend box
    # ax2.plot(x, y2, color='g', label='C.NG1') # appear in legend box
    # ax1.set_facecolor('xkcd:grey')
    # ax2.set_facecolor('xkcd:grey')
    # ax1.set_ylabel('E.ES1', color='b')      # y-axis label
    # ax2.set_ylabel('C.NG1', color='g')      # y-axis label
    # ax1.legend(bbox_to_anchor=(0.88, 0.3))  # (right, up)
    # ax2.legend(bbox_to_anchor=(0.88, 0.7))  # (right, up)

    # plt.legend()
    # fig.patch.set_facecolor('grey')
    # plt.tight_layout()
    # st.pyplot(fig)

    # df_bdh_ri = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh,True,False,False)
    # df_bdh_rc = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh,False,True,False)
    # df_bdh_ric = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh,True,True,False)
    # df_bdh_ri_remio = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh, True, False, True)
    # df_bdh_ri_remio_date = PY_Sys_DataStructures.df_Reset_Index_Columns(df_bdh,True,False,True,'date')

    # st.write(df_bdh)
    # st.write(df_bdh_ri)
    # st.write(df_bdh_rc)
    # st.write(df_bdh_ric)
    # st.write(df_bdh_ri_remio)
    # st.write(df_bdh_ri_remio_date)
    # for a in range(5):
    #     st.write("...")

    # AgGrid(df_bdh_ri_remio_date, height=500, fit_columns_on_grid_load=False)

    #---------------------------------------------------------------------------------------
    st.subheader("Bloomberg bdp bdh xlsm csv")

    # Name of data set
    s_XXX = '_io_streamlit_bbg_bdp_bdh_in.xlsm'

    # p = path, n = name, e = ext
    s_p = ''
    s_p = s_p + os.getcwd()

    st.write("file in")
    try:
        from up_down import up, down
        df_bdp_bdh = down()
    except:
        # INPUT_FILEPATH = "H:\General\Wylie\DROPBOX\_\work\_io_streamlit_bbg_bdp_bdh_in.xlsm.csv"
        INPUT_FILEPATH = s_p + '\\' + s_XXX + '.csv'
        st.write('file_in = ' + INPUT_FILEPATH)
        df_bdp_bdh = pd.read_csv(INPUT_FILEPATH, index_col=0, header=0).dropna().drop_duplicates()
    df_bdp_bdh = df_bdp_bdh.drop_duplicates()
    df_bdp_bdh.columns = df_bdp_bdh.columns.str.upper()
    bdh_tickers_and_field = list(df_bdp_bdh.set_index('TICKER').itertuples())

    req_start_date = (dt.datetime.today() - relativedelta(years=10)).strftime("%Y%m%d")
    # req_start_date = (dt.datetime.today() - relativedelta(months=3)).strftime("%Y%m%d")
    req_end_date = dt.datetime.today().strftime("%Y%m%d")

    # bdh_tickers_and_field = [("zarjpy curncy", "last_price"),
    #                         ("EI959494 corp", "yld_ytm_mid"),
    #                         ("ai1 index", "last_price"),
    #                         ("cnymusd index", "px_last")]

    # fields_request = ['NAME', 'TRADING_DAY_START_TIME_EOD', 'TRADING_DAY_END_TIME_EOD', 'PRICE_LAST_TIME_RT', 'DATE_OFFSET', 'SECURITY_TYP', 'COUNTRY','LAST_PRICE']
    bdp_fields = ['NAME',
                  'TRADING_DAY_START_TIME_EOD',
                  'TRADING_DAY_END_TIME_EOD',
                  'PRICE_LAST_TIME_RT',
                  'SECURITY_TYP',
                  'COUNTRY',
                  'META_BDH_ALL_COUNT',
                  'META_BDH_ALL_RANGE',
                  'META_BDH_ALL_DT_START',
                  'META_BDH_ALL_DT_END',
                  'META_BDH_ALL_MIN',
                  'META_BDH_ALL_MAX',
                  'META_BDH_ALL_MEAN',
                  'META_BDH_ALL_STDEV',
                  'META_BDH_USR_COUNT',
                  'META_BDH_USR_RANGE',
                  'META_BDH_USR_DT_START',
                  'META_BDH_USR_DT_END',
                  'META_BDH_USR_MIN',
                  'META_BDH_USR_MAX',
                  'META_BDH_USR_MEAN',
                  'META_BDH_USR_STDEV',
                  'DATE_OFFSET',
                  'VALUE_LIVE'
                  ]

    # i convert everything to lower case at the end of the main function #lowercase_dataframe
    # [fr.lower() for fr in bdp_fields]

    # make adjustment to final table before exporting to csv
    # final_adj = [{"LAST_PRICE" : "Value_Live"}]

    final_adj = list(set(df_bdp_bdh.FIELD.str.upper().tolist()))

    # override specific bdp fields for individiual tickers
    # request_field_override = [{"TICKER": "EI959494 CORP", 'strOverrideField': 'LAST_PRICE', 'strOverrideValue': 'YLD_YTM_BID'}]


    # p = path, n = name, e = ext
    s_out_p = ''
    s_out_p = s_out_p + os.getcwd()  # leave output file in the same dir
    # s_out_p = s_out_p + os.getcwd() + '\_data'

    # n = name
    s_n = ''
    s_n_1 = s_n + s_XXX + "_{}".format('out')
    s_n_2 = s_n + s_XXX + "_{}_".format('out') + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # s_n = s_n + datetime.now().strftime("%Y%m%d_%H%M%S") + "_{}".format('bbg_bdp_bdh')

    # e = ext
    s_e_csv = ''
    s_e_csv = s_e_csv + '.csv'
    s_e_xlsx = ''
    s_e_xlsx = s_e_xlsx + '.xlsx'

    # pne = path name ext
    st.write("file out")
    s_out_pne_csv_1 = os.path.join(s_out_p, s_n_1 + s_e_csv)
    s_out_pne_csv_2 = os.path.join(s_out_p, s_n_2 + s_e_csv)
    s_out_pne_xlsx = os.path.join(s_out_p, s_n_1 + s_e_xlsx)
    st.write('file_out_1 = ' + s_out_pne_csv_1)
    st.write('file_out_2 = ' + s_out_pne_csv_2)
    st.write('start = ' + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # bbg_bdp_bdh.bbg_bdp_bdh_request(bdh_tickers_and_field, bdp_fields, req_start_date, req_end_date, s_out_pne_csv_1, s_out_pne_csv_2)
    st.write('end   = ' + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # for a in range(25):
    #     st.write("...")

    st.write("end")



if sidebar_selectbox_mode == "DEV_SQL":

    with st.echo():
        # GAMA: create connection to DB
        GAMA_engine = create_engine('mssql+pyodbc://GAMCTSQL01/RiskDB?driver=SQL+Server+Native+Client+11.0')

        GAMA_qry = """
        """
        GAMA_qry_EOD_List_Funds = """
        -- GAMA EOD List of Funds
        SELECT DISTINCT FundDescription, Fund, FundName
        FROM EOD.v_RiskExposure
        """
        GAMA_qry_EOD_List_Traders = """
        -- GAMA EOD List of Traders
        SELECT DISTINCT TraderKey, TraderCode, TraderName
        FROM EOD.v_RiskExposure
        ORDER BY TraderKey
        """
        GAMA_qry_EOD_List_Traders_Current = """
        -- GAMA EOD List of Traders (Current List)
        SELECT DISTINCT TraderKey, TraderCode, TraderName
        FROM EOD.v_RiskExposure
        WHERE TraderCode in ('JPRA','SWZ','ADM','CASH','CMA','SHL','EDL','RUD','SPC','CHU','CMC','ABR','JEDL')
        ORDER BY TraderKey
        """
        GAMA_qry_EOD_List_Tickers_For_Trader_CHU = """
        -- GAMA EOD List of Tickers for Trader
        SELECT DISTINCT GAMASecurityName, BBTicker, UnderlyingBBTicker
        FROM EOD.v_RiskExposure
        WHERE AsOfDate = (SELECT MAX(AsOfDate) FROM EOD.v_RiskExposure)
        AND TraderCode = 'CHU'
        """
        GAMA_qry_EOD_List_Tickers_For_Trader_ABR = """
        -- GAMA EOD List of Tickers for Trader
        SELECT DISTINCT GAMASecurityName, BBTicker, UnderlyingBBTicker
        FROM EOD.v_RiskExposure
        WHERE AsOfDate = (SELECT MAX(AsOfDate) FROM EOD.v_RiskExposure)
        AND TraderCode = 'ABR'
        """
        GAMA_qry_Live_List_Tickers_For_Trader_ABR = """
        -- GAMA Live List of Tickers for Trader
        SELECT DISTINCT GAMASecurityName, BBTicker, UnderlyingBBTicker
        FROM Intraday.v_Exposure
        WHERE TraderCode = 'ABR'
        """

# https://www.poems.com.sg/docs/poemsapi/

if sidebar_selectbox_mode == "_prd/_data_sg_cb_mas/_api_R_r_SG_SGD":
    # _prd/_data_sg_cb_mas/
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("_api_R_r_SG_SGD")
    try:
        csv_file_path = 'C:\Dropbox\_\work\_prd\_data_sg_cb_mas\_api_R_r_SG_SGD.csv'

        selected_option = st.sidebar.radio("Data:", ["_", "Get", "Read", "Parse"], index=0)

        if selected_option == "Get":
            toc.h2("Get")
            # C:\Dropbox\_\work\_prd\_data_sg_cb_mas\_api_R_r_SG_SGD_20240227.pdf
            # API Key: 5cc7f4c7-acbf-4bd1-9a2f-08a80d968ebf
            import urllib3
            http = urllib3.PoolManager()
            response = http.request('GET',
                                    'https://eservices.mas.gov.sg/apimg-gw/server/monthly_statistical_bulletin_non610mssql/domestic_interest_rates_daily/views/domestic_interest_rates_daily',
                                    headers={'keyid': '5cc7f4c7-acbf-4bd1-9a2f-08a80d968ebf'})
            response_json = response.json()
            response_json_df = pd.DataFrame(response_json)
            response_json_df.to_csv(csv_file_path, index = False)
            st.write(csv_file_path)
            st.write(dt.datetime.fromtimestamp(os.path.getmtime(csv_file_path)))
            st.write(os.path.getsize(csv_file_path))

            toc.h3("Parse: response_json = Dict")
            # st.write(type(response_json))
            # st.write(response_json)
            toc.h3("Parse: response_json = Dict.get('name')")
            # st.write(response_json.get('name'))
            toc.h3("Parse: response_json = Dict.get('elements')")
            # st.write(response_json.get('elements'))
            toc.h3("Parse: response_json = Dict.get('elements')[12311]")
            st.write(response_json.get('elements')[12311])
            toc.h3("Parse: response_json = Dict.get('elements')[12311].get('end_of_day')")
            st.write(response_json.get('elements')[12311].get('end_of_day'))
            toc.h3("Parse: response_json = Dict.get('elements')[12311].get('sora')")
            st.write(response_json.get('elements')[12311].get('sora'))
            toc.h3("Parse: response_json = Dict to List")
            list_records = response_json.get('elements', [])

            dates = [item.get('end_of_day') for item in list_records]
            rates = [item.get('sora') for item in list_records]
            df = pd.DataFrame({'Dates': dates, 'r_SGD_SORA': rates})
            df['Dates'] = pd.to_datetime(df['Dates'])
            df.set_index('Dates', inplace=True)
            df = df.dropna(subset=['r_SGD_SORA'])
            st.write(df)

            # Prices
            fig, ax = plt.subplots(figsize=(10, 6))  # streamlit plot: Add this at start
            plt.plot(df.r_SGD_SORA, marker='', linestyle='-', color='red', label='r_SGD_SORA')
            plt.xlabel('Date')
            plt.ylabel('Rate [%]')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, '.2f')))
            ax.set_facecolor('#333333')  # Set the background color to dark gray
            plt.title('SGD Overnight Rate Average (SORA)')
            plt.legend()
            plt.show()

            # Annotate the last point
            last_point_index = df.index[-1]
            last_point_value = df['r_SGD_SORA'].iloc[-1]
            plt.annotate(f"Date = {last_point_index.strftime('%Y-%m-%d')} \nRate [%] = {last_point_value}",
                         xy=(last_point_index, last_point_value ),
                         xytext=(last_point_index + pd.DateOffset(days=365), last_point_value + 0),
                         arrowprops=dict(facecolor='black', shrink=0.1),
                         )

            st.pyplot(fig)  # streamlit plot: Add this at end

        if selected_option == "Read":
            toc.h2("Read")
            df_csv = pd.read_csv(csv_file_path)
            json_df_csv = df_csv.to_dict(orient='records')
            st.write(json_df_csv)

        if selected_option == "Parse":
            toc.h2("Parse")
            df_csv = pd.read_csv(csv_file_path)
            json_df_csv = df_csv.to_dict(orient='records')
            toc.h2("Parse List")
            st.write(json_df_csv[0])
            toc.h2("Parse List Dict -> Strings")
            st.write(json_df_csv[0].get('name'))
            st.write(json_df_csv[0].get('elements'))
            toc.h2("Parse String to Dict")

            s_json = json_df_csv[0].get('elements')
            # st.write(type(s_json))
            st.write(s_json)

    except Exception as e:
        st.write(dt.datetime.now())
        st.write(e)
        fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
        fel.write(str(dt.datetime.now()))
        fel.write(' : ')
        fel.write(str(e))
        fel.write('  \n')
        fel.close()
    st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "_prd/_data_sg_cb_mas/_api_I_Life":
    # _prd/_data_sg_cb_mas/
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("_api_I_Life")
    with st.echo():
        try:
            # https://eservices.mas.gov.sg/apimg-portal/api-catalog-details/10581
            # https://eservices.mas.gov.sg/apimg-gw
            import urllib3
            http = urllib3.PoolManager()
            response = http.request('GET',
                                    'https://eservices-int.mas.gov.sg/monthly_statistical_bulletin_hist/ii_7_life_insurance_companies_new_business_annual/views/ii_7_life_insurance_companies_new_business_annual',
                                    headers={
                                        'keyid': '44be73c8-6a20-49b3-a005-010b692b6684'
                                    })
            st.write(response.data)
            st.write(response.data.decode('utf-8'))
            st.write(response.status)
            st.write(response.headers['Content-Type'])

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

# ==============================================================================================================================================
if sidebar_selectbox_mode == "PY_Template":
    # asdfasdf
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Template")
    with st.echo():
        try:
            st.write('Code Here')
            class MyClass:
                class_var_s = '___class_variable'
                class_var_dict = {'price':0, 'delta':0, 'gamma':0}

                def __init__(self,s):
                    self.class_var_s = s

                # dunder method
                # with this __str__ dunder method, it will print the return string, instead of object address
                def __str__(self):
                    s = '|'
                    s = s + f'class_var_s = {self.class_var_s}' + '|'
                    return s

                def get_class_var_s(self):
                    return self.class_var_s

            cls_obj = MyClass('hello')
            st.write(cls_obj.get_class_var_s())

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

# ==============================================================================================================================================
if sidebar_selectbox_mode == "PY_Template_Lite":
    # asdfasdf
    # create a TOC and put it in the sidebar
    toc = TOC()
    list_section = ["_", "Section 1"]
    section = st.sidebar.radio("Select", list_section)

    s = "Section 1"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write('Code Here')

# ==============================================================================================================================================
if sidebar_selectbox_mode == "Jelly_Belly":
    # asdfasdf
    # create a TOC and put it in the sidebar
    toc = TOC()
    list_section = ["_", "Music", "Math"]
    section = st.sidebar.radio("Select", list_section)


    s = "Math"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.markdown("""<style>body {background-color: pink;}</style>""", unsafe_allow_html=True, )
            st.write('Code Here')


# ==============================================================================================================================================
# FessorPro
# ==============================================================================================================================================
if sidebar_selectbox_mode == "PY_Virtual_Env":
    # asdfasdf
    # create a TOC and put it in the sidebar
    toc = TOC()
    list_section = ["_", "Section 1"]
    section = st.sidebar.radio("Select", list_section)

    s = "Python Interpreters"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write('Code Here')

    # Python Interpreter 9.6
    # Python Interpreter 9.11
    # \algo1
    # \algo1\python_interpreter_9.6_virtual_env
    # \algo2
    # \algo2\python_interpreter_9.11_virtual_env

    # s_pne_interpreter = C:\Python311\python.exe
    # s_pne_python_file = C:\Dropbox\_\work\_io_streamlit.py
    # s_pne_interpreter s_pne_python_file

    # C:\Dropbox\_\work\___lib_py>
    # C:\Dropbox\_\work\___lib_py>

    # C:\>python -m venv ve_01
    # C:\>/usr/bin/python3 -m venv ve_01

    # File, New

# ==============================================================================================================================================
csv_pne_SBIN = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\SBIN.csv'
csv_pne_sp500 = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\sp500.csv'
csv_pne_stock_value = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\stock_value.csv'
csv_pne_unicorn = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\\Unicorn_companies.csv'
# df_SBIN = pd.read_csv(csv_pne_SBIN)
# df_sp500 = pd.read_csv(csv_pne_sp500)
# df_stock_value = pd.read_csv(csv_pne_stock_value)
# df_unicorn = pd.read_csv(csv_pne_unicorn)
# ==============================================================================================================================================
import numpy as np
import pandas as pd
npa = np.arange(25).reshape(5, 5)
list_cols = ['c1', 'c2', 'c3', 'c4', 'c5']
list_rows = ['r1', 'r2', 'r3', 'r4', 'r5']
df = pd.DataFrame(npa, index=list_rows, columns=list_cols)
# ==============================================================================================================================================

if sidebar_selectbox_mode == "PY_Pandas":
    # create a TOC and put it in the sidebar
    toc = TOC()
    # toc.placeholder(sidebar=True)

    list_section = [
        "_",
        "DataFrame: Create: from npa",
        "DataFrame: Create: from list of list",
        "DataFrame: Create: from list of dict",
        "DataFrame: Create: from dict of list",
        "DataFrame: Create: from csv",
        "DataFrame: Create: from www: read_csv",
        "DataFrame: Create: from www: read_html",
        "DataFrame: View: head, tail",
        "DataFrame: View: info",
        "DataFrame: View: describe",
        "DataFrame: View: shape",
        "DataFrame: Class: Attribute: values = npa",
        "DataFrame: Class: Attribute: index, columns",
        "DataFrame: Class: Method: sort_values",
        "DataFrame: Apply: index",
        "DataFrame: Apply: index: set, reset",
        "DataFrame: Apply: [][]: Get 1 Column: as Series, as DataFrame",
        "DataFrame: Apply: [][]: Get list of n Columns",
        "DataFrame: Apply: loc",
        "DataFrame: Apply: iloc",
        "DataFrame: Compare: [][], loc, iloc",
        "DataFrame: Apply: iloc: Last 5 Rows, Last 3 Col",
        "DataFrame: Apply: iloc: All Rows, Some Cols",
        "DataFrame: Apply: []: Rows: Match Condition",
        "DataFrame: Apply: Add: Col",
        "DataFrame: Apply: Cols(pd.merge, df.join), Rows(pd.concat)",
        "DataFrame: Apply: drop: row(axis=0)",
        "DataFrame: Apply: drop: col(axis=1)",
        "DataFrame: Apply: drop_duplicates: row",
        "DataFrame: Apply: isna, notna",
        "DataFrame: Apply: drop_na",
        "DataFrame: Apply: fillna",
        "DataFrame: Apply: ffill, bfill",
        "DataFrame: Apply: Stats",
        "DataFrame: Apply: Strings: Accessor: .str",
        "DataFrame: Apply: Strings: Accessor: .str: first letter",
        "DataFrame: Apply: Strings: Accessor: .str: first word",
        "DataFrame: Apply: Strings: Accessor: .str.replace()",
        "DataFrame: Apply: Strings: Accessor: .str.upper()",
        "DataFrame: Apply: Strings: Accessor: .str.lower()",
        "DataFrame: Apply: Strings, DateTime",
        "DataFrame: Apply: Strings, DateTime: pd.to_datetime",
        "DataFrame: Apply: astype: float to int",
    ]
    section = st.sidebar.radio("Pandas", list_section)

    s = "DataFrame: Create: from npa"
    if section == s:
        toc.h1(s)
        with st.echo():
            npa = np.arange(9).reshape(3,3)
            st.write(npa)
        with st.echo():
            df = pd.DataFrame(npa)
            st.write(df)
        with st.echo():
            df.index = [11,22,33]
            df.columns = ['a','b','c']
            st.write(df)

    s = "DataFrame: Create: from list of list"
    if section == s:
        toc.h1(s)
        with st.echo():
            lol = [ [1,2,3], [4,5,6], [7,8,9] ]
            df_lol = pd.DataFrame(lol)
            st.write(df_lol)
        with st.echo():
            df_lol.index = [1,2,3]
            df_lol.columns = ['A','B','C']
            st.write(df_lol)

    s = "DataFrame: Create: from list of dict"
    if section == s:
        toc.h1(s)
        with st.echo():
            list_of_dict = [
                {"name":'aapl', "price":300, "sector":'tech'},
                {"name":'msft', "price":400, "sector":'tech'},
                {"name":'oily', "price":500, "sector":'oil'},
            ]
            df_lod = pd.DataFrame(list_of_dict)
            st.write(df_lod)

    s = "DataFrame: Create: from dict of list"
    if section == s:
        toc.h1(s)
        with st.echo():
            dict_of_list = {
                'name': ['aapl','msft','oily'],
                'price': [300,400,500],
                'sector': ['tech','tech','oil'],
            }
            df_dol = pd.DataFrame(dict_of_list)
            st.write(df_dol)

    s = "DataFrame: Create: from csv"
    if section == s:
        toc.h1(s)
        with st.echo():
            csv_pne_SBIN = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\SBIN.csv'
            df_SBIN = pd.read_csv(csv_pne_SBIN)
            st.write(df_SBIN.head())
            st.write(df_SBIN.tail())
        with st.echo():
            csv_pne_sp500 = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\sp500.csv'
            df_sp500 = pd.read_csv(csv_pne_sp500)
            st.write(df_sp500.head())
            st.write(df_sp500.tail())
        with st.echo():
            csv_pne_stock_value = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\stock_value.csv'
            df_stock_value = pd.read_csv(csv_pne_stock_value)
            st.write(df_stock_value.head())
            st.write(df_stock_value.tail())
        with st.echo():
            csv_pne_unicorn = 'C:\Dropbox\_\work\___lib_py_fessorpro\pandas files\pandas files\\Unicorn_companies.csv'
            df_unicorn = pd.read_csv(csv_pne_unicorn)
            st.write(df_unicorn.head())
            st.write(df_unicorn.tail())

    s = "DataFrame: Create: from www: read_csv"
    if section == s:
        toc.h1(s)
        with st.echo():
            url_csv = 'https://public.fyers.in/sym_details/NSE_CD.csv'
            df1 = pd.read_csv(url_csv)
            st.write(df1)

    s = "DataFrame: Create: from www: read_html"
    if section == s:
        toc.h1(s)
        with st.echo():
            url_wiki = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            list1 = pd.read_html(url_wiki)
            st.write(list1)
        with st.echo():
            df0 = list1[0]
            st.write(df0)
        with st.echo():
            df0['Date added'] = pd.to_datetime(df0['Date added'])
            st.write( df0[df0['Date added'] >= '2010-01-01'] )
        with st.echo():
            # how many companies in healthcare
            st.write(df0[df0['GICS Sector']=='Health Care'])
            st.write(df0[df0['GICS Sector']=='Health Care'].count()[0])
        with st.echo():
            # Companies with sector IT, founded after 2010
            # df0['Founded'] = df0['Founded'][:4]
            # st.write(df0[df0['GICS Sector'].value_counts())
            df0['Founded_4'] = df0['Founded'].str[:4].astype(int)
            st.write(df0)

            # st.write(
            #     df0[
            #         (df0['GICS Sector']=='Information Technology') &
            #         (df0['Founded'] > 2010)
            #         ]
            # )

        with st.echo():
            url_SPX = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            url_NFT = 'https://en.wikipedia.org/wiki/NIFTY_50'
            list_SPX = pd.read_html(url_SPX)
            list_NFT = pd.read_html(url_NFT)
            # st.write(list_SPX)
            # st.write(list_NFT)
            df_SPX = list_SPX[0][['Security','Symbol','GICS Sector']]
            df_NFT = list_NFT[2][['Company name','Symbol','Sector[18]']]
            dict_SPX_old_new = {
                'Security':'secu',
                'Symbol':'symbol',
                'GICS Sector':'sector'
            }
            dict_NFT_old_new = {
                'Company name':'secu',
                'Symbol':'symbol',
                'Sector[18]':'sector'
            }
            df_SPX.rename(columns=dict_SPX_old_new, inplace=True)
            df_NFT.rename(columns=dict_NFT_old_new, inplace=True)
            df = pd.concat([df_SPX, df_NFT])
            df.reset_index(inplace=True)
            # st.write(df.head())
            # st.write(df.tail())
            df_1 = df[df['secu'].str[0]=='A']
            df_1.reset_index(inplace=True)
            st.write(df_1.head())
            st.write(df_1.tail())

        with st.echo():
            df1 = list1[1]
            st.write(df1)
            df1.to_csv('C:\Dropbox\_\work\___lib_py_fessorpro\\newcomp.csv')


    s = "DataFrame: View: head, tail"
    if section == s:
        toc.h1(s)
        left, right = st.columns(2)
        with left:
            with st.echo():
                st.write(df_SBIN.head())
            with st.echo():
                st.write(df_SBIN.head(10))
        with right:
            with st.echo():
                st.write(df_SBIN.tail())
            with st.echo():
                st.write(df_SBIN.tail(10))

    s = "DataFrame: View: info"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_sp500.head())
        with st.echo():
            st.write(df_sp500.info)
        with st.echo():
            st.write(df_sp500.info())

    s = "DataFrame: View: describe"
    if section == s:
        toc.h1(s)
        st.write(df_sp500.head())
        st.write(df_sp500.tail())
        with st.echo():
            st.write(df_sp500.describe())

    s = "DataFrame: View: shape"
    if section == s:
        toc.h1(s)
        st.write(df_sp500.head())
        st.write(df_sp500.tail())
        with st.echo():
            st.write(df_sp500.shape)

    s = "DataFrame: Class: Attribute: values = npa"
    if section == s:
        toc.h1(s)
        st.write(df_sp500.head())
        st.write(df_sp500.tail())
        with st.echo():
            st.write(df_sp500.values)
            # st.write(type(df_sp500.values)) # ndarray

    s = "DataFrame: Class: Attribute: index, columns"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_sp500.index)
            # st.write(type(df_sp500.index))
        with st.echo():
            st.write(df_sp500.columns)
            # st.write(type(df_sp500.columns))

    s = "DataFrame: Class: Method: sort_values"
    if section == s:
        toc.h1(s)
        with st.echo():
            left, right = st.columns(2)
            with left:
                st.write(df_sp500.sort_values('Price'))
            with right:
                st.write(df_sp500.sort_values('Price',ascending=False))

    s = "DataFrame: Apply: index"
    if section == s:
        toc.h1(s)
        st.write(df_unicorn.head())
        st.write(df_unicorn.tail())
        with st.echo():
            st.write(df_unicorn['Company'][2])
        with st.echo():
            df_unicorn.set_index('Company',inplace=True)
            st.write(df_unicorn.head())
            st.write(df_unicorn.tail())
        with st.echo():
            st.write(df_unicorn['Year Founded']['SHEIN'])
            st.write(df_unicorn['Country']['Zopa'])

    s = "DataFrame: Apply: index: set, reset"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_unicorn)
        with st.echo():
            df_unicorn.set_index('Company',inplace=True)
            st.write(df_unicorn)
        with st.echo():
            df_unicorn.reset_index(inplace=True)
            st.write(df_unicorn)

    s = "DataFrame: Apply: [][]: Get 1 Column: as Series, as DataFrame"
    if section == s:
        toc.h1(s)
        with st.echo():
            c1,c2,c3 = st.columns(3)
            with c1:
                st.write(df_sp500.Name)
                st.write(type(df_sp500.Name).__name__)
            with c2:
                st.write(df_sp500['Name'])
                st.write(type(df_sp500['Name']).__name__)
            with c3:
                st.write(df_sp500[['Name']])
                st.write(type(df_sp500[['Name']]).__name__)

    s = "DataFrame: Apply: [][]: Get list of n Columns"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_sp500[['Name','EPS']].head())
            st.write(df_sp500[['Name','EPS']].tail())

    s = "DataFrame: Apply: loc"
    if section == s:
        toc.h1(s)
        df_unicorn.set_index('Company', inplace=True)
        st.write(df_unicorn)
        with st.echo():
            st.write(df_unicorn.loc['SpaceX','Date Joined'])

    s = "DataFrame: Apply: iloc"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_unicorn.tail())
        with st.echo():
            st.write(df_unicorn.iloc[-1,-2])

    s = "DataFrame: Compare: [][], loc, iloc"
    if section == s:
        toc.h1(s)
        st.write(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("df['c4']['r2']")                  # [][]: col, row
            st.write(df['c4']['r2'])                    # [][]: col, row

            st.write("df[['c3','c4']]['r3':'r4']")      # [][]: col list, row slicing
            st.write(df[['c3','c4']]['r3':'r4'])        # [][]: col list, row slicing

        with col2:
            st.write("df.loc['r2','c4']")               # loc : row, col
            st.write(df.loc['r2','c4'])                 # loc : row, col

            st.write("df.loc[['r3','r4'],['c3','c4']]") # loc : col, row
            st.write(df.loc[['r3','r4'],['c3','c4']])   # loc : col, row

            st.write("df.loc['r3':'r4','c3':'c4']")     # loc : col, row: slicing
            st.write(df.loc['r3':'r4','c3':'c4'])       # loc : col, row: slicing

        with col3:
            st.write("df.iloc[1,3]")                    # iloc: row, col
            st.write(df.iloc[1,3])                      # iloc: row, col

            st.write("df.iloc[[2,3],[2,3]]")            # iloc: row, col
            st.write(df.iloc[[2,3],[2,3]])              # iloc: row, col

            st.write("df.iloc[2:3+1,2:3+1]")            # iloc: row, col: slicing
            st.write(df.iloc[2:3+1,2:3+1])              # iloc: row, col: slicing

    s = "DataFrame: Apply: iloc: Last 5 Rows, Last 3 Col"
    if section == s:
        toc.h1(s)
        st.write(df_unicorn)
        with st.echo():
            st.write(df_unicorn.iloc[-5:, -3:])

    s = "DataFrame: Apply: iloc: All Rows, Some Cols"
    if section == s:
        toc.h1(s)
        st.write(df)
        with st.echo():
            st.write(df.iloc[:, 1:-1])  # iloc: all row, some col: slicing

    s = "DataFrame: Apply: []: Rows: Match Condition"
    if section == s:
        toc.h1(s)

        toc.h2("df: get rows conditional: where sector is health care")
        with st.echo():
            st.write(df_sp500[df_sp500['Sector'] == 'Health Care'])

        toc.h2("df: get rows conditional: where EPS > 20")
        with st.echo():
            st.write(df_sp500[df_sp500['EPS'] > 20])

        toc.h2("df: get rows conditionals: where Price > 1000, EPS > 20")
        with st.echo():
            st.write(df_sp500[ (df_sp500['Price']>1000) & (df_sp500['EPS']>20) ])
            st.write(df_sp500[ (df_sp500['Price']>1000) & (df_sp500['EPS']>20) ][['Name','Price','EPS']])

        toc.h2("df: get rows conditionals: founded > 2015")
        with st.echo():
            st.write(df_unicorn[df_unicorn['Year Founded']>2015])

        toc.h2("df: get rows conditionals: founded > 2015, continent is north america")
        with st.echo():
            st.write(
                df_unicorn[
                    (df_unicorn['Year Founded']>2015) &
                    (df_unicorn['Continent']=='North America')
                    ]
            )

        toc.h2("df: get rows conditionals: founded > 2015, continent is north america, valuation between 10 and 13")
        with st.echo():
            st.write(
                df_unicorn[
                    (df_unicorn['Year Founded']>2015) &
                    (df_unicorn['Continent']=='North America') &
                    (10 <= df_unicorn['Valuation']) &
                    (df_unicorn['Valuation'] <= 13)
                    ]
            )

        toc.h2("df: get rows conditionals: founded > 2015, continent is north america, valuation between 10 and 13, city in ny or pittsburgh")
        with st.echo():
            st.write(
                df_unicorn[
                    (df_unicorn['Year Founded']>2015) &
                    (df_unicorn['Continent']=='North America') &
                    (10 <= df_unicorn['Valuation']) &
                    (df_unicorn['Valuation'] <= 13) &
                    df_unicorn['City'].isin(['New York','Pittsburgh'])
                    ]
            )

    s = "DataFrame: Apply: Add: Col"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_stock_value)
            df_stock_value['sum'] = df_stock_value['goog'] - df_stock_value['amzn'] + df_stock_value['tsla']
            st.write(df_stock_value)

    s = "DataFrame: Apply: Cols(pd.merge, df.join), Rows(pd.concat)"
    if section == s:
        toc.h1(s)
        df11 = pd.DataFrame(np.arange(6).reshape(3, 2) + 1 + 0, index=[1, 2, 3], columns=['A', 'B'])
        df12 = pd.DataFrame(np.arange(6).reshape(3, 2) + 1 + 6, index=[1, 2, 3], columns=['C', 'D'])
        df21 = pd.DataFrame(np.arange(6).reshape(3, 2) + 1 + 6, index=[4, 5, 6], columns=['A', 'B'])
        df11 = df11.rename_axis('index_id')
        df12 = df12.rename_axis('index_id')
        df21 = df21.rename_axis('index_id')
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.echo():
                st.write(df11)
            with st.echo():
                st.write(df21)
            with st.echo():
                st.write(pd.concat([df11, df21]))
        with c2:
            with st.echo():
                st.write(df12)
        with c3:
            with st.echo():
                st.write(pd.merge(df11,df12,on='index_id'))
        with c4:
            with st.echo():
                st.write(df11.join(df12))

    s = "DataFrame: Apply: drop: row(axis=0)"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_unicorn.head())
        with st.echo():
            df_unicorn.drop(2,axis=0,inplace=True)
            st.write(df_unicorn.head())
        with st.echo():
            df_unicorn.set_index('Company',inplace=True)
            st.write(df_unicorn.head())
        with st.echo():
            df_unicorn.drop('Stripe',axis=0,inplace=True)
            st.write(df_unicorn.head())

    s = "DataFrame: Apply: drop: col(axis=1)"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_unicorn.head())
        with st.echo():
            df_unicorn.drop('Industry',axis=1,inplace=True)
            st.write(df_unicorn.head())
        with st.echo():
            df_unicorn.drop(['City','Continent'],axis=1,inplace=True)
            st.write(df_unicorn.head())

    s = "DataFrame: Apply: drop_duplicates: row"
    if section == s:
        toc.h1(s)
        df = pd.DataFrame([[1,2],[3,2],[5,6],[7,2]], index=[1,2,3,4], columns=['A','B'])
        with st.echo():
            st.write(df)
        with st.echo():
            df.drop_duplicates('B',inplace=True)    # keep every first unique value in col 'B'
            st.write(df)

    s = "DataFrame: Apply: isna, notna"
    if section == s:
        toc.h1(s)
        df = pd.DataFrame([[1,2],[3,],[5,6],[7,2]], index=[1,2,3,4], columns=['A','B'])
        st.write(df)
        na_ys, na_no = st.columns(2)
        with na_ys:
            with st.echo():
                st.write(df.isna())
        with na_no:
            with st.echo():
                st.write(df.notna())
        with na_ys:
            with st.echo():
                st.write(df.isna().sum(axis=0))
            with st.echo():
                st.write(df.isna().sum(axis=1))
        with na_no:
            with st.echo():
                st.write(df.notna().sum(axis=0))
            with st.echo():
                st.write(df.notna().sum(axis=1))

    s = "DataFrame: Apply: drop_na"
    if section == s:
        toc.h1(s)
        df = pd.DataFrame([[1,2,3],[4,np.nan,5],[6,7,8],[np.nan,10,11]], index=[1,2,3,4], columns=['A','B','C'])
        st.write(df)
        with st.echo():
            df_dropna_byRow = df.dropna(axis=0)     # remove any row with nan values
            st.write(df_dropna_byRow)
        with st.echo():
            df_dropna_byCol = df.dropna(axis=1)     # remove any col with nan values
            st.write(df_dropna_byCol)

    s = "DataFrame: Apply: fillna"
    if section == s:
        toc.h1(s)
        df = pd.DataFrame([[1,2,3],[4,np.nan,5],[6,7,8],[np.nan,10,11]], index=[1,2,3,4], columns=['A','B','C'])
        st.write(df)
        with st.echo():
            df_fillna = df.fillna(-1)     # replace nan with -1
            st.write(df_fillna)

    s = "DataFrame: Apply: ffill, bfill"
    if section == s:
        toc.h1(s)
        df = pd.DataFrame([[1,2,3],[4,np.nan,5],[6,7,8],[np.nan,10,11]], index=[1,2,3,4], columns=['A','B','C'])
        st.write(df)
        with st.echo():
            df_ffill = df.ffill()    # forward fill = replace nan with previous row's value
            st.write(df_ffill)
        with st.echo():
            df_bfill = df.bfill()    # back fill = replace nan with next row's value
            st.write(df_bfill)

    s = "DataFrame: Apply: Stats"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(df_sp500.head())
        with st.echo():
            st.write(df_sp500.head()['Price'].mean())
        with st.echo():
            st.write(df_sp500.head()['Price'].median())
        with st.echo():
            st.write(df_sp500.head()['Price'].mode()[0])
        with st.echo():
            st.write(df_sp500.head()['Price'].min())
        with st.echo():
            st.write(df_sp500.head()['Price'].max())
        with st.echo():
            st.write(df_sp500.head()['Price'].var())
        with st.echo():
            st.write(df_sp500.head()['Price'].std())
        with st.echo():
            st.write(df_sp500['EPS'].value_counts())    # like histogram table

    s = "DataFrame: Apply: Strings: Accessor: .str: first letter"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            st.write(df)
        with st.echo():
            df['Name_1'] = df['Name'].str[0]
            st.write(df)

    s = "DataFrame: Apply: Strings: Accessor: .str: first word"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            df['Name_word_1'] = df['Name'].str.split(" ").str[0]
            st.write(df)

    s = "DataFrame: Apply: Strings: Accessor: .str.replace()"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            df['s_name'] = df['Name'].str.replace(' ','_')
            st.write(df)

    s = "DataFrame: Apply: Strings: Accessor: .str.upper()"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            df['s_name_upper'] = df['Name'].str.upper()
            st.write(df)

    s = "DataFrame: Apply: Strings: Accessor: .str.lower()"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            df['s_name_lower'] = df['Name'].str.lower()
            st.write(df)

    s = "DataFrame: Apply: Strings, DateTime"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_stock_value.copy()
            st.write(df.head())
        with st.echo():
            st.write(type(df['Date'][0]).__name__) # String
            st.write(df.head())
        with st.echo():
            df['Date'] = pd.to_datetime(df['Date'])
            st.write(type(df['Date'][0]).__name__) # DateTime
            st.write(df.head())
        with st.echo():
            df['MthNum'] = df['Date'].dt.month
            df['MthStr'] = df['Date'].dt.month_name()
            df['DayNum'] = df['Date'].dt.day
            df['DayStr'] = df['Date'].dt.day_name()
            st.write(df.head())

    s = "DataFrame: Apply: Strings, DateTime: pd.to_datetime"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_SBIN.copy()
            st.write(df.head())
            df.rename(columns={'date':'DateTime'}, inplace=True)
            st.write(df.head())
            st.write(type(df['DateTime'][0]).__name__)  # String
        with st.echo():
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            st.write(df.head())
            st.write(type(df['DateTime'][0]).__name__)
        with st.echo():
            df['Date'] = df['DateTime'].dt.date
            df['Time'] = df['DateTime'].dt.time
            st.write(df.head())

    s = "DataFrame: Apply: astype: float to int"
    if section == s:
        toc.h1(s)
        with st.echo():
            df = df_sp500.copy()
            st.write(df)
            df = df.astype({'EPS':int})     # float gets rounded down to int
            st.write(df)

# ==============================================================================================================================================

if sidebar_selectbox_mode == "PY_NumPy":
    toc = TOC()
    list_section = [
        "_",
        "npa_1d: Indexing",
        "npa_2d: Indexing",
        "npa_2d: generate: reshape: Indexing",
        "npa_3d: Indexing",
        "npa: Generate: zeros",
        "npa: Generate: ones",
        "npa: Generate: arange",
        "npa: Generate: random: reshape",
        "npa_1d: from list",
        "npa_2d: from list of list",
        "npa_3d: from list of list of list"
    ]
    section = st.sidebar.radio("Select", list_section)

    s = "npa_1d: Indexing"
    if section == s:
        toc.h1(s)
        with st.echo():
            npa_1d = np.arange(11,11*6,11,dtype=int)
            st.write(npa_1d)
        with st.echo():
            st.write(npa_1d[3])
        with st.echo():
            st.write(npa_1d[-2])

    s = "npa_2d: Indexing"
    if section == s:
        toc.h1(s)

        with st.echo():
            lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 1 list of 3 lists of 3 elements
            npa_2d_3x3 = np.array(lol)
            st.write(npa_2d_3x3)

        col_get_subrow, col_get_subcol = st.columns(2)
        with col_get_subrow:
            with st.echo():
                st.write(npa_2d_3x3[1,1:3])
            with st.echo():
                st.write(npa_2d_3x3[1][1:3])
            with st.echo():
                st.write(npa_2d_3x3[1][[1,2]])

        with col_get_subcol:
            with st.echo():
                st.write(npa_2d_3x3[1:3,1])
            with st.echo():
                st.write(npa_2d_3x3[1][1])
                st.write(npa_2d_3x3[2][1])
            with st.echo():
                st.write([npa_2d_3x3[1,1],npa_2d_3x3[2,1]])

    s = "npa_2d: generate: reshape: Indexing"
    if section == s:
        toc.h1(s)
        with st.echo():
            npa_2d_5x5 = np.arange(25).reshape(5,5)
        with st.echo():
            st.write(npa_2d_5x5)
        with st.echo():
            st.write(npa_2d_5x5[2,1:4])
        with st.echo():
            st.write(npa_2d_5x5 * np.eye(5))
        with st.echo():
            st.write(npa_2d_5x5[[0,1,2,3,4],[0,1,2,3,4]])
            st.write(npa_2d_5x5[[0,1,2,3,4],[4,3,2,1,0]])

    s = "npa_3d: Indexing"
    if section == s:
        toc.h1(s)
        with st.echo():
            lolol = [ [[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]] ] # 1 list of 2 lists of 3 lists
            npa_3d = np.array(lolol)
            st.write(npa_3d)
        with st.echo():
            st.write(npa_3d[1][1][1])
        with st.echo():
            st.write(npa_3d[-1][-2][-1])
        with st.echo():
            st.write(npa_3d[1,1,1])

    s = "npa: Generate: zeros"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(np.zeros(5,dtype=int))

    s = "npa: Generate: ones"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(np.ones(5,dtype=int))

    s = "npa: Generate: arange"
    if section == s:
        toc.h1(s)
        with st.echo():
            st.write(np.arange(1,5,dtype=int))
            st.write(np.arange(5,10,2,dtype=int))

    s = "npa: Generate: random: reshape"
    if section == s:
        toc.h1(s)
        with st.echo():
            npa_rand_2x5 = np.random.randint(100, 200, 10).reshape(2, 5)
            st.write(npa_rand_2x5)

    s = "npa_1d: from list"
    if section == s:
        toc.h1(s)
        with st.echo():
            col_list, col_npa = st.columns(2)

            with col_list:
                toc.h2("list: operators applied to list")
                list_mix = [1,2,3,'asdf',[],{}]  # many data types, takes up memory
                list_int = [1,2,3]
                npa_int = np.array(list_int)
                st.write('list_int')
                st.write(list_int)
                st.write('list_int * 3')
                st.write(list_int * 3)
                try:
                    st.write('list_int + 1')
                    st.write(list_int + 1)
                except Exception as e:
                    st.write(e)

            with col_npa:
                toc.h2("npa: operators applied to list elements")
                import numpy as np
                st.write('npa_int')
                st.write(npa_int)
                st.write('npa_int * 3')
                st.write(npa_int * 3)
                st.write('npa_int + 1')
                st.write(npa_int + 1)

    s = "npa_2d: from list of list"
    if section == s:
        toc.h1(s)
        with st.echo():
            lol = [[1,2,3],[4,5,6]]
            npa_2d = np.array(lol)
            st.write(npa_2d)

    s = "npa_3d: from list of list of list"
    if section == s:
        toc.h1(s)
        with st.echo():
            lolol = [ [[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]] ] # 1 list of 2 lists of 3 lists
            npa_3d = np.array(lolol)
            st.write(npa_3d)




if sidebar_selectbox_mode == "PY_DateTime_Holiday":
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("DateTime: Holidays: FINRA")
    with st.echo():
        # URL of the website with the table
        url = "https://www.finra.org/filing-reporting/market-transparency-reporting/holiday-calendar"

        # Read HTML tables from the URL
        list_tables = pd.read_html(url)

        # Assume the table of interest is the first table on the page
        toc.h2("Get DataFrame of Dates and Holidays")
        if list_tables:
            df_hols = list_tables[0]    # Select the first table
            dict_col_old_new = {0: 'DateString', 1: 'Holiday'}
            df_hols.rename(columns=dict_col_old_new, inplace=True)
            st.write(df_hols)           # display
        else:
            st.write("No tables found on the page.")

        toc.h2("Get List of DateStrings")
        list_s_hols = df_hols['DateString'].tolist()
        st.write(list_s_hols)

        list_dto_hols = []
        toc.h2("Get List of DateTime Objects")
        for i in range(len(list_s_hols)):
            # list_s_hols[i] = list_s_hols[i].strip()
            dto = datetime.datetime.strptime(list_s_hols[i],'%A, %B %d, %Y')
            list_dto_hols.append(dto)
        st.write(list_dto_hols)

    toc.generate()




if sidebar_selectbox_mode == "PY_DateTime":
    toc = TOC()
    list_section = [
            "_",
            "DateTime",
            "DateTime: Import",
            "DateTime: Attributes",
            "DateTime: to String: strftime = String Format Time",
            "DateTime: fr String: strptime = String Parse Time",
            "DateTime: fr Epoch: fromtimestamp(ep)",
            "Epoch: fr DateTime: .timestamp()",
            "DateTime, Date, Time, TimeDelta",
            "DateTime: Exercise"
    ]
    section = st.sidebar.radio("Select", list_section)

    # =====================================================================================
    # DateTime Representations
    # 1) string
    # 2) DateTime object
    # 3) Epoch = int
    # =====================================================================================

    s = "DateTime"
    if section == s:
        toc.h1(s)
        with st.echo():
            import datetime
            st.write(datetime.datetime.now())
            st.write(datetime.datetime.now().timestamp())
        with st.echo():
            import time
            st.write(time.time())


    s = "DateTime: Import"
    if section == s:
        toc.h1(s)
        with st.echo():
            import datetime
            dto_0 = datetime.datetime(2024,3,2,1,2,3)
            st.write(dto_0)
        with st.echo():
            import datetime as dt
            dto_1 = dt.datetime(2024, 3, 2, 1, 2, 3)
            st.write(dto_1)
        with st.echo():
            from datetime import datetime
            dto_2 = datetime(2024, 3, 2, 1, 2, 3)
            st.write(dto_2)
        with st.echo():
            from datetime import *
            dto_3 = datetime(2024, 3, 2, 1, 2, 3)
            st.write(dto_3)

    s = "DateTime: Attributes"
    if section == s:
        toc.h1(s)
        with st.echo():
            dto = datetime.datetime(2024,3,2,10,20,30)
            st.write(dto)
        with st.echo():
            st.write(dto.year)
        with st.echo():
            st.write(dto.month)
        with st.echo():
            st.write(dto.day)
        with st.echo():
            st.write(dto.weekday())                 # 0-6 = Mon-Sun
            st.write(dto.strftime("%a"))
            st.write(dto.strftime("%A"))
        with st.echo():
            st.write(dto.hour)
        with st.echo():
            st.write(dto.minute)
        with st.echo():
            st.write(dto.second)

    s = "DateTime: to String: strftime = String Format Time"
    if section == s:
        toc.h1(s)
        with st.echo():
            import datetime
            dt = datetime.datetime.now()
            st.write(dt)
        with st.echo():
            s_dt = dt.strftime('%Y%m%d_%H%M%S')
            st.write(s_dt)

    s = "DateTime: fr String: strptime = String Parse Time"
    if section == s:
        toc.h1(s)
        with st.echo():
            import datetime
            s = '2024/03/02 Sat 01:23:45'
            st.write(s)
            st.write(s[:4])
        with st.echo():
            sf = '%Y/%m/%d %a %H:%M:%S'
            dt_s_sf = datetime.datetime.strptime(s,sf)
            st.write(dt_s_sf)
            st.write(dt_s_sf.year)

    s = "DateTime: fr Epoch: fromtimestamp(ep)"
    if section == s:
        toc.h1(s)
        with st.echo():
            ep = 1709313338.234497
            dt = datetime.datetime.fromtimestamp(ep)
            st.write(dt)
        with st.echo():
            dt_sgt = datetime.datetime.fromtimestamp(ep)
            st.write(dt_sgt) # sgt = utc + 08:00
        with st.echo():
            dt_utc = datetime.datetime.fromtimestamp(ep,tz=datetime.timezone.utc)
            st.write(dt_utc)

    s = "Epoch: fr DateTime: .timestamp()"
    if section == s:
        toc.h1(s)
        with st.echo():
            dt = datetime.datetime.now()
            st.write(dt)
        with st.echo():
            ep_dt = dt.timestamp()
            st.write(ep_dt)

    s = "DateTime, Date, Time, TimeDelta"
    if section == s:
        toc.h1(s)
        with st.echo():
            from datetime import datetime, date, time, timedelta
            dto = datetime.now()
            st.write(dto)
            st.write(dto + timedelta(days=1))
            st.write(dto + timedelta(minutes=1441))

    s = "DateTime: Exercise"
    if section == s:
        toc.h1(s)

        toc.h2("Exercise: Print all days in 2024, Mar")
        with st.echo():
            from datetime import *
            dto = datetime(2024, 3, 1)
            while True:
                if dto.year == 2024 and dto.month == 4:
                    break
                st.write(dto)
                dto = dto + timedelta(days=1)

        toc.h2("Exercise: Print all days in 2024, Mar, if Sat/Sun")
        with st.echo():
            from datetime import *
            dto = datetime(2024, 3, 1)
            while True:
                if dto.year == 2024 and dto.month == 4:
                    break
                if dto.weekday() == 5 or dto.weekday() == 6:
                    st.write(f"{dto}, {dto.strftime('%a')}")
                dto = dto + timedelta(days=1)


if sidebar_selectbox_mode == "PY_Environment_Virtual":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Virtual Environment")
    with st.echo():
        try:
            # VS Code can have more than 1 python version interpreter
            st.write(sys.version)

            # Broker Fyers | 2019 | 3.9
            # Broker IBKR  | 2019 | 3.9
            # ib insync
            # zerodha
            # pykiteconnect

            pass

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Module":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Package: Install, Uninstall")
        # requirement.txt
        # virtual environment

    toc.h1("Module: Standard: datetime")
    with st.echo():
        import datetime
        dt = datetime.datetime(2023,12,31,23,59,59)
        st.write(dt)
        dt2 = dt + datetime.timedelta(days=10)
        st.write(dt2)

    toc.h1("Module: Standard: time")
    with st.echo():
        import time
        st.write('start: (sleep 1 sec)')
        time.sleep(1)
        st.write('end')

    toc.h1("Module: Standard: os")
    with st.echo():
        import os
        st.write(os.getcwd())

    toc.h1("Module: Standard: sys")
    with st.echo():
        import sys
        # st.write(sys.exit()) # stop the program
        pass

    toc.h1("Module: User Defined")
    with st.echo():
        import ___lib_py.PY_Math as PY_Math_Test
        st.write(PY_Math_Test.is_prime(18))
        st.write(PY_Math_Test.is_prime(19))

    toc.h1("Module: User Defined: Fin_Deriv")
    with st.echo():
        # C:\Dropbox\_\work\Fin_Deriv\_Class_Fin_Deriv.py
        # C:\Dropbox\_\work\_io_streamlit.py
        import Fin_Deriv._Class_Fin_Deriv as cls_fd
        obj_E_Stk_US_MSFT   = cls_fd._Class_Fin_Deriv('E','Stk','MSFT US Equity',{'price': 100, 'delta': 1, 'gamma': 0})
        obj_R_B_Govt_US_01y = cls_fd._Class_Fin_Deriv('R','B_Govt','___XXXXX Corp',{'price': 95, 'delta': 0, 'gamma': 0})
        st.write(obj_E_Stk_US_MSFT)
        st.write(obj_R_B_Govt_US_01y)

    toc.h2("Module: User Defined: Fin_Deriv: Call Class Method")
    with st.echo():
        st.write('Call a Class Method')
        f_ans = cls_fd.get_float_power(2.1,3)
        st.write(f_ans)
        s_ans = cls_fd.get_str_power(2.1,3)
        st.write(s_ans)

    toc.generate()

if sidebar_selectbox_mode == "PY_Class_Object__Broker":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Broker")
    try:
        class Broker:
            market = 'NYSE'
            stock_prices = {'aapl':100, 'msft':200, 'ibm':300, 'amzn':400, 'fb':500}

            # Constructor
            # dunder method
            def __init__(self, name, id, password, balance):
                # Instance Attributes
                self.name = name
                self.id = id
                self.password = password
                self.balance = balance
                self.portfolio = []

            # dunder method
            # whenever object is printed, it will print address
            # with this __str__ dunder method, it will print the return string
            def __str__(self):
                s = '|'
                s = s + f'Name      = {self.name}' + '|'
                s = s + f'Balance   = {self.balance}' + '|'
                s = s + f'Portfolio = {self.portfolio}' + '|'
                return s

            # Method
            def about_me(self):
                return f'My name is {self.name} and my balance is {self.balance}'

            def get_portfolio(self):
                return self.portfolio

            def buy(self,s_name):
                #  if stock exists, get the price
                stock_price_found = self.stock_prices.get(s_name)
                if stock_price_found:
                    # if balance enough
                    if self.balance >= stock_price_found:
                        # buy
                        self.portfolio.append(s_name)
                        self.balance = self.balance - stock_price_found
                        return f'{self.name} bought {s_name}'
                    else:
                        return 'not enough balance'
                    # add to portfolio
                    # remove
                else:
                    return f'{s_name} not in stock_prices'

            def sell(self,s_name):
                # if stock exists in user portfolio
                if s_name in self.portfolio:
                    self.portfolio.remove(s_name)
                    self.balance = self.balance + self.stock_prices[s_name]
                    return f'sold {s_name}'
                else:
                    return f'{s_name} not in {self.name} portfolio'

        toc.h2("Create Object")
        u1 = Broker('User1','1','aaa',20000)
        u2 = Broker('User2','2','bbb',17000)
        st.write(u1)
        st.write(u2)

        toc.h2("self")
        st.write(u1.about_me())         # the instance calls the method, passing itself into the method
        st.write(Broker.about_me(u1))   # the instance is passed in as "self"

        toc.h2("Mkt")
        st.write(Broker.stock_prices)

        toc.h2("buy KO")
        st.write(u1)
        st.write(u1.buy('apple'))

        toc.h2("sell KO")
        st.write(u1)
        st.write(u1.sell('aapl'))

        toc.h2("buy OK, sell OK")
        st.write(u1)
        st.write(u1.buy('aapl'))
        st.write(u1)
        st.write(u1.sell('aapl'))
        st.write(u1)

    except Exception as e:
        st.write(dt.datetime.now())
        st.write(e)
        fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
        fel.write(str(dt.datetime.now()))
        fel.write(' : ')
        fel.write(str(e))
        fel.write('  \n')
        fel.close()


    toc.generate()


if sidebar_selectbox_mode == "PY_Class_Object__Constructor":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Class Student")
    with st.echo():
        try:
            class Student:
                dress_code = 'formal'
                shoe_color = 'black'

                # constructor = method that is executed when new object is created
                def __init__(self, in_name, in_email, in_phone):
                    self.name = in_name
                    self.email = in_email
                    self.phone = in_phone
                    self.wallet = 0

                def get_avg_grades(self,grades):
                    return sum(grades)/len(grades)

            s1 = Student('Wylie','wylie.chan@gmail.com','+6591724555')
            s2 = Student('Jaehee','jaehee0819@gmail.com','+6591724111')

            st.write(s1.dress_code)
            st.write(s1.shoe_color)
            st.write(s1.get_avg_grades([55,65,75]))

            # Add some object/instance attributes
            s1.name = 'Adam'


        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Class_Object__Attribute_Method":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Classes and Objects: Intro")
    with st.echo():
        try:
            # Structural Programming
            # Object-Oriented Programming

            i=10
            s='hello'
            l=[1,2,3]
            d={'a':1,'b':2,'c':3}
            # st.write(i,s,l,d)
            # st.write(type(i),type(s),type(l),type(d))

            # Class = Blueprint of an Object
            # Object = Instance of Class


            class Building:
                # class attributes
                shape = 'cube'
                design = 'greek'
                def name_of_building(self,name):
                    return 'building name is: ' + name
                def area(self,length, breadth):
                    return length * breadth

            # objects
            b1 = Building()
            b2 = Building()

            # access the object's attributes
            st.write('b1.shape ---')
            st.write(b1.shape)
            st.write('b1.design ---')
            st.write(b1.design)

            # object attribute = instance attribute
            b2.material = 'steel'
            st.write(b2.material)
            st.write(b2.name_of_building('LPV'))
            st.write(b1.area(10,20))

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Files_Import_py_Library_Package_Module":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Library = Package = Module")
    with st.echo():
        try:
            # Interpreter
            #   PSL = Python Standard Library
            #       os, time, random
            #   Download using pip
            #
            import random
            import os
            import math
            st.write(random.random())
            st.write(random.randint(1,10))
            st.write(math.sqrt(9))

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()


if sidebar_selectbox_mode == "PY_Files_Import_py":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Import .py file")
    with st.echo():
        try:
            f_py_01 = open('C:\Dropbox\_\work\_io_streamlit_file_py_01.py', 'w')  # r = read, w = write
            f_py_01.write("import streamlit as st"+"\n")
            f_py_01.write("a = 11"+"\n")
            f_py_01.write("b = 22"+"\n")
            f_py_01.write("c = {'a':1,'b':2}"+"\n")
            f_py_01.write("\n")
            f_py_01.write("def f():"+"\n")
            f_py_01.write("\t"+"st.write('f: a='+str(a))"+"\n")
            f_py_01.write("\t"+"st.write('f: b='+str(b))"+"\n")
            f_py_01.write("\t"+"st.write('f: c='+str(c))"+"\n")
            f_py_01.write("\t"+"return 0"+"\n")
            f_py_01.write("\n")
            f_py_01.write("if __name__=='__main__':"+"\n")              # inside main = if file is called, this won't run
            f_py_01.write("\t"+"print('f: main: inside')"+"\n")
            f_py_01.write("\t"+"st.write('f: main: inside')"+"\n")
            f_py_01.write("print('f: main: outside')"+"\n")
            f_py_01.write("st.write('f: main: outside')"+"\n")
            f_py_01.close()

            import _io_streamlit_file_py_01 as f1       # all the code from file will be here, and will be executed
            st.write(f1.a)
            st.write(f1.b)
            st.write(f1.c)
            st.write(f1.f())

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)
            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()
        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Exceptions_Handling_SaveToFile":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Exceptions: Handling: Save to Log File")
    with st.echo():
        try:
            fe1 = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_01.txt', 'w')  # r = read, w = write
            fe1.write('11')
            fe1.close()

            fe2 = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_02.txt', 'w')  # r = read, w = write
            fe2.write('22')
            fe2.close()

            fe1 = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_01.txt', 'r')  # r = read, w = write
            fe2 = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_02.txt', 'r')  # r = read, w = write
            i1 = fe1.read()
            i2 = fe2.read()

            i1 = int(i1)
            i2 = int(i2)

            st.write(i1)
            st.write(i2)
            st.write(i1 * i2)

        except Exception as e:
            st.write(dt.datetime.now())
            st.write(e)

            fel = open('C:\Dropbox\_\work\_io_streamlit_file_text_exception_log.txt', 'a')  # r = read, w = write, a = append
            fel.write(str(dt.datetime.now()))
            fel.write(' : ')
            fel.write(str(e))
            fel.write('  \n')
            fel.close()


        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Exceptions_Handling":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Exceptions: Handling")
    with st.echo():
        try:
            f1 = float(st.number_input('enter a number 1: '))
            f2 = float(st.number_input('enter a number 2: '))
            st.write(f1)
            st.write(f2)
            st.write(f1/f2)

        except Exception as e:
            st.write(e)
            if type(e) == ZeroDivisionError:
                st.write('Error Resolution: Don\'t divide by zero')

        st.write('OK')

    toc.generate()

if sidebar_selectbox_mode == "PY_Exceptions":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Exceptions: IndentationError: unindent does not match any outer indentation level")
    with st.echo():
        try:
         a=100
        except Exception as e:
            st.write(e)

    toc.h1("Exceptions: ZeroDivisionError: division by zero")
    with st.echo():
        try:
            a=100
            st.write(a/0)
        except Exception as e:
            st.write(e)

    toc.h1("Exceptions: ValueError: invalid literal for int() with base 10: 'b'")
    with st.echo():
        try:
            a='b'
            st.write(int(a))
        except Exception as e:
            st.write(e)

    toc.h1("Exceptions: IndexError: string index out of range")
    with st.echo():
        try:
            a='b'
            st.write(a[0])
            st.write(a[1])
        except Exception as e:
            st.write(e)

    toc.h1("Exceptions: KeyError: 'b'")
    with st.echo():
        try:
            d = {'a':100}
            st.write(d['a'])
            st.write(d['b'])
        except Exception as e:
            st.write(e)

if sidebar_selectbox_mode == "PY_Files":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    # C:\Dropbox\_\work\_io_streamlit.py
    toc.h1("File: Create or Overwrite")
    with st.echo():
        f = open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','w') # w = write
        f.write('hello')
        f.close()
        st.write(f)

    toc.h1("File: Read")
    with st.echo():
        f = open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','r') # r = read
        s = f.read()    # read into a string
        st.write(s)
        f.close()

    toc.h1("File: Write Append")
    with st.echo():
        f = open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','a') # a = append
        f.write('world')
        f.write('  \n') # streamlit needs 2 white space in front of '\n'
        f.write('!!!')
        f.close()
        st.write(f)

    toc.h1("File: Read")
    with st.echo():
        f = open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','r') # r = read
        s = f.read()    # read into a string
        st.write(s)
        f.close()

    toc.h1("File: Write Append: Without need to close")
    with st.echo():
        with open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','a') as f: # a = append
            f.write('  \nappend without close')
        st.write(f)

    toc.h1("File: Read")
    with st.echo():
        f = open('C:\Dropbox\_\work\_io_streamlit_file_text.txt','r') # r = read
        s = f.read()    # read into a string
        st.write(s)
        f.close()

    toc.generate()

if sidebar_selectbox_mode == "PY_Functions_Dict_IO_Streamlit":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    dict_ref = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    dict_usr = {}
    st.session_state.dict_usr_st = {}

    toc.h1("Functions: Dict: Input: Streamlit")
    with st.echo():

        def dict_take_input_streamlit():

            # create text input box
            ticker = st.text_input('Enter Ticker:')

            # Check if ticker is found in the global dictionary
            found = dict_ref.get(ticker)

            if found: # if found
                # enter in local dictionary
                dict_usr[ticker] = found
                st.session_state.dict_usr_st[ticker] = found

            # return st.session_state.dict_usr

        st.write('dict_ref')
        st.write(dict_ref)
        # dict_temp = dict_take_input_streamlit()
        # st.write(dict_temp)
        dict_take_input_streamlit()
        st.write('dict_usr')
        st.write(dict_usr)
        st.write('st.session_state.dict_usr_st')
        st.write(st.session_state.dict_usr_st)

    toc.generate()

if sidebar_selectbox_mode == "PY_Functions_Dict_IO_Console":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Functions: Dict: Input: Console")
    print("Functions: Dict: Input: Console")

    with st.echo():

        def dict_take_input_console(dict_in):
            dict_local = {}
            while True:
                ticker = input('Enter Ticker:')
                if ticker == 'q':
                    break
                found = dict_in.get(ticker)
                if found:
                    dict_local[ticker] = found
                    print(dict_local)
                else:
                    print('not found')
            return dict_local

        print('\n')
        dict_ref = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        print('dict_ref')
        print(dict_ref)
        dict_temp = dict_take_input_console(dict_ref)
        print(dict_temp)

    toc.generate()

if sidebar_selectbox_mode == "PY_Functions_Dict_Print":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    dict_global = {'aapl': 100, 'amzn': 200, 'meta': 300, 'nvda': 400}

    toc.h1("Functions: Dict: Print: key-value pairs: of global dict")
    with st.echo():

        # method 1: bad
        def dict_print_global():
            for i,j in dict_global.items():
                st.write(i,j)

        st.write('dict_print_global()')
        dict_print_global()


    toc.h1("Functions: Dict: Print: key-value pairs: of local dict passed in as argument")
    with st.echo():

        # method 2: good
        def dict_print(d):
            for i,j in d.items():
                st.write(i,j)

        st.write('dict_print')
        dict_print(dict_global)

    toc.generate()

if sidebar_selectbox_mode == "PY_Functions_Parameters_Arguments":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Functions: Parameter(=Variable), Argument(=Value)")
    with st.echo():
        def f_param(param):
            return ("parameter = "+ param)

        st.write(f_param("argument"))

    toc.h1("Functions: Parameter(=Variable), Argument(=Value) Types")
    with st.echo():
        my_int = (5)
        my_tup = (5,)
        # st.write(type(my_int)) # int
        st.write(my_int)
        # st.write(type(my_tup)) # tuple
        st.write(my_tup)

    toc.h1("Functions: Parameter(=Variable), Argument(=Value): *args = Arbitrary Arguments")
    with st.echo():
        def f_arg_arbitrary(*tuple_args):
            return tuple_args

        # st.write(type(f_arg_arbitrary('arg0'))) # tuple
        st.write(f_arg_arbitrary('arg0'))

        # st.write(type(f_arg_arbitrary('arg0', 'arg1', 'arg2'))) # tuple
        st.write(f_arg_arbitrary('arg0', 'arg1', 'arg2'))

    toc.h1("Functions: Parameter(=Variable), Argument(=Value): kwargs = Keyword Arguments")
    with st.echo():
        def f_arg_keyword(arg0, arg1, arg2):
            st.write('arg0 = ' + str(arg0))
            st.write('arg1 = ' + str(arg1))
            st.write('arg2 = ' + str(arg2))

        st.write(f_arg_keyword(arg1=1, arg2=2, arg0=0)) # args can be in any order

    toc.h1("Functions: Parameter(=Variable), Argument(=Value): **kwargs = Arbitrary Keyword Arguments")
    with st.echo():
        def f_arg_keyword_arbitrary(**kwargs):
            st.write(kwargs)                # writes a dict
            st.write(kwargs.get('arg0'))    # writes a value
            st.write(kwargs.get('arg1'))    # writes a value
            st.write(kwargs.get('arg2'))    # writes a value

        f_arg_keyword_arbitrary(arg0=0, arg1=1, arg2=2)



    toc.h1("Functions: Nested: Guess next number")
    with st.echo():
        def f1(a):
            st.write('f1')
            return a+1
        def f2(*args):
            st.write('f2')
            last = args[-1]
            return f1(last)
        def f3(*args):
            st.write('f3')
            st.write(*args)
            st.write(type(args))
            return f2(*args)

        st.write(f3(1,2,3,4))



    toc.generate()

if sidebar_selectbox_mode == "PY_Functions":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Functions: List Sum")
    with st.echo():

        def lst_sum(lst):
            result = 0
            for i in lst:
                result += i
            return result

        st.write(lst_sum([1,2,3,4]))

        # get the nth fibonacci number
        def fib_nth(n, a=1, b=1):
            result = []
            # fib_nth(n=1) = 1 | i = 1 | _
            # fib_nth(n=2) = 1 | i = 2 | a=1, b=1
            # fib_nth(n=3) = 2 | i = 3 | a=1, b=2
            for i in range(3, n + 1):  # for i = 3 to n
                a, b = b, a + b
            return b

    toc.h1("Functions: List Average")
    with st.echo():

        def lst_avg(lst):
            result = 0
            for i in lst:
                result += i
            return result/len(lst)

        st.write(lst_avg([1,2,3,4]))

        # get the nth fibonacci number
        def fib_nth(n, a=1, b=1):
            result = []
            # fib_nth(n=1) = 1 | i = 1 | _
            # fib_nth(n=2) = 1 | i = 2 | a=1, b=1
            # fib_nth(n=3) = 2 | i = 3 | a=1, b=2
            for i in range(3, n + 1):  # for i = 3 to n
                a, b = b, a + b
            return b


    toc.h1("Functions: Fibonacci")
    with st.echo():

        def fib(n, a=0, b=1):
            result = []
            while a < n:
                result.append(a)
                a, b = b, a + b
            return result

        st.write(fib(5))

        # get the nth fibonacci number
        def fib_nth(n, a=1, b=1):
            result = []
            # fib_nth(n=1) = 1 | i = 1 | _
            # fib_nth(n=2) = 1 | i = 2 | a=1, b=1
            # fib_nth(n=3) = 2 | i = 3 | a=1, b=2
            for i in range(3, n + 1):  # for i = 3 to n
                a, b = b, a + b
            return b

    toc.h1("Functions: Reverse")
    with st.echo():
        def reverse(lst):
            return lst[::-1]

        st.write(reverse(fib(5)))

    toc.h1("Functions: Prime")
    with st.echo():
        def is_prime(n):
            mods = []
            if n <= 1:
                return False
            for i in range(2,n):
                r = n % i
                if r == 0:
                    return False
            return True

        def is_prime_fast(n):
            if n <= 1:
                return False
            elif n == 2:
                return True
            elif n % 2 == 0:
                return False
            else:
                # Check for factors up to the square root of the number
                for i in range(3, int(n ** 0.5) + 1, 2):
                    if n % i == 0:
                        return False
                return True

        st.write(is_prime(0))
        st.write(is_prime(1))
        st.write(is_prime(2))

    toc.h1("Functions: Prime: List")
    with st.echo():
        def list_primes(n):
            primes = []
            i = 2
            while len(primes) < n:
                if is_prime(i):
                    primes.append(i)
                i += 1
            return primes

        st.write(list_primes(10))

    toc.h1("Functions: Palindrome: String")
    with st.echo():
        def is_palindrome(s):
            return list(s) == list(s)[::-1]

        st.write(is_palindrome('abc'))
        st.write(is_palindrome('abcba'))

    toc.h1("Functions: Find Closest Strike")
    with st.echo():
        def get_ATM(price,strike_gap):
            atm_using_remainder = price - price % strike_gap
            atm_using_quotient = (price // strike_gap) * strike_gap
            return atm_using_quotient

        st.write(get_ATM(19799,50))

    toc.h1("Functions: Default Params")
    with st.echo():
        def power_of_default(base=2, exponent=2):
            return base ** exponent

        st.write(power_of_default())
        st.write(power_of_default(3))
        st.write(power_of_default(2,3))

    toc.h1("Functions: Params Specified, Sequence Irrelevant")
    with st.echo():
        def power_of(base, exponent):
            return base ** exponent

        st.write(power_of_default(2,3))
        st.write(power_of_default(base=2,exponent=3))
        st.write(power_of_default(exponent=3, base=2))


    toc.generate()

if sidebar_selectbox_mode == "PY_Loops_Nested_Console":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    print('-' * 10)
    print("Console: Print: Default: line-end")
    toc.h1("Console: Print: Default: line-end")
    with st.echo():
        print(1)
        print(2)

    print('-' * 10)
    print("Console: Print: Default: line-end: Equivalent")
    toc.h1("Console: Print: Default: line-end: Equivalent")
    with st.echo():
        print(1,end='\n')
        print(2)

    print('-' * 10)
    print("Console: Print: Default: line-end: Equivalent: Override")
    toc.h1("Console: Print: Default: line-end: Equivalent")
    with st.echo():
        print(1,end='')
        print(2)

    print('-' * 10)
    print("Loops: Nested")
    toc.h1("Loops: Nested")
    with st.echo():

        print('-' * 10)
        def pattern(n=5):
            for i in range(1,n+1):
                # print('*')
                for j in range(1,i+1):
                    print('*',end='')
                print('\n',end='')
        pattern()

        print('-' * 10)
        def pattern1(n=5):
            for i in range(1,n+1):
                print('*' * i)
        pattern1()


    toc.generate()

if sidebar_selectbox_mode == "PY_Loops":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Loops: Q1: print number 1 to 10")
    with st.echo():
        for i in range(0,10):
            st.write(i+1)

    toc.h1("Loops: Q2: print 100 even numbers")
    with st.echo():
        list_even = []
        for i in range(0,100):
            list_even.append((i+1)*2)
            # st.write((i+1)*2)
        st.write(list_even)

    toc.h1("Loops: Q3: print 100 odd numbers")
    with st.echo():
        list_odd = []
        for i in range(0,100):
            list_odd.append(i*2+1)
        st.write(list_odd)

    toc.h1("Loops: Q4: total, average")
    with st.echo():
        list_n = [2,3,4,5]
        n = len(list_n)
        total = 0
        for i in list_n:
            total = total + i
        avg = total / n
        st.write(total)
        st.write(avg)

    toc.h1("Loops: Q5: While: Needs to loop, but not sure for how long")
    with st.echo():
        x=0
        while True:
            x=x+1
            st.write(x)
            if x==10:
                break

    toc.generate()

if sidebar_selectbox_mode == "PY_Loops_Type_04":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Loops: Type 04: for i in range(0,len(list))")
    with st.echo():
        dict1 = {'a': 1, 'b': 2, 'c': 3}
        st.write(list(dict1.keys()))
        st.write(list(dict1.values()))
        st.write(list(dict1.items()))

    with st.echo():
        for i in dict1:
            st.write(i)

    with st.echo():
        for i in dict1.keys():
            st.write(i)

    with st.echo():
        for i in dict1.values():
            st.write(i)

    with st.echo():
        for key,val in dict1.items():
            st.write(key,val)

    with st.echo():
        for tuple_key_val in dict1.items():
            st.write(tuple_key_val)

if sidebar_selectbox_mode == "PY_Loops_Type_03":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Loops: Type 03: for i in range(0,len(list))")
    with st.echo():
        l1 = ['a', 'b', 'c', 'd', 'e']
        for i in range(0,len(l1)):
            st.write('list item ', i, '=', l1[i])

if sidebar_selectbox_mode == "PY_Loops_Type_02":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Loops: Type 02: for i in range(n)")

    with st.echo():
        for i in range(5):
            st.write('loop', i)

if sidebar_selectbox_mode == "PY_Loops_Type_01":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Loops: Type 01: for element in (List, String)")

    with st.echo():
        list1=[1,2,'amzn','sunil']
        st.write(list1[0],list1[1],list1[2],list1[3])

    with st.echo():
        for element in list1:
            st.write(element)

    with st.echo():
        l1 = ['a', 'b', 'c', 'd', 'e']
        for element in l1:
            st.write(element)

    with st.echo():
        s1 = 'abcde'
        for element in s1:
            st.write(element)

if sidebar_selectbox_mode == "PY_DataStructures_Blog_InvestmentPortfolio":

    # https://fessorpro.com/blogs/b/python-problems-on-datastructure-3605
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Data")
    # https://jsoncrack.com/editor

    with st.echo():
        investment_portfolio = {
            "investor_name": "Jane Doe",
            "portfolio_id": "JD1234",
            "assets": {
                "stocks": [
                    {
                        "ticker": "AAPL",
                        "quantity": 50,
                        "purchase_price": 120.00,
                        "current_price": 130.00
                    },
                    {
                        "ticker": "MSFT",
                        "quantity": 30,
                        "purchase_price": 200.00,
                        "current_price": 210.00
                    }
                ],
                "bonds": [
                    {
                        "identifier": "US123456",
                        "quantity": 100,
                        "purchase_price": 1000.00,
                        "current_price": 1020.00,
                        "maturity_date": "2030-01-01"
                    }
                ],
                "mutual_funds": [
                    {
                        "name": "XYZ Growth Fund",
                        "quantity": 200,
                        "purchase_price": 15.00,
                        "current_price": 15.50
                    }
                ]
            },
            "cash_holdings": 10000.00,
            "investment_goals": {"retirement": 2035, "education": 2025}
        }

    toc.h1("Q 1: Read Data")
    st.write(f"Task: Display the current price of the mutual fund \"XYZ Growth Fund\"")

    with st.echo():
        st.write(investment_portfolio.get('assets').get('mutual_funds')[0].get('current_price'))
    with st.echo():
        st.write(investment_portfolio['assets']['mutual_funds'][0]['current_price'])
    with st.echo():
        st.write(investment_portfolio['assets']['mutual_funds'][0])
    with st.echo():
        st.write(investment_portfolio['assets']['mutual_funds'])
    with st.echo():
        st.write(investment_portfolio['assets'])
    with st.echo():
        st.write(investment_portfolio)

    toc.h1(f"Q 2: Write Data: List of Dicts: Append")
    st.write(f'Task: Add a new stock with ticker "GOOG", 40 shares, a purchase price of 1500.00, and a current price of 1520.00 to the stocks in the assets.')
    with st.echo():
        st.write(investment_portfolio['assets']['stocks'])
    with st.echo():
        investment_portfolio['assets']['stocks'].append(
            {
                "ticker": "GOOG",
                "quantity": 40,
                "purchase_price": 1500.00,
                "current_price": 1520.00
            }
        )
    with st.echo():
        st.write(investment_portfolio['assets']['stocks'])

    toc.h1(f"Q3: Update Data")
    st.write(f'Task: Update the quantity of "AAPL" stock to 60.')
    with st.echo():
        st.write(investment_portfolio['assets']['stocks'][0])
    with st.echo():
        # get the index of the dictionary which contains the key-value pair
        list_of_dicts = investment_portfolio['assets']['stocks']
        search_key = 'ticker'
        search_value = 'AAPL'
        index_of_dict = next((index for index, item in enumerate(list_of_dicts) if item[search_key] == search_value), None)
    with st.echo():
        # check before
        st.write(investment_portfolio['assets']['stocks'][index_of_dict])
    with st.echo():
        # update datea
        investment_portfolio['assets']['stocks'][index_of_dict]['quantity'] = 60
    with st.echo():
        # check after
        st.write(investment_portfolio['assets']['stocks'][index_of_dict])

    toc.h1(f"Q4: Delete Data")
    st.write(f'Task: Remove the stock with identifier "GOOG" from the stocks in assets.')
    with st.echo():
        st.write(investment_portfolio['assets']['stocks'][2])
    with st.echo():
        # get the index of the dictionary which contains the key-value pair
        list_of_dicts = investment_portfolio['assets']['stocks']
        search_key = 'ticker'
        search_value = 'GOOG'
        index_of_dict = next((index for index, item in enumerate(list_of_dicts) if item[search_key] == search_value), None)
        st.write(index_of_dict)
    with st.echo():
        # check before
        st.write(investment_portfolio['assets']['stocks'])
    with st.echo():
        # delete data
        del investment_portfolio['assets']['stocks'][index_of_dict]
    with st.echo():
        # check after
        st.write(investment_portfolio['assets']['stocks'])

    toc.h1(f"Q5: Read Nested Data")
    st.write(f'Task: Read and display the maturity date of the bond bond identifier "US123456" from the bonds in assets.')
    with st.echo():
        st.write(investment_portfolio['assets']['bonds'][0])
    with st.echo():
        # get the index of the dictionary which contains the key-value pair
        list_of_dicts = investment_portfolio['assets']['bonds']
        search_key = 'identifier'
        search_value = 'US123456'
        index_of_dict = next((index for index, item in enumerate(list_of_dicts) if item[search_key] == search_value), None)
        st.write(index_of_dict)
    with st.echo():
        # read
        st.write(investment_portfolio['assets']['bonds'][index_of_dict]['maturity_date'])

    toc.h1(f"Q6: Update Nested Data")
    st.write(f'Task: Change the current price of "MSFT" stock to 215.00.')
    with st.echo():
        st.write(investment_portfolio['assets']['stocks'][1])
    with st.echo():
        # get the index of the dictionary which contains the key-value pair
        list_of_dicts = investment_portfolio['assets']['stocks']
        search_key = 'ticker'
        search_value = 'MSFT'
        index_of_dict = next((index for index, item in enumerate(list_of_dicts) if item[search_key] == search_value), None)
        st.write(index_of_dict)
    with st.echo():
        # check before
        st.write(investment_portfolio['assets']['stocks'][index_of_dict])
    with st.echo():
        # update data
        investment_portfolio['assets']['stocks'][index_of_dict]['current_price'] = 215.00
    with st.echo():
        # check after
        st.write(investment_portfolio['assets']['stocks'][index_of_dict])

    toc.h1(f"Q7: Add Nested Data")
    st.write(f'Task: Add a new goal "vacation" set for the year 2028 in the investment goals.')
    with st.echo():
        st.write(investment_portfolio['investment_goals'])
    with st.echo():
        investment_portfolio['investment_goals']['vacation'] = 2028
    with st.echo():
        st.write(investment_portfolio['investment_goals'])

    toc.h1(f"Q8: Delete an Item from a List Inside a Dictionary")
    st.write(f'Task: Remove the mutual fund "XYZ Growth Fund" from the portfolio.')
    with st.echo():
        st.write(investment_portfolio['assets']['mutual_funds'])
    with st.echo():
        # get the index of the dictionary which contains the key-value pair
        list_of_dicts = investment_portfolio['assets']['mutual_funds']
        search_key = 'name'
        search_value = 'XYZ Growth Fund'
        index_of_dict = next((index for index, item in enumerate(list_of_dicts) if item[search_key] == search_value), None)
        st.write(index_of_dict)
    with st.echo():
        del investment_portfolio['assets']['mutual_funds'][index_of_dict]
    with st.echo():
        st.write(investment_portfolio['assets']['mutual_funds'])

    toc.generate()

if sidebar_selectbox_mode == "PY_DataStructures":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    # python collaboration
    # https://replit.com/


    toc.h1("hello")
    with st.echo():
        s = "hello"

    toc.h1("Data Structures: Dict of Lists")
    with st.echo():
        s = "asdf"

    toc.h1("Data Structures: List of Lists")
    with st.echo():
        s = "asdf"

    toc.h1("Data Structures: Problem 1")
    with st.echo():
        # https://fessorpro.com/blogs/b/python-problems-on-datastructure
        # https://jsoncrack.com/

        # stocks = list of 3 dicts
        stocks = [
            {
                "name": "Company A",
                "symbol": "CMPA",
                "sector": "Technology",
                "current_price": 100.0,
                "historical_data": [
                    {
                        "date": "2024-01-10",
                        "prices": {
                            "open": 98.0,
                            "close": 100.0,
                            "high": 101.0,
                            "low": 97.0
                        },
                        "volume": 12000
                    },
                    {
                        "date": "2024-01-09",
                        "prices": {
                            "open": 97.0,
                            "close": 98.0,
                            "high": 99.0,
                            "low": 96.0
                        },
                        "volume": 15000
                    }
                ],
                "locations": ["New York", "London"]
            },
            {
                "name": "Company B",
                "symbol": "CMPB",
                "sector": "Finance",
                "current_price": 200.0,
                "historical_data": [
                    {
                        "date": "2024-01-10",
                        "prices": {
                            "open": 198.0,
                            "close": 200.0,
                            "high": 202.0,
                            "low": 196.0
                        },
                        "volume": 18000
                    },
                    {
                        "date": "2024-01-09",
                        "prices": {
                            "open": 196.0,
                            "close": 198.0,
                            "high": 199.0,
                            "low": 195.0
                        },
                        "volume": 17000
                    }
                ],
                "locations": ["Tokyo", "Singapore"]
            },
            {
                "name": "Company C",
                "symbol": "CMPC",
                "sector": "Healthcare",
                "current_price": 300.0,
                "historical_data": [
                    {
                        "date": "2024-01-10",
                        "prices": {
                            "open": 295.0,
                            "close": 300.0,
                            "high": 302.0,
                            "low": 294.0
                        },
                        "volume": 22000
                    },
                    {
                        "date": "2024-01-09",
                        "prices": {
                            "open": 294.0,
                            "close": 295.0,
                            "high": 296.0,
                            "low": 293.0
                        },
                        "volume": 21000
                    }
                ],
                "locations": ["Berlin", "Paris"]
            }
        ]

        st.write(type(stocks))

        toc.h2("Data Structures: Q1. Read Data: from list of dicts: Company B, Current Price")
        # st.write(stocks)
        # st.write(stocks[1])
        st.write(stocks[1].get('current_price'))
        st.write(stocks[1]['current_price'])

        toc.h2("Data Structures: Q2. Write Data: add new dict to the list of dicts")
        st.write(len(stocks))
        # data = new dict
        data = { "name": "Company D",
                 "symbol": "DDD",
                 "sector": "D_Sector",
                 "current_price": 400.0,
                 "locations": ["Mumbai", "Bangalore"]
                }
        stocks.append(data)
        st.write(len(stocks))
        st.write(stocks)

        toc.h2("Data Structures: Q3. Update Data: Company C, Current Price = 310")
        st.write(stocks[2].get('current_price'))
        stocks[2]['current_price'] = 310.0
        st.write(stocks[2].get('current_price'))

        toc.h2("Data Structures: Q4. Delete Data: Company A, Historical Data, Date=2024-01-09")
        st.write(stocks[0].get('historical_data'))
        # del stocks[0].get('historical_data')[1]
        # del stocks[0].get('historical_data')[  stocks[0]['historical_data'][0]['date'] == '2024-01-09'  ]
        st.write(stocks[0].get('historical_data')[1])

        toc.h2("Data Structures: Q5. Read Nested Data: Company B, Historical Data, Date=2024-01-10")
        st.write(stocks[0]['historical_data'][0]['prices']['close'])                # method 1:
        st.write(stocks[0].get('historical_data')[0].get('prices').get('close'))    # method 2: if dict, can use get

        toc.h2("Data Structures: Q6. Read Nested Data: Company A, Historical Data, Date=2024-01-10, opening price 98 to 99")
        st.write(stocks[0]['historical_data'][0]['prices']['open'])
        stocks[0]['historical_data'][0]['prices']['open'] = 99.0
        st.write(stocks[0]['historical_data'][0]['prices']['open'])

        toc.h2("Data Structures: Q7. Add Nested Data: Company C, Historical Data, Date=2024-01-11, opening price 98 to 99")
        # st.write(stocks)
        # st.write(stocks[2])
        st.write(stocks[2]['historical_data'])
        # st.write(stocks[2]['historical_data'][0])

        data = {    "date": "2024-01-11",
                    "prices":{
                                "open": 295,
                                "close": 300,
                                "high": 302,
                                "low": 294
                                },
                    "volume": 22000
        }
        stocks[2]['historical_data'].append(data)

        st.write(stocks[2]['historical_data'])

        toc.h2("Data Structures: Q8. Delete item from List in Dict: Company B, Locations, Singapore")
        st.write(stocks[1]['locations'])
        stocks[1]['locations'].remove('Singapore')
        st.write(stocks[1]['locations'])

    # Data Structures
    # https: // fessorpro.com / blogs / b / problems - on - python - datastructure - 5897

    # Conditionals
    # https: // fessorpro.com / blogs / b / problems - on - conditionals - -4075

    # Variables and Strings
    # https: // fessorpro.com / blogs / b / problems - on - variable - and -strings

    # https://dashboard.fessorpro.com/s/courses/653364f9e4b0987722577e95/take
    # 00:50

if sidebar_selectbox_mode == "PY_DataStructures_Sets":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Data Structures: Sets") # All elements are unique
    with st.echo():
        s = "hello"

if sidebar_selectbox_mode == "PY_DataStructures_Dictionaries":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Data Structures: Dictionaries")
    with st.echo():
        dict1={'a':1, 'b':2, 'c':3}
        st.write(list(dict1.keys()))

    with st.echo():
        st.write(list(dict1.values()))

    with st.echo():
        st.write(list(dict1.items()))

    toc.h1("Data Structures: Dictionaries: dict.get(value) = None")
    with st.echo():
        dict_d = {}
        dict_d['k1'] = 'v1'
        dict_d['k2'] = 'v2'
        dict_d['k3'] = 'v3'
        st.write(dict_d)
        st.write(dict_d.get('k4'))  # No Error
        # st.write(dict_d['k4'])  # KeyError: 'k4'


    toc.h1("Data Structures: Dictionaries: add, del, pop")
    with st.echo():
        dict_d = {'k1':'v1', 'k2':'v2'}
        st.write(dict_d)

        dict_d['k3'] = 'v3'
        st.write(dict_d)

        del dict_d['k2']
        st.write(dict_d)

        dict_d.pop('k1')
        st.write(dict_d)


    toc.h1("Data Structures: Dictionaries: Lookup value using key, not index")
    with st.echo():
        dict_d = {'k1':'v1', 'k2':'v2'}
        st.write(dict_d)

if sidebar_selectbox_mode == "PY_DataStructures_Tuples":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Data Structures: Tuples = immutable, cannot modify")
    with st.echo():
        tuple_empty = ()
        tuple_1_element = (10,)
        tuple_strings = ('a', 'bc', 'def')
        st.write(type(tuple_strings))
        st.write(tuple_strings)
        st.write(tuple_strings[1])

if sidebar_selectbox_mode == "PY_DataStructures_Lists":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Data Structures: Lists: del list[index]")
    with st.echo():
        list_a = [0, 1, 2, 3, 4, 5]
        st.write(list_a)
        del list_a[:3]   # start to 2
        st.write(list_a)

    toc.h1("Data Structures: Lists: list.pop(index)")
    with st.echo():
        list_a = [0, 1, 2, 3, 4, 5]
        st.write(list_a)
        list_a.pop(4)   # remove the element at index '4'
        st.write(list_a)

    toc.h1("Data Structures: Lists: list.remove(value)")
    with st.echo():
        list_a = [0, 1, 2, 3, 4, 5]
        st.write(list_a)
        list_a.remove(3)   # remove the value '3'
        st.write(list_a)

    toc.h1("Data Structures: Lists: list.insert(index,value)")
    with st.echo():
        list_a = [0, 1, 2, 3, 4, 5]
        st.write(list_a)
        list_a.insert(3,'apple')   # insert at index 3
        st.write(list_a)

    toc.h1("Data Structures: Lists: list.append(value)")
    with st.echo():
        list_a = [0, 1, 2, 3, 4, 5]
        st.write(list_a)
        list_a.append('apple')   # append at the end
        st.write(list_a)

    toc.h1("Data Structures: Lists")
    with st.echo():
        list_empty = []
        list_numbers = [10, 20, 30.5]
        list_mixed_types = ['hello', 20, 30.5]
        st.write(list_mixed_types[1])
        st.write(list_mixed_types[1:3])

if sidebar_selectbox_mode == "PY_UserInput":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("User Input: Float, Approx to 2 d.p., use string slicing")
    with st.echo():
        f_input = "12345.6789"
        f_tgt   = "12345.67"
        # f = st.text_area("Enter Float: ")
        f = "12345.6789"
        st.write("f = " + f)
        i_point = f.find(".")
        st.write(i_point)
        st.write(f[:i_point+3])

    toc.h1("User Input: If odd num char, print mid, cap. If even num char, print mid 2 char, cap.")
    with st.echo():
        # s = st.text_area("Enter String: ")
        s = "abdcde"
        st.write("s = " + s)
        i_mid = int(len(s)/2)
        st.write(i_mid)
        if len(s) % 2 == 0:
            st.write(s[i_mid-1:i_mid+1].upper())
        else:
            st.write(s[i_mid].upper())


    toc.h1("User Input: Even Odd")
    with st.echo():
        # n = int(st.text_area("Enter num: "))
        n = 8
        st.write("n = " + str(n))
        if n % 2 == 1:
            st.write("odd")
            st.write(n ** 3)
        if n % 2 == 0:
            st.write("even")
            st.write(n ** 2)


    toc.h1("User Input: Buy Sell")
    with st.echo():
        day = "Wednesday"
        # day = input("Enter day of the week: ")
        # day = st.text_area("Enter day of the week: ")
        num_stocks = 10
        # num_stocks = input("Enter num stocks: ")
        # num_stocks = int(st.text_area("Enter num stocks: "))
        st.write("day = " + day)
        st.write("num_stocks = " + str(num_stocks))
        if num_stocks < 3 or day == "Monday":
            st.write("buy")
        elif num_stocks >= 10 and day == "Wednesday":
            st.write("sell")
        else:
            st.write("hold")

if sidebar_selectbox_mode == "PY_Conditionals":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("Conditional: Problem: Hand Game")
    with st.echo():
        # hand1 = input("hand1: S, P, R")
        hand1 = "scissors"
        # hand1 = "paper"
        # hand1 = "rock"
        # hand2 = "scissors"
        hand2 = "Paper"
        # hand2 = "rock"
        st.write("hand1 = " + hand1)
        st.write("hand2 = " + hand2)
        # print which hand wins the game
        if hand1[0].lower() == hand2[0].lower():
            st.write("draw")
        else:
            if hand1[0].lower() == "s":
                if hand2[0].lower() == "p":
                    st.write("hand 1 wins")
                elif hand2[0].lower() == "r":
                    st.write("hand 2 wins")
            elif hand1[0].lower() == "p":
                if hand2[0].lower() == "s":
                    st.write("hand 2 wins")
                elif hand2[0].lower() == "r":
                    st.write("hand 1 wins")
            elif hand1[0].lower() == "r":
                if hand2[0].lower() == "s":
                    st.write("hand 1 wins")
                elif hand2[0].lower() == "p":
                    st.write("hand 2 wins")


    toc.h1("Conditional: Problem")
    with st.echo():
        # number = randint(1,100)
        number = 15
        st.write(number)
        # if number is divisible by 3 print fizz
        # if number is divisible by 5 print buzz
        # if number is divisible by 3 and 5, print fizzbuzz
        if number % 3 ==  0 and number % 5 == 0:
            print("fizzbuzz")
            st.write("fizzbuzz")
        elif number % 3 == 0:
            print("fizz")
            st.write("fizz")
        elif number % 5 == 0:
            print("buzz")
            st.write("buzz")
        else:
            pass

    toc.h1("Conditional: or, and")
    with st.echo():
        day = "Wednesday"
        num_stocks = 10
        if num_stocks < 3 or day == "Monday":
            st.write("buy")
        elif num_stocks >= 10 and day == "Wednesday":
            st.write("sell")
        else:
            st.write("hold")

    toc.h1("Conditional: if * n (check all)")
    with st.echo():
        num_stocks = 10
        if num_stocks < 3:
            st.write("buy")
        if num_stocks > 10:
            st.write("sell")
        if 3 <= num_stocks <= 10:
            st.write("hold")

    toc.h1("Conditional: if, elif *n (check only 1),  else")
    with st.echo():
        num_stocks = 10
        if num_stocks < 3:
            st.write("buy")
        elif num_stocks > 10:
            st.write("sell")
        else:
            st.write("hold")

    toc.h1("Conditional: if, else")
    with st.echo():
        num_stocks = 10
        if num_stocks < 3:
            st.write("buy")
        else:
            st.write("sell")

    toc.h1("Conditional: if")
    with st.echo():
        num_stocks = 2
        if num_stocks < 3:
            st.write("buy")
        st.write("always")

    toc.h1("Type: Boolean: True, False")
    with st.echo():
        b_mkt_open = True
        st.write(type(b_mkt_open))
        st.write(b_mkt_open)

    toc.h1("Comparison Operators: <, <=, ==, >=, >, !=")
    with st.echo():
        st.write(5 < 10)
        st.write(5 == 10)

    with st.echo():
        a = 6==6
        b = 5+6
        st.write(a)
        st.write(b)

    with st.echo():
        a = 'Hello'
        b = 'hello'
        check_eq  = a == b
        check_neq = a != b
        st.write(check_eq)
        st.write(check_neq)

if sidebar_selectbox_mode == "PY_Strings":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    toc.h1("String: Problem 3: Swap the first and last element of the given string")
    with st.echo():
        s = "hello how7 are yo8u"
        a = "uello how7 are yo8h"
        c1 = s[0]
        cn = s[-1]
        print(cn + s[1:-1] + c1)
        st.write(cn + s[1:-1] + c1)

    toc.h1("String: Problem 2: Get the content between two single-digit numbers in a new string")
    with st.echo():
        s = "hello how7 are yo8u"
        a = " are yo"
        list_of_indices = [index for index, char in enumerate(s) if char.isdigit()]
        ss = s[list_of_indices[0]+1:list_of_indices[1]]
        print("|" + ss + "|")
        st.text("|" + ss + "|")

    toc.h1("String: Problem 1: Find any single-digit number, replace it with the square")
    with st.echo():
        s = "hello how7 are yo8u"
        a = "hello how49 are yo64u"
        list_of_indices = [index for index, char in enumerate(s) if char.isdigit()]
        # st.write(list_of_indices)
        def square_the_int_string(s_num):
            return str(int(s_num) ** 2)
        ss = s
        for i in list_of_indices:
            ss = ss.replace(s[i],square_the_int_string(s[i]))
        print(ss)
        st.write(ss)

        # for i in range(0,10):
        #     ii = s.find(str(i))
        #     if ii > 0:
        #         number = s[ii:ii + len(str(i))]
        #         st.write(number)

    toc.h1("String: join*()")
    with st.echo():
        s = "Hello World From Wylie"
        list_of_words = s.split()
        sentence = " ".join(list_of_words)
        st.write(list_of_words)
        st.write(sentence)

    toc.h1("String: split(): Split sentence into list of words")
    with st.echo():
        s = "Hello World From Wylie"
        st.write(s.startswith("Hello"))
        st.write(s.startswith("hello"))
        st.write(s.endswith("Wylie"))
        st.write(s.endswith("wylie"))

    toc.h1("String: startswith(), endswith()")
    with st.echo():
        s = "   3 spaces on left, 5 spaces on right     "

    toc.h1("String: strip(), lstrip(), rstrip: Remove left right spaces")
    with st.echo():
        s = "   3 spaces on left, 5 spaces on right     "
        st.write("|" + s + "|")
        st.write(len(s))
        st.write("|" + s.strip() + "|")
        st.write(len(s.strip()))
        st.write("|" + s.lstrip() + "|")
        st.write(len(s.lstrip()))
        st.write("|" + s.rstrip() + "|")
        st.write(len(s.rstrip()))

    toc.h1("String: replace()")
    with st.echo():
        s = "Hello World From Wylie"
        st.write(s.replace("World","Earth"))

    toc.h1("String: index()")
    with st.echo():
        s = "Hello World From Wylie"
        i1 = s.index('o')
        i2 = s.index('o', i1+1)
        st.write(i1)
        st.write(i2)

    toc.h1("String: find(): Find substring within string")
    with st.echo():
        s = "Hello World From Wylie"
        i1 = s.find('o')
        i2 = s.find('o', i1+1)
        st.write(i1)
        st.write(i2)

    toc.h1("String: count(): Count frequency of substring")
    with st.echo():
        s = "Hello World From Wylie"
        st.write(s.count('e'))
        st.write(s.count('o'))

    toc.h1("String case functions")
    with st.echo():
        s = "Hello World From Wylie"
        st.write(s.upper())
        st.write(s.lower())
        st.write(s.capitalize())
        st.write(s.title())

    toc.h1("String: Functions on strings")
    with st.echo():
        f = 123.456
        st.write(str(f))
        st.write(len(str(f)))

    toc.h1("String fstring")
    with st.echo():
        f_name = "Wylie"
        l_name = "Chan"
        s1 = "My full name is " + f_name + " " + l_name + "."
        s2 = f"My full name is {f_name} {l_name}."
        st.write(s1)
        st.write(s2)

    toc.h1("String Split")
    with st.echo():
        number = 123.45
        answer_tgt = 45.123
        list_of_strings = str(number).split('.')
        answer = float(list_of_strings[1] + '.' + list_of_strings[0])
        st.write(answer)

    toc.h1("String Manipulation: len(), slicing")
    with st.echo():

        number = 56.98
        answer_tgt = 98.56
        i_half = len(str(number)) // 2
        answer_1 = float(str(number)[-i_half:] + '.' + str(number)[:i_half])
        answer_2 = float(str(number)[-2:] + '.' + str(number)[:2])
        answer_3 = float(str(number)[3:] + '.' + str(number)[:-3])

        st.write(str(answer_1))
        st.write(str(answer_2))
        st.write(str(answer_3))

    toc.h1("String Suffix")
    with st.echo():
        s="hello"
        s_tgt = "llo"
        st.write(s[2:5])
        st.write(s[2:])
        st.write(s[-3:])

    toc.h1("String Indexing Half")
    with st.echo():
        word1 = 'Good'
        half1 = len(word1) // 2
        word2 = 'Evening'
        half2 = len(word2) // 2
        st.write( word1[half1:])
        st.write( word2[half2:])

    toc.h1("String Reverse")
    with st.echo():
        n = 12.34
        s_rev = float(str(n)[::-1])
        st.write(s_rev)

    toc.h1("String Types")
    with st.echo():
        a = '100'
        b = float(a)
        st.write(a)
        st.write(b)
        c = type(a)
        d = type(b)
        print(c, d)
        print(10 / 2)

    toc.h1("Strings Indexing: Start, End+1, Step")
    with st.echo():
        s = "abcdefghij"
        s_tgt = "adg"
        st.write("skip index by 3 steps")
        st.write(s[0:9:3])
    with st.echo():
        s = "python program"
        s_tgt = "pto"
        st.write("skip index by 2 steps")
        st.write(s[0:5:2])

    toc.h1("Strings Indexing: (+), (-)")
    with st.echo():
        st.text("  P  y  t  h  o  n")
        st.text("  0  1  2  3  4  5")
        st.text(" -6 -5 -4 -3 -2 -1")
        s = "python"
    with st.echo():
        st.write("Find t")
        st.write(s[2])
        st.write(s[-4])
    with st.echo():
        st.write("Find n")
        st.write(s[5])
        st.write(s[-1])
    with st.echo():
        st.write("Find tho")
        st.write(s[2:5])
        st.write(s[-4:-1])
        st.write("Find tho using both (+) and (-) indices mixed")
        st.write(s[2:-1])
        st.write(s[-4:5])
    with st.echo():
        st.write("If pos(i_start) >= pso(i_end), we get null string")
        st.write(s[2:2])
        st.write(s[2:-4])
        st.write(s[-4:2])
        st.write("end")

    # Tab less memory than Space
    toc.h1("Strings Newline, Tab")
    with st.echo():
        st.write("Newline =|\n\n|")
        st.write("Tab     =|\t\t|")

    toc.h1("String * Integer")
    with st.echo():
        a = '10'
        b = 10
        st.write(a * b)

    toc.h1("Strings with Quotes")
    with st.echo():
        s1 = "Hells's Grannies"
        s2 = 'Hells"s Grannies'

    # 12
    toc.h1("Strings from Integer")
    with st.echo():
        i = 100
        s = '100'
        st.write(str(i)+s)

    # 11
    toc.h1("Strings")
    with st.echo():
        st.write('Single Quotes')
        st.write("Double Quotes")

    with st.echo():
        st.write("5" + '5')
        st.write(5 + 5)

    toc.h1("Strings Concatenated with/without Space")
    with st.echo():
        first_name = 'nifty'
        second_name = 'fifty'
        full_name = first_name + second_name
        st.write(full_name)
        st.write(first_name, second_name)

if sidebar_selectbox_mode == "PY_Math":
    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    # 10
    toc.h1("Find Closest Strike")
    price = 19799
    strike_gap = 50
    atm_tgt = 19750
    atm_using_remainder = price - price % strike_gap
    atm_using_quotient = (price // strike_gap) * strike_gap
    st.write("price = dividend = " + str(price))
    st.write("strike_gap = divisor = " + str(strike_gap))
    st.write("atm_using_remainder = price - price % strike_gap")
    st.write(atm_using_remainder)
    st.write("atm_using_quotient = (price // strike_gap) * strike_gap")
    st.write(atm_using_quotient)

    # 9
    toc.h1("Mathematical Way of Truncating to 2 d.p.")
    stock_price = 100
    stop_loss_pct = 1/3
    stop_loss_level = stock_price * stop_loss_pct
    st.write(stop_loss_level)
    st.write(int(stop_loss_level * 100)/100)

    # 8
    toc.h1("Buy a Stock")
    X = 1450
    capital = 900
    capital_allocation_X = 1/3
    market_val_X = capital * capital_allocation_X
    investors_needed = int(X / (capital * capital_allocation_X)) + 1
    st.write("Stock Price = " + str(X))
    st.write("Capital from 1 Investor = " + str(capital))
    st.write("Capital Allocation in the Stock = " + str(capital_allocation_X))
    st.write("Market Value in Stock = " + str(market_val_X))
    st.write("Since Market Value in Stock < Stock Price, we can't even buy 1 Stock.")
    st.write("We need more Investor Capital. How many Investors?")
    st.write("Investors Needed = " + str(investors_needed))

    # 7
    toc.h1("modulus operator to get remainder")
    st.write("int(10 % 3)")
    st.write(int(10 % 3))

    # 6
    toc.h1("int function to get quotient")
    st.write("int(10 / 3)")
    st.write(int(10 / 3))
    st.write("10//3")
    st.write(10//3)

    # 5
    toc.h1("int function is quotient after dividing by 1")
    st.write("int(4.8)+1")
    st.write(int(4.8)+1)

    # 4
    toc.h1("Order of Operation")
    st.write("(5 + 7) * 3 = " + str((5 + 7) * 3))
    st.write(" 5 + 7  * 3 = " + str( 5 + 7  * 3))
    st.write("PEMDAS = Parentheses Exponent Multiplication Division Addition Subtraction = ()^*/+-")

    # 3
    toc.h1("Integers and Floats")
    st.write("Integer = " + str(35))
    st.write("Float   = " + str(30.5))

    # 2
    toc.h1("Mathematical Operators")
    st.write(" 3  + 5 = " + str( 3 + 5))
    st.write("10  - 5 = " + str(10 - 5))
    st.write(" 3  * 5 = " + str( 3 * 5))
    st.write("30  / 6 = " + str(30 / 6))
    st.write(" 2 ** 3 = " + str( 2** 3))
    st.write("10  % 3 = " + str(10 % 3))

if sidebar_selectbox_mode == "PY":

    # create a TOC and put it in the sidebar
    toc = TOC()
    toc.placeholder(sidebar=True)
    toc.generate()

    # 1
    toc.h1("Python Interpreter")
    st.write("High Level Python --> Interpreter --> Low-Level Binary 01001")


    toc.generate()

    toc.generate()











