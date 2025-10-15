# from __future__ import annotations

from textwrap import wrap
from typing import Any, Dict, Sequence
from pathlib import Path

import streamlit as st
import colorlover as cl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, weibull_min
from pelicun.base import convert_to_MultiIndex, pelicun_path
from st_search.fuzzy_visuals import render_fuzzy_search
from auth.simple_auth import ensure_login, current_user
from auth.login_ui import render_login_panel
from st_ui.auth_ui import render_login_panel
from st_ui.sidebar import render_sidebar
from st_ui.main_page import render_main_page


from visuals_core import build_fragility_figure

# st.write(st.session_state)

render_sidebar()
render_main_page()

# st.write(st.session_state)



# DATA_PATH = (
#     Path(__file__).parent / 'seismic' / 'building' / 'component' / 'FEMA P-58 2nd Edition' / 'fragility.csv'
# )
# df = pd.read_csv(DATA_PATH) #(r"seismic\building\component\FEMA P-58 2nd Edition\fragility.csv")

# search_ui = render_fuzzy_search()

# st.write(st.session_state)

# st.write(df)
# sf = df.iloc[0]

# st.title("SF")
# st.write(sf)

# mi_sf = convert_to_MultiIndex(sf, axis = 0)
# st.write(mi_sf)

# st.write(build_fragility_figure(mi_sf))
