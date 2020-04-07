from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import cm
import numpy as np
import pandas as pd


df = pd.read_hdf("./data/covid19.h5", "covid19_data")

# Available data
ctry_arr = df.index.levels[0].values
dates_arr = pd.to_datetime(df.columns[3:-1].values)
sts_arr = np.unique(df.index.levels[1].values)


# Choropleth chart
n_color = 6
cmap = cm.get_cmap("Blues")

colorscale = []
for i in range(n_color):
    print(i, i / (n_color - 1))
    color_tp = tuple((np.array(cmap(i / (n_color - 1))[0:-1]) * 255).astype(int))
    color = "rgb" + str(color_tp)
    colorscale += [[10 ** (-(n_color - i - 1)), color]]
colorscale[0][0] = 0.0

ix = pd.IndexSlice
slc_st = ix[:, "deaths"]
col_date = dates_arr[-1].strftime("%Y-%m-%d")

iso = df.loc[slc_st, "ISO"].groupby('Country/Region').min()
cases = df.loc[slc_st, col_date].groupby('Country/Region').sum()

df.loc[slc_st, col_date]
ctry = df.index.unique(0)


fig = go.Figure(
    data=go.Choropleth(
        locations=iso,
        z=cases,
        text=ctry,
        colorscale=colorscale,
        marker_line_color="darkgray",
        marker_line_width=0.5,
        colorbar_title="Cases",
    )
)

fig.update_layout(
    title_text="Number of cases",
    margin=dict(l=0, r=0, b=0, t=0),
    geo=dict(showframe=False, showcoastlines=False, projection_type="equirectangular"),
    annotations=[dict(x=0.55, y=0.1, xref="paper", yref="paper",)],
)

fig.show()

dct = fig.to_dict()

# Pie Chart

slc_st = ix[:, "deaths"]
col_date = dates_arr[-1].strftime("%Y-%m-%d")
cases = df.loc[:, col_date].groupby("status").sum()
cases["active"] = cases["confirmed"] - cases["deaths"] - cases["recovered"]

del cases["confirmed"]

fig = go.Figure(
    data=go.Pie(
        labels=cases.index,
        values=cases.values,
        textinfo="percent+value+label",
        hole=0.3,
    )
)
fig.show()


import plotly.express as px
import plotly.io as pio

df3 = px.data.tips()
df3
df2 = df.reset_index()
df2.head()

ix = pd.IndexSlice
df4 = df.loc[ix[:, 'confirmed'], '2020-04-05'].reset_index()

fig = px.sunburst(
    df4,
    path=['status', 'Country/Region'],
    values='2020-04-05')

fig.update_traces(
    go.Sunburst(
        hovertemplate='<b>%{label} </b> <br> Cases: %{value:,.0f} ( %{percentEntry:.2%})</br>'),
    insidetextorientation='radial',
    textinfo='label+percent entry'
)

dct = fig.to_dict()



fig = make_subplots(
    rows=3, cols=3,
    specs=[[{"rowspan": 2, "colspan": 3}, None, None],
           [None, None, None],
           [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
    print_grid=True)

fig.add_trace(dct['data'][0], row=3, col=1)

dct['data'][0]
