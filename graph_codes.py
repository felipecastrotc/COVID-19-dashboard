import plotly.graph_objects as go
from matplotlib import cm
import numpy as np
import panads as pd


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

iso = df.loc[slc_st, "ISO"].groupby(CTRY_K).min()
cases = df.loc[slc_st, col_date].groupby(CTRY_K).sum()

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
