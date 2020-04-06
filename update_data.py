import os
import numpy as np
import pandas as pd
import pycountry
import datetime

data_path = "./data/csse_covid_19_data/csse_covid_19_time_series/"
data_path_day = "./data/csse_covid_19_data/csse_covid_19_daily_reports/"

cv_cfm = "time_series_covid19_confirmed_global.csv"
cv_dth = "time_series_covid19_deaths_global.csv"
cv_rec = "time_series_covid19_recovered_global.csv"

# Read .csv files
df_cfm = pd.read_csv(data_path + cv_cfm)
df_dth = pd.read_csv(data_path + cv_dth)
df_rec = pd.read_csv(data_path + cv_rec)

# Set index as Country/Region
idxs = ["Country/Region", "Province/State"]
df_cfm = df_cfm.set_index(idxs)
df_dth = df_dth.set_index(idxs)
df_rec = df_rec.set_index(idxs)

# Join the datasets
df = pd.concat([df_cfm, df_dth, df_rec], keys=["confirmed", "deaths", "recovered"])
df = df.swaplevel(2, 0)
df = df.swaplevel(1, 0)
df = df.sort_index()
df.index.names = list(df.index.names)[0:-1] + ["status"]

date_col = zip(pd.to_datetime(df.columns[2:]), df.columns[2:])
df = df.rename(columns={ufmt: str(fmt.date()) for fmt, ufmt in date_col})

today = datetime.datetime.today()

tdy_str = today.strftime("%m-%d-%Y")
tdy_ymd = today.strftime("%Y-%m-%d")

col_dflt = ["Country_Region", "Province_State", "Lat", "Long_"]
col_trs = {
    "Country_Region": "Country/Region",
    "Province_State": "Province/State",
    "Long_": "Long",
    "Confirmed": tdy_ymd,
    "Recovered": tdy_ymd,
    "Deaths": tdy_ymd,
}

if today >= pd.to_datetime(df.columns[-1]):
    day_fl = data_path_day + tdy_str + ".csv"
    if os.path.exists(day_fl):
        # Get the daily data
        df_day = pd.read_csv(day_fl)

        # Formating the confirmed data
        df_day_cfm = df_day[col_dflt + ["Confirmed"]].T
        df_day_cfm = df_day_cfm.rename(col_trs).T

        # Formating the deaths data
        df_day_dth = df_day[col_dflt + ["Deaths"]].T
        df_day_dth = df_day_dth.rename(col_trs).T

        # Formating the recovered data
        df_day_rec = df_day[col_dflt + ["Recovered"]].T
        df_day_rec = df_day_rec.rename(col_trs).T

        # Set index as Country/Region
        df_day_cfm = df_day_cfm.set_index(idxs)
        df_day_dth = df_day_dth.set_index(idxs)
        df_day_rec = df_day_rec.set_index(idxs)

        # Join the datasets
        df_day = pd.concat(
            [df_day_cfm, df_day_dth, df_day_rec],
            keys=["confirmed", "deaths", "recovered"],
        )
        df_day = df_day.swaplevel(2, 0)
        df_day = df_day.swaplevel(1, 0)
        df_day = df_day.sort_index()

        df = pd.concat([df, df_day])


# Local translation
key2iso = {"US": "United States"}
df.rename(index=key2iso, level=0, inplace=True)

df["ISO"] = np.nan

ctry_dct = {}
for country in pycountry.countries:
    ctry_dct[country.name] = country.alpha_3

ctry_lst = np.unique(df.index.get_level_values(0))
for ctry in ctry_lst:
    if ctry in ctry_dct.keys():
        df["ISO"].loc[ctry] = ctry_dct[ctry]
    else:
        for ctry_ky in ctry_dct.keys():
            if ctry in ctry_ky:
                df["ISO"].loc[ctry] = ctry_dct[ctry_ky]
                break

df.reset_index(inplace=True)
df.set_index(["Country/Region", "status"], inplace=True)

df.to_hdf("./data/covid19.h5", "covid19_data")


# ddata_path = "./data/csse_covid_19_data/csse_covid_19_daily_reports/"
# cv_day = "04-01-2020.csv"

# # Read .csv files
# df_day = pd.read_csv(ddata_path + cv_day)
