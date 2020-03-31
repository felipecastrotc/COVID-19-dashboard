# COVID-19 Dashboard

This is a dashboard developed in Python based on Dash from Plotly, to monitor the COVID-19 evolution across the world. The main features of this dashboard is the possibility to fit a sigmoid function for the number of cases or deaths or recovered, using different time interval. The uncertainty of the predictions are also estimated.

The predictions are relatively simple and are estimated just to understand and analyse better the data publicly available. The data used were made available by the Johns Hopkins University [repository](https://github.com/CSSEGISandData/COVID-19).

It is planned to add more interesting graphs to the dashboard in the future.


## Getting Started

Clone the git repo, then install the requirements with pip

```

git clone https://github.com/plotly/dash-sample-apps
cd COVID-19-dashboard
pip install -r requirements.txt

```
Run the app

```

python app.py

```