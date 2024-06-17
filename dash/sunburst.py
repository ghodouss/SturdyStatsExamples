"""
Creates a Sunburst Plot on top of an index of your choice.

Go to URL: http://localhost:8050/?folder_id={index_id}
"""

### TODO: SET YOU API KEY HERE!!
api_key = "XXXXXXXXXXXXX"

import requests
from urllib.parse import urlencode, urlparse, unquote
from datetime import date, timedelta

import pandas as pd
import colorcet as cc
from plotly import graph_objects as go
from dash import html, dcc, Input, Output, callback, State, register_page, Dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from diskcache import Cache


BASE = "https://sturdystatistics.com/api/text/v1/index"
cache = Cache("cachingHelpsSpeed")
DATE_FIELD = "published"

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/charter-webfont@4/charter.min.css"
    ],
)

def post(url, params):
    res = requests.post(BASE + url , json={"api_key": api_key, **params})
    return res

@cache.memoize(expire=120)
def _get(url):
    return requests.get(url)

def get(url, params):
    params={"api_key": api_key, **params}
    parsed = urlencode(params)
    res = _get(BASE + url + "?" + parsed ) 
    return res

DELIMITER = "___"
def parse_url_params(params: str) -> dict:
    try:
        pl = params.replace("?", "").replace("=", "&").split("&")
        vals = { pl[i]: unquote(pl[i+1]) for i in range(0, len(pl), 2) }
    except Exception as e:
        print(params, e)
        raise e
    return vals

daysago = lambda x: str(date.today() - timedelta(x))
_startdates = [
    ("All Time", ""),
    ("Past Week", daysago(7)),
    ("Past Month", daysago(31)),
    ("Past 3 Months", daysago(30*3)),
    ("Past Year", daysago(365)),
]
startdates = [
    {"label": tr[0], "value": tr[1]}
    for tr in _startdates
]

def startdate_to_filter(startdate):
    if DATE_FIELD is None:
        return ""
    if len(startdate) == 0:
        return ""
    else:
        return f"{DATE_FIELD} > '{startdate}' or {DATE_FIELD} is NULL"

def getTopicDF(folder_id, query, filters: str):
    res = get(f"/{folder_id}/doc", dict(query=query, filters=filters))
    res = res.json()
    topic_df = pd.DataFrame(res["topics"]).dropna()
    topic_df.prevalence *= topic_df.topic_group_short_title.apply(
            lambda x: .1 if x == "Other" else 1.0)
    topic_group_df = pd.DataFrame(res["topic_groups"])
    return topic_df, topic_group_df


url_params = State("app-location", "search")
gc = cc.palette.glasbey_hv

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    return fig

def construct_sunburst_data(folder_id: str, query: str, filters: str) ->  dict:
    children = []
    parents = []
    values = []
    ids = []
    colors = []

    topic_df, topic_group_df = getTopicDF(folder_id, query, filters)
    topic_df = topic_df.sort_values("prevalence", ascending=False).iloc[:100]

    tgs = topic_group_df.to_dict("records")
    for tg in tgs:
        label = tg["short_title"].split()
        pos = len(label) // 2
        label = " ".join(label[:pos]) + "<br>" + " ".join(label[pos:])
        children.append(label) 
        parents.append(None)
        values.append(0)
        ids.append(f"topic_group{DELIMITER}{tg['topic_group_id']}")
        colors.append(gc[tg['topic_group_id']])

    topics = topic_df.to_dict('records')
    for t in topics:
        label = t["short_title"].split()
        pos = len(label) // 2
        label = " ".join(label[:pos]) + "<br>" + " ".join(label[pos:])
        children.append(label)
        parents.append(f"topic_group{DELIMITER}{t['topic_group_id']}")
        values.append(t['prevalence'])
        ids.append(f"topic{DELIMITER}{t['topic_id']}{DELIMITER}{t['topic_group_id']}")
        colors.append(gc[t['topic_group_id']])
    return dict(labels=children, parents=parents, values=values, ids=ids, marker=dict(colors=colors))


@callback(
    Output("ds-live-sunburst", "figure"),
    url_params,
    Input("ds-search-query", "n_submit"),
    State("ds-search-query", "value"),
    Input("ds-search-startdates", "value"),
)
def update_sunburst(folder_id, _, query, startdate):
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    if query is None:
        query = ""
    query = query.strip()
    filters = startdate_to_filter(startdate)
    burst = go.Sunburst(**construct_sunburst_data(folder_id, query, filters), maxdepth=2)
    fig = go.Figure(burst)
    fig.update_layout(
        font_family="Charter",
        autosize=True,
        plot_bgcolor= "rgba(0, 0, 0, 0)",
        paper_bgcolor= "rgba(0, 0, 0, 0)",
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig

@callback(Output(component_id='ds-article_excerpts', component_property='children'),
          url_params,
          Input(component_id='ds-live-sunburst', component_property='clickData'),
          Input("ds-search-query", "n_submit"),
          State("ds-search-query", "value"),
          Input("ds-search-startdates", "value"),
)
def get_article_excerpts(folder_id, clickDataSB, _, query, startdate):
    folder_id = parse_url_params(folder_id)["folder_id"]
    if query is None and clickDataSB is None:
        return []
    if query is None:
        query = ""
    filters = startdate_to_filter(startdate)

    search_params = dict(query=query.strip(), filters=filters)
    if clickDataSB is not None:
        point = clickDataSB["points"][0]
        pid = point["id"].split(DELIMITER)
        uid = int(pid[1])
        if pid[0] == "topic":
            search_params["topic_ids"] = uid 
        elif pid[0] == "topic_group":
            search_params["topic_group_id"] = uid
    res = get(f"/{folder_id}/doc", search_params).json()
    df = pd.DataFrame(res["docs"])

    def procRow(row):
        text = row["text"]
        metadata = row["metadata"]
        title = metadata.get("title", "")
        link = metadata.get("link", "")
        src = urlparse(link).netloc
        published = metadata.get("published", "")

        header = html.Div(
            [ html.Span(html.A(title, href=link, target="_blank"), title=title, className="cardlink"),
              html.Span(src, className="cardinfo", title=src),
              html.Span(published, className="cardinfo"),
             ],
            className="cardheader")

        x = html.Div([
            header, 
            *[html.P(t) for t in text.split("\n")]
            ],
            className="nu-article-card"
        )
        return x
    return df.apply(procRow, axis=1).tolist()

@callback(Output(component_id='ds-gpt_paragraph', component_property='children'),
          url_params,
          Input(component_id='ds-live-sunburst', component_property='clickData'),
)
def get_gpt_paragraph(folder_id, clickData):
    folder_id = parse_url_params(folder_id)["folder_id"]
    if clickData is None:
        return []
    point = clickData["points"][0]
    pid = point["id"].split(DELIMITER)
    if pid[0] == "topic":
        df, _ = getTopicDF(folder_id, "", "")
        df_row = df.loc[df.topic_id == int(pid[1])].to_dict("records")[0]
    elif pid[0] == "topic_group":
        _, df = getTopicDF(folder_id, "", "")
        df_row = df.loc[df.topic_group_id == int(pid[1])].to_dict("records")[0]
    else:
        return []

    name = df_row['short_title']
    form = df_row['executive_paragraph_summary']
    children = [
        html.Div([
            html.H4(f"{name}"),
            html.P(f"{form}"),
        ], className="nu-topic-abstract")
    ]
    return children


################################################################################
## sunburst column

# put the plot in a "nu-full-img" container to absorb the full
# available space
sunburst_plot = \
    html.Div(
        [ dcc.Graph(id="ds-live-sunburst", clear_on_unhover=True, config={'displayModeBar':False}),
         html.Div(),
         ],
        className="nu-full-img"
    )

# search bar
search_bar = \
    html.Div(
        [ dbc.Input(id="ds-search-query",
                    placeholder="Enter a Search Query...",
                    type="text",)
         ],
        className="nu-search-bar",
    )

sunburst_column = \
    html.Main(
        [ html.Div(id='my-output', hidden=True),
          search_bar,
          html.Div([ sunburst_plot, html.Div()],
                   className="sunburst")
         ],
        className="nu-column-sunburst"
    )


################################################################################
## articles column

startdate_choice = \
    html.Div(
        [ dmc.Select(
            data=startdates,
            value="",
            clearable=False,
            searchable=False,
            id="ds-search-startdates"),
         ],
    className="engine-dropdown",
    hidden=DATE_FIELD is None,
)


articles_column = \
    html.Aside(
        [ 
         startdate_choice,
          html.Div(id="abstract_top", hidden=False),
          html.Div(id="ds-gpt_paragraph"),
          dcc.Loading(html.Div(id="ds-article_excerpts")),
         ],
        className="nu-column-articles",
    )


################################################################################
## page layout

app.layout = \
    html.Div(
        [ html.Div(id="hidden-output-1", hidden=True),
          html.Div(id="hidden-output-search-doc", hidden=True),
          html.Div([ sunburst_column, articles_column],
                   className="nu-content"),
          dcc.Interval(
              id='startup-interval-component',
              interval=10e10, # only call on startup
              n_intervals=0,  # initialize with 0
              max_intervals=1),
            html.Div(id="hiddenInput"),
          dcc.Location(
            id="app-location",
            refresh="callback-nav"
        ),
         ],

    )

if __name__ == '__main__':
    app.run(debug=True)
