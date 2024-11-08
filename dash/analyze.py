from base64 import b64encode
import datetime
from io import StringIO
from pathlib import Path
from time import time
import numpy as np
import copy
import duckdb
import requests
import glasbey
from urllib.parse import unquote, urlencode, urlparse
from concurrent.futures import ThreadPoolExecutor
import openai
import spacy

import pandas as pd
from plotly import graph_objects as go
from plotly import express as px
from dash import ALL, html, dcc, Input, Output, callback, no_update, State, ctx, register_page, Dash, _dash_renderer
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from diskcache import Cache
import srsly
from sturdystats import Index

API_KEY = "XXXXXX"
BASE = "https://sturdystatistics.com/api/text/v1/index"

DELIMITER = "___"
nlp = spacy.load("en_core_web_sm")
cache = Cache("cacheDir/prod")
MIN_CONFIDENCE = 85
QUANTS = [0.0, .2, .4, .6, .8, 1.0]

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/charter-webfont@4/charter.min.css",
        dmc.styles.ALL,
    ],
)
server = app.server
_dash_renderer._set_react_version("18.2.0")



def parse_url_params(params: str) -> dict:
    try:
        pl = params.replace("?", "").replace("=", "&").split("&")
        vals = { pl[i]: unquote(pl[i+1]) for i in range(0, len(pl), 2) }
    except Exception as e:
        print(params, e)
        raise e
    return vals

def summary_prompt(examples):
    ex = "\n\n".join(examples)
    return f"""CONTEXT:
You are a helpful AI researcher who studies news and research articles and produces summary abstracts for human researchers.  It is essential that you provide accurate information, and it is OK to say when you are uncertain.  You always use only the information provided to you and do not speculate, imagine, or hallucinate.

You have detected an important, recurring theme in the news.  Some examples where this theme is discussed in real articles are listed in the EXAMPLES section below.  Given the information provided in the EXAMPLES sections, please answer the question in the REQUEST section.

EXAMPLES:

{ex}


REQUEST:
    Please provide a one paragraph "executive summary" of the theme and all the examples provided.  """



@cache.memoize(expire=3600)
def _get(url, api_key):
    return requests.get(url, headers={"x-api-key": api_key})

def get(url, params, api_key):
    params={**params}
    parsed = urlencode(params)
    res = _get(BASE + url + "?" + parsed, api_key) 
    if res.status_code != 200:
        print(res.content)
    return res

@cache.memoize(expire=3600)
def init_index(index_id, api_key):
    return Index(id=index_id, API_key=api_key, _base_url=BASE)

@cache.memoize(expire=24*3600)
def queryMeta(index_id, api_key, query):
    index =  init_index(index_id, api_key) 
    return index.queryMeta(query)

def duckdbquant(numfields, field):
    return f"list_where({QUANTS[1:]}, [{', '.join(numfields[field])}])[1]"

def topic_count_query(field, filter1, numfields):
    quant = duckdbquant(numfields, field) if field in numfields.keys() else ""
    quant = quant + f" as {field}" if len(quant) > 0 else field
    filter1 = filter1.replace("search(", "search(row_id, ")
    if "search(" in filter1:
        query = f"""
WITH t1 AS (
    SELECT * EXCLUDE({field}), {quant}
    FROM doc_meta
), t2 AS (
    SELECT 
        t1.*, 
        p.paragraph_id,
        p.row_id,
        p.c_mean_avg
    FROM t1 
    INNER JOIN doc_meta_para p
    ON t1.doc_id = p.doc_id
    )
SELECT list_reduce(
        list(c_mean_avg),
        (x, y) -> list_transform(
        list_zip(x,y),
            x -> x[1] + x[2]
        )
    ) as topic_counts,
    {field}
FROM t2
{filter1}
GROUP BY {field} 
"""
    else:
        query = f"""
WITH t1 AS (
    SELECT * EXCLUDE({field}), {quant}
    FROM doc_meta
)
SELECT list_reduce(
        list(sum_topic_counts),
        (x, y) -> list_transform(
        list_zip(x,y),
            x -> x[1] + x[2]
        )
    ) as topic_counts,
    {field}
FROM t1 
{filter1}
GROUP BY {field}"""
    return query

@cache.memoize(expire=24*3600)
def getNumericFields(index_id, api_key):
    index =  init_index(index_id, api_key) 
    NUMERIC_TYPES = ["BIGINT", "DOUBLE", "FLOAT", "HUGEINT", "INTEGER", "UBIGINT", "UHUGEINT", "UINTEGER"]
    num_fields = [x["column_name"] for x in index.queryMeta("DESCRIBE doc_meta") if x["column_type"] in NUMERIC_TYPES] # type: ignore
    
    def _buildNumFilter(f):
        ## TODO cutoff?
        n_distinct = index.queryMeta(f"SELECT approx_count_distinct({f}) as c FROM doc_meta ")[0]["c"]  # type: ignore
        if n_distinct < 15: # type: ignore
            return None
        quant_vals = index.queryMeta(f"SELECT quantile_cont({f}, {QUANTS}) as q FROM doc_meta")[0]["q"].copy() # type: ignore
        quant_vals[-1] +=1
        q_filters = []
        for i in range(1, len(quant_vals)):
            clause = f"( {f} >= {quant_vals[i-1]} AND {f} < {quant_vals[i]} )"
            q_filters.append(clause)
        return q_filters
    tmp = { f: _buildNumFilter(f) for f in num_fields }
    res = { k: v for k,v in tmp.items() if v is not None }
    return res


def procFig(fig):
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
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    return fig

def exportFigButton(fig, filter1, name, className="nu-export-button"):
    buffer = StringIO()
    fig.write_html(buffer)
    data = buffer.getvalue().encode()
    data = b64encode(data).decode()
    f1 = filter1.replace("(", "").replace(")", "") \
            .replace("WHERE", "").replace("'", "")
    return html.A(
        dmc.Button("export", variant="subtle", color="blue", size="compact-xs"),
        href="data:text/html;base64,"+data,
        className=className,
        download=(name + f1 + ".html").replace(" ", "_")
    )

def procFilter(f):
    if f is None: f = ""
    if len(f.strip()) == 0: return ""
    return f

def getgbp():
    NC = np.array([1, 4, 7, 10, 15, 20, 30, 45, 60, 80, 100])
    data_path = Path("glasebyBlocks_"+"_".join([ str(x) for x in NC]) + ".srsly")
    if not data_path.exists():
        data = [np.array(glasbey.create_block_palette([i]*100, grid_size=96)) for i in NC ]
        srsly.write_msgpack(data_path, data)
    data = srsly.read_msgpack(data_path)
    return NC, data



def getPalleteMap(topic_df):
    group_counts = topic_df.value_counts("topic_group_id") + 1
    gcdict = group_counts.to_dict()
    NC, pals = getgbp() 
    group_to_color = dict()
    for i in range(topic_df.topic_group_id.max()+1):
        if i not in gcdict.keys(): continue
        palind = np.argmax(np.where(NC < gcdict[i])[0])+1
        nc = NC[palind]
        start = max(0, ((nc - gcdict[i]) //2) - 1)
        colors = pals[palind][i*nc:(i+1)*nc]
        group_to_color[i] = colors[start:start+gcdict[i]+1].tolist()
    def get_color(group_id):
        tmp = group_to_color[int(group_id)]
        return tmp.pop(0)

    g_to_c = { gid: get_color(gid) for gid in topic_df.topic_group_id.unique() }
    t_to_c = { tid: get_color(gid) for tid, gid in zip(topic_df.topic_id, topic_df.topic_group_id) }
    return t_to_c, g_to_c


@cache.memoize(expire=360000000)
def getTopicWords(index_id, api_key):
    index =  init_index(index_id, api_key) 
    pandata: dict = index.getPandata() # type: ignore
    phi = pandata["model"]["output"]["topic_model/word-branch/phi/mean_avg.npy"]
    beta = pandata["model"]["output"]["topic_model/word-branch/beta/mean_avg.npy"]
    rel = (phi / (beta)**0.75).T
    idx = np.argsort(rel, axis=1)[:,-100:-1]
    topics = np.array(pandata['nu_vocab'])[idx]
    return topics

def getTopicDF(folder_id, query, filter1: str, filter2: str, api_key, limit=40):
    res = get(f"/{folder_id}/topic/diff", dict(query=query, q1="true=true", q2="true=true", limit=10000, min_confidence=-1, cutoff=-1), api_key=api_key)
    topic_df = pd.DataFrame(res.json()["topics"]).dropna()
    topic_group_df = topic_df[["topic_group_short_title", "topic_group_id"]] .drop_duplicates()
    topic_group_df.columns = ["short_title", "topic_group_id"]

    if len(filter1.strip()) > 0 or len(query.strip()) > 0:
        res = get(f"/{folder_id}/topic/diff", dict(query=query, q1=filter1, q2=filter2, limit=limit, min_confidence=MIN_CONFIDENCE), api_key=api_key)
        res = res.json()
        _topic_df = pd.DataFrame(res["topics"]).dropna()
        if len(_topic_df) > 0:
            topic_df = _topic_df
            topic_df["prevalence"] = ( topic_df.confidence - (MIN_CONFIDENCE - 1) ) / ( 100 - (MIN_CONFIDENCE - 1 ) )
    else:
        topic_df.prevalence *= topic_df.topic_group_short_title.apply(
                lambda x: .1 if x == "Other" else 1.0)

    return topic_df, topic_group_df


url_params = State("app-location", "search")

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    return fig

def construct_sunburst_data(folder_id: str, query: str, filter1: str, filter2: str, api_key) ->  dict:
    children = []
    parents = []
    values = []
    ids = []
    colors = []

    topic_df, topic_group_df = getTopicDF(folder_id, query, filter1, filter2, api_key)
    topic_df = topic_df \
            .dropna() \
            .sort_values("prevalence", ascending=False) \
            .drop_duplicates("short_title") \
            .iloc[:150]

    topic_group_set = set(topic_df.topic_group_id.unique())
    topic_group_df = topic_group_df.loc[topic_group_df.topic_group_id.apply(lambda x: x in topic_group_set)]
    t_to_c, g_to_c = getPalleteMap(topic_df) 


    tgs = topic_group_df.to_dict("records")
    tgs = topic_group_df.to_dict("records")
    for tg in tgs:
        label = tg["short_title"].split()
        pos = len(label) // 2
        label = " ".join(label[:pos]) + "<br>" + " ".join(label[pos:])
        children.append(label) 
        parents.append(None)
        values.append(0)
        ids.append(f"topic_group{DELIMITER}{tg['topic_group_id']}")
        colors.append(g_to_c[tg['topic_group_id']])

    topics = topic_df.to_dict('records')
    for t in topics:
        label = t["short_title"].split()
        pos = len(label) // 2
        label = " ".join(label[:pos]) + "<br>" + " ".join(label[pos:])
        children.append(label)
        parents.append(f"topic_group{DELIMITER}{t['topic_group_id']}")
        values.append(t['prevalence'])
        ids.append(f"topic{DELIMITER}{t['topic_id']}{DELIMITER}{t['topic_group_id']}")
        colors.append(t_to_c[t['topic_id']])
    return dict(labels=children, parents=parents, values=values, ids=ids, marker=dict(colors=colors))


def getFilter1(ids, values, queries):
    ids = [ x["index"] for x in ids ]
    filter1 = " "
    queries = queries or []
    queries = [ q.replace("'", "''") for q in queries ]  
    values = [ v.replace("'", "''") if type(v) == str else v for v in values ]  
    query = [ f"( search('{q}') )" for q in queries if len(q.strip()) > 0]
    filter1 += " AND ".join([  f"{field}='{value}' " 
        for field, value in zip(ids, values)
        if value is not None ] + query)
    return filter1



@callback(
    Output("q-live-sunburst", "figure"),
    Output("q-live-sunburst-export-div", "children"),
    url_params,
    Input("q-exact-filter", "data"),
    State(component_id='q-search-query', component_property='value'),
)
def update_sunburst(folder_id, filter_data, queries):
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    if filter_data["triggered_id"] == "q-live-sunburst":
        return no_update, no_update

    filter1 = " AND ".join(list(filter_data["filter1"].values())).strip()
    filter2 = ""
    if len(filter1.strip()) > 0:
        filter2 += f" NOT ({filter1}) "
    queries = queries or []
    query = " ".join(queries or []).strip()


    burst = go.Sunburst(**construct_sunburst_data(folder_id, query, filter1, filter2, api_key), maxdepth=2)
    fig = go.Figure(burst)
    fig = procFig(fig)
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig, exportFigButton(fig, filter1, "sunburst", "nu-sunburst-export-button")

@callback(
    Output(component_id='q-article_excerpts', component_property='children'),
    Output(component_id='q-article_excerpt_prompt', component_property='data'),
        url_params,
        Input("q-exact-filter", "data"),
        State(component_id='q-search-query', component_property='value'),
)
def get_article_excerpts(folder_id, filter_data, queries):
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    max_excerpts_per_doc = params.get("max_excerpts_per_doc", 1)
    date_field = params.get("date_field")
    summary_cutoff = int(params.get("summary_cutoff", 0))

    topic_ids = filter_data["topic_id"]
    filter1 = " AND ".join(list(filter_data["filter1"].values())).strip()
    filter2 = ""
    if len(filter1.strip()) > 0: filter2 += f" NOT ({filter1}) "
    if topic_ids is None and len(filter1.strip()) == 0:
        return [], no_update
    queries = queries or []
    query = " ".join(queries or []).strip()
    if topic_ids is None:
        topic_ids = getTopicDF(folder_id, query, filter1, filter2, api_key)[0].topic_id.apply(str).tolist()
    else: topic_ids = [ str(topic_ids) ]

    search_params = dict(query=query, filters=filter1, max_excerpts_per_doc=max_excerpts_per_doc, topic_ids=",".join(topic_ids))

    res = get(f"/{folder_id}/doc", search_params, api_key=api_key).json()
    df = pd.DataFrame(res["docs"])

    df2 = duckdb.sql("""
    SELECT 
        doc_id,
        sum(nu_score) as nu_score,
        first(metadata) as metadata,
        array_to_string(
            min_by(text, paragraph_id::integer, 5),
            '\n...\n'
        ) as text
    FROM df
    GROUP BY doc_id
    ORDER BY nu_score
    """).to_df()


    topic_words = set()
    if len(topic_ids) == 1:
        topics = getTopicWords(folder_id, api_key)[[ int(t) for t in topic_ids]]
        N = max(100 // len(topic_ids), 12)
        topics = topics[:, :N]
        topic_words = set([t for ts in topics for t in ts])

    def procRow(row):
        text = row["text"]
        metadata = row["metadata"]
        title = metadata.get("title", "")
        link = metadata.get("link", "")
        src = urlparse(link).netloc
        published = metadata.get(date_field or "published", "")

        header = html.Div(
            [ html.Span(html.A(title, href=link, target="_blank"), title=title, className="cardlink"),
              html.Span(src, className="cardinfo", title=src),
              html.Span(published, className="cardinfo"),
             ],
            className="cardheader")

        def procText(text):
            doc = nlp(text)
            queries = query.lower().split(" ")
            ts = [ (
                t.text_with_ws,
                t.lemma_.lower() in queries or t.orth_.lower() in queries,
                t.lemma_.lower() in topic_words or t.orth_.lower() in topic_words,
            )  for t in doc ]
            procT = [ html.B(t[0]) if t[1] else (html.I(t[0]) if t[2] else t[0]) for t in ts ]
            return procT



        x = html.Div([
            header, 
            *[html.P(procText(t)) for t in text.split("\n")]
            ],
            className="nu-article-card"
        )
        return x

    children, prompt = df2.apply(procRow, axis=1).tolist(), summary_prompt(df2.text.tolist())
    if len(df) < summary_cutoff: prompt = ""
    return children, prompt

callback(
    Output("q-gpt_paragraph", "children"),
    Input("q-article_excerpt_prompt", "data"),
    prevent_initial_call=True,
)
def summary(prompt):
    prompt = [
        {"role": "user", "content": prompt},
    ]
    return openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=4000,
    )["choices"][0]["message"]["content"]

@callback(Output(component_id='q-gpt_short_title', component_property='children'),
          url_params,
    Input("q-exact-filter", "data"),
    suppress_callback_exceptions=True,
)
def get_gpt_paragraph(folder_id, filter_data):
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    topic_id = filter_data["topic_id"]
    name = ""
    filter1 = " AND ".join(list(filter_data["filter1"].values())).strip().replace("(", "").replace(")", "")
    if topic_id is not None:
        df, _ = getTopicDF(folder_id, "", "", "", api_key, limit=1000)
        df_row = df.loc[df.topic_id == topic_id].to_dict("records")[0]
        name = df_row['short_title']
    if len(name.strip()) > 0 and len(filter1.strip()) > 0:
        name += " | " 
    name += filter1
    children = [
        html.Div([
            html.H5(f"{name}"),
        ], className="nu-topic-abstract")
    ]
    return children

@callback(
    Output("q-comp_filters", "children"),
    Input("app-location", "search")
)
def initSegControllers(params):
    params = parse_url_params(params)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    comp_fields = (params.get("comp_fields") or "").split(",")
    index =  init_index(folder_id, api_key) 
    numfields = getNumericFields(folder_id, api_key)
    f_to_v = { field: [ x[field] for x in index.queryMeta(f"SELECT distinct {field} FROM doc_meta ORDER BY {field} desc") ] #type: ignore \ 
              if field not in numfields.keys() \
              else numfields[field]  \
              for field in comp_fields if len(field.strip()) > 0}

    def buildMulti(k, v):
        return dmc.MultiSelect(
            data=[{"label": str(x), "value": str(x)} for x in v],
            searchable=True,
            id={"type": "q-comp-filter", "index": k})
    def buildSegCon(k, v):
        isnum = k in numfields.keys()
        return dmc.SegmentedControl(
            data=[{
                "label": f"ALL {k}" + 
                   (" Quantiles" if isnum else "") + 
                   ("s" if not isnum and not k.endswith("s") else ""), 
               "value": None}] + [{
                "label": str(x) if not isnum else QUANTS[i+1], 
                "value": str(x)} for i, x in enumerate(v)],
            orientation="horizontal", 
            id={"type": "q-comp-filter", "index": k})


    controllers = [ html.Div(
        (buildMulti(k,v) if len(v) > 18 else buildSegCon(k,v)),
        className="nu-sb-seg-div",
    )
    for k, v in f_to_v.items()]

    return controllers 

@callback(
    Output("q-search-query", "data"),
    Input("app-location", "search")
)
def initQuerySuggestions(params):
    params = parse_url_params(params)
    suggestions = params.get("search_suggestions")
    res = []
    if suggestions is not None and len(suggestions.strip()) > 0:
        res = suggestions.split(",")
    return res

def get_min_max_dates(folder_id, api_key, date_field):
    index =  init_index(folder_id, api_key) 
    res: dict = index.queryMeta(f"SELECT max({date_field}) as max_date, min({date_field}) as min_date FROM doc_meta \
            WHERE {date_field} != 'NaT' and {date_field} is not NULL")[0] # type: ignore
    max_date = pd.Timestamp(res["max_date"]).date()
    min_date = pd.Timestamp(res["min_date"]).date()
    return min_date, max_date

@callback(
    Output("q-date-range-div", "hidden"),
    Output("q-date-range", "min_date_allowed"),
    Output("q-date-range", "start_date"),
    Output("q-date-range", "max_date_allowed"),
    Output("q-date-range", "end_date"),
    Input("app-location", "search")
)
def initDateRange(params):
    params = parse_url_params(params)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    date_field = params.get("date_field")
    has_date_filter = params.get("has_date_filter", "true").strip() == "true"
    is_simple = params.get("barmode", "simple") == "simple"

    hidden = (date_field is None) or (not has_date_filter)
    if hidden:
        return hidden, None, None, None, None 
    min_date, max_date = get_min_max_dates(folder_id, api_key, date_field)
    yearago = max_date - datetime.timedelta(days=180)
    min_date_init = min_date
    if is_simple:
        min_date_init = max([pd.Timestamp(yearago), pd.Timestamp(min_date)]).date()
    return hidden, min_date, min_date_init, max_date, max_date

@cache.memoize(expire=24*3600)
def getFieldCounts(index_id, api_key, field, limit=12, f=""):
    index =  init_index(index_id, api_key) 
    f = f.replace("search(", "search(row_id, ")
    if "search(" in f:
        query = f"""WITH t1 AS (
    SELECT *
    FROM doc_meta
    ), t2 AS (
    SELECT 
        t1.*, 
        p.paragraph_id,
        p.row_id,
        p.c_mean_avg
    FROM t1 
    INNER JOIN doc_meta_para p
    ON t1.doc_id = p.doc_id
    )
    SELECT {field}, count(distinct doc_id) as c 
    FROM t2 {f} 
    GROUP BY {field} ORDER BY c DESC LIMIT {limit}
        """
    else:
        query = f"""SELECT {field}, count(doc_id) as c 
    FROM doc_meta {f} 
    GROUP BY {field} ORDER BY c DESC LIMIT {limit}"""
    res = index.queryMeta(query)
    return res

def getdiffdf(index_id, api_key, field, value, filter1):
    filter1 = copy.deepcopy(filter1)
    if field in filter1.keys(): filter1.pop(field)
    f1A = " AND ".join(list(filter1.values())).strip()
    f1B = value
    filter1[field] = value 
    filter1 = " AND ".join(list(filter1.values())).strip()
    q21 = f" NOT ({filter1}) "
    q22 = f"(NOT ({f1B})) " + (f"AND {f1A}" if len(f1A) > 0 else "")
    q22 = q21 if "search(" in filter1 else q22

    dfs = []
    for q2 in [q22, q21]:
        data = get(f"/{index_id}/topic/diff", dict(q1=filter1, q2=q2, limit=1000, min_confidence=90, cutoff=2.0), api_key=api_key)
        if data.status_code != 200:
            return None
        data = data.json()
        dfs.append(pd.DataFrame(data["topics"]))
        if len(dfs[-1]) == 0: return None
    df, valid_topics = dfs[0], set(dfs[1].topic_id)
    df = df.loc[df.topic_id.apply(lambda x: x in valid_topics)]
    if len(df) == 0:
        return None
    df = df.sort_values(["confidence", "prevalence"], ascending=False)
    df = df.drop(columns=["executive_paragraph_summary", "one_sentence_summary"])
    return df

@cache.memoize(expire=24*3600)
def getFieldDiffDF(folder_id, api_key, field, filter1, date_field):
    filter1 = copy.deepcopy(filter1)
    numfields = getNumericFields(folder_id, api_key)
    if field == date_field:
        f = " AND ".join(list(filter1.values())).strip()
        if len(f) > 0: f = " WHERE " + f
        tmpdf1 = pd.DataFrame(getFieldCounts(folder_id, api_key, field, 10000000, f))
        tmpdf2, _ = processDateField(tmpdf1, field)
        datedf = tmpdf2.loc[tmpdf2.c > 0].sort_values(field)
        dates = [None, *datedf[field].apply(lambda x: str(x.date())).to_list(), None]
        def procDate(i):
            f = []
            if i == 1:
                if tmpdf1.loc[tmpdf1[field] <= dates[i]].c.sum() == 0: return None
            if i == len(dates)-1:
                if tmpdf1.loc[tmpdf1[field] > dates[i-1]].c.sum() == 0: return None
            if dates[i-1] is not None: f+= [f" {date_field} > '{dates[i-1]}' "]
            if dates[i] is not None: f+= [f" {date_field} <= '{dates[i]}' "]
            return " AND ".join(f)
        res = [procDate(i) for i in range(1, len(dates)) ]
        res = [ r for r in res if r is not None ]
    elif field in numfields.keys():
        res = numfields[field]
    else:
        res = getFieldCounts(folder_id, api_key, field, 12)
        res = [ r[field] for r in res ] # type: ignore
        res = [ f"{field}='{r}'" for r in res ]
    with ThreadPoolExecutor(max_workers=len(res)) as executor:
        func = lambda x: getdiffdf(folder_id, api_key, field, x, filter1)
        dfs = executor.map(func, res)
    dfs = [x for x in dfs if x is not None]
    if len(dfs) == 0: return []
    df = pd.concat(dfs) # type: ignore
    df = duckdb.sql("SELECT topic_id, topic_group_id, short_title, topic_group_short_title, sum(confidence+prevalence) as confidence \
    FROM df GROUP BY topic_id, topic_group_id, short_title, topic_group_short_title \
    ORDER BY confidence desc").to_df().drop_duplicates("short_title")
    ## TODO drop other?
    #df = df.loc[df.topic_group_short_title != "Other"]
    return df


def getDateInterval(datemax, datemin):
    diff = (pd.Timestamp(datemax) - pd.Timestamp(datemin)).days
    interval = "D"
    if diff < 14: interval = "D" 
    elif diff < 70: interval = "W" 
    elif diff < 120: interval = "SME" 
    elif diff < 366: interval = "ME" 
    elif diff < 2*360: interval = "QE" 
    elif diff < 4*360: interval = "2QE" 
    elif diff < 10*360: interval = "YE" 
    elif diff < 20*360: interval = "2YE" 
    elif diff < 40*360: interval = "4YE" 
    else: interval = "5YE" 
    return interval

def processDateField(df, field):
    df[field] = df[field].apply(lambda x: pd.Timestamp(pd.Timestamp(x).date()) )
    interval = getDateInterval(df[field].max(), df[field].min())
    df = df.resample(interval, on=field).sum().reset_index()
    return (df, sorted(list(set(df[field].tolist()))))

@callback(
    Output("q-preload-big-plots-null-output", "children"),
    url_params,
    Input("q-exact-filter", "data"),
)
def preloadBigPlotCache(params, filter_data):
    params = parse_url_params(params)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    date_field = params.get("date_field")
    discrete_fields = params.get("stacked_plot_fields", None)
    if discrete_fields is None: discrete_fields = params.get("bar_plot_fields", "")
    discrete_fields = discrete_fields.split(",")
    if date_field is not None: discrete_fields.append(date_field)
    discrete_fields = [ x.strip() for x in discrete_fields
        if len(x.strip()) > 0 ]
    complex_bar = params.get("barmode", "complex") == "complex"

    def build_topic_df(field):
        tmpf1 = copy.deepcopy(filter_data["filter1"])
        if field in tmpf1.keys() and field != date_field: tmpf1.pop(field)
        tmpf1 = dict(sorted(tmpf1.items()))
        return getFieldDiffDF(folder_id, api_key, field, tmpf1, date_field)
    def build_topic_df2(_):
        tmpf1 = copy.deepcopy(filter_data["filter1"])
        filter1 = " AND ".join(tmpf1.values()).strip()
        filter2 = f" NOT ({filter1}) "
        topic_df, _ = getTopicDF(folder_id, "", filter1, filter2, api_key, 40)
        return topic_df.drop_duplicates("short_title") 
    if complex_bar or len(filter_data["filter1"].values()) == 0:
        f_to_df_func = build_topic_df
    else: f_to_df_func = build_topic_df2
    _ = { field: f_to_df_func(field) for field in discrete_fields} 
    return no_update

@callback(
        Output("q-big-plots", "children"),
        url_params,
        Input("q-exact-filter", "data"),
        Input(component_id='q-norm-big-plot-bool', component_property='value'),
        State("q-big-plots", "children"),
        prevent_initial_call=True,
        suppress_callback_exceptions=True,
)
def bigPlots(folder_id, filter_data, normalize: bool, old_plots):
    t1 = time()
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    date_field = params.get("date_field")
    complex_bar = params.get("barmode", "complex") == "complex"
    discrete_fields = params.get("stacked_plot_fields", None)
    if discrete_fields is None: discrete_fields = params.get("bar_plot_fields", "")
    discrete_fields = discrete_fields.split(",")
    if date_field is not None: discrete_fields.append(date_field)
    discrete_fields = [ x.strip() for x in discrete_fields
        if len(x.strip()) > 0 ]

    numfields = getNumericFields(folder_id, api_key)
    g_to_c_sb = dict() ## Hack maybe just dont use sb

    def build_topic_df(field):
        tmpf1 = copy.deepcopy(filter_data["filter1"])
        if field in tmpf1.keys() and field != date_field: tmpf1.pop(field)
        tmpf1 = dict(sorted(tmpf1.items()))
        return getFieldDiffDF(folder_id, api_key, field, tmpf1, date_field)
    def build_topic_df2(_):
        tmpf1 = copy.deepcopy(filter_data["filter1"])
        filter1 = " AND ".join(tmpf1.values()).strip()
        filter2 = f" NOT ({filter1}) "
        topic_df, _ = getTopicDF(folder_id, "", filter1, filter2, api_key, 40)
        return topic_df.drop_duplicates("short_title") 
    if complex_bar or len(filter_data["filter1"].values()) == 0:
        f_to_df_func = build_topic_df
    else: f_to_df_func = build_topic_df2

    trigger_field = None
    tdata = filter_data["triggered_id"]
    if isinstance(tdata, dict) and tdata["type"] == "q-bar-plot-lg":
        trigger_field = tdata["index"]

    plots = []
    for field in discrete_fields:
        if field == trigger_field:
            pos = [ i for i, p in enumerate(old_plots) if p["props"].get("id", dict()) == tdata][0]
            plots.extend(old_plots[pos:pos+2])
            continue

        topic_df = f_to_df_func(field)

        filter1 = copy.deepcopy(filter_data["filter1"])
        if field in filter1.keys(): filter1.pop(field)
        filter1 = dict(sorted(filter1.items()))
        filter1 = " AND ".join(list(filter1.values())).strip()
        if len(filter1.strip()) > 0: filter1 = " WHERE " + filter1
        ## TODO hack

        query = topic_count_query(field, filter1, numfields)
        df = pd.DataFrame(queryMeta(folder_id, api_key, query))

        ## TODO hack
        for i in range(len(df.topic_counts.iloc[0])):
            df[f"topic_{i}"] = df.topic_counts.apply(lambda x: x[i])
            if i %10 == 0: df= df.copy()
        if len(df) <= 1: continue
        if field == date_field: 
            df, intervals = processDateField(df, date_field)
        df = df[[field]+ [ f"topic_{i}" for i in topic_df.topic_id ]]
        df.columns = [field]+ [ l for l in topic_df.short_title]
        sumL = [ df[l].var()*v for l, v in zip(topic_df.short_title, topic_df.confidence) ][:max(len(topic_df)//2, 100 if complex_bar else 40)]
        inds_set = set(np.argsort(sumL)[::-1][:])
        df = df[[field]+ [ l for i, l in enumerate(topic_df.short_title) if i in inds_set] ]
        df = df.set_index(field)
        df = df.stack().rename_axis((None, None)).reset_index(level=1).reset_index()
        df.columns = [field, "topic", "occurrences"]
        occ = "(t1.occurrences / o_sum)" if normalize else "t1.occurrences"
        df = duckdb.sql(f"""
SELECT t1.topic, t1.{field}, {occ} as occurrences FROM 
df t1 
INNER JOIN ( 
    SElECT {field}, o_sum FROM ( 
        SELECT {field}, sum(occurrences) as o_sum 
        FROM df
        GROUP BY {field}
    )
    ORDER BY o_sum DESC
    LIMIT 12
) t2 
ON t1.{field} = t2.{field}""").to_df()
        topic_dict = topic_df.set_index("short_title").to_dict("index")
        for c in ["topic_id", "topic_group_id", "topic_group_short_title"]:
            df[c] = df.topic.apply(lambda x: topic_dict[x][c])
        g_to_c = {i: v for i, v in enumerate(getgbp()[1][0])}
        g_to_c = {**g_to_c, **g_to_c_sb}

        def procFilterData(x):
            ## TODO support pos == 0 better
            if field != date_field: 
                if field not in numfields.keys():
                    return f"({field}='{x[field]}')"
                return numfields[field][QUANTS.index(x[field]) - 1]
            val = f"{field}<='{str(x[field].date())}' "
            pos = intervals.index(x[field])
            if pos > 0: val += f"AND {field} >='{str(intervals[pos-1].date())}'"
            return f"({val})"
        df["custom_data"] = df.apply(lambda x: [dict(
            filter_data={field: procFilterData(x)},
            topic_id=x['topic_id'], index=field,), x["topic"], x["topic_group_short_title"]], axis=1) 
        df["color"] = df.topic_group_id.apply(lambda x: g_to_c[x])
        ranked_tgs = duckdb.sql("SELECT topic_group_id, sum(occurrences) as occ FROM df GROUP BY topic_group_id ORDER BY occ desc").to_df()
        fig = go.Figure()
        for tg in ranked_tgs.topic_group_id:
            tmp = df.loc[df.topic_group_id == tg].copy()
            tid_to_occ = duckdb.sql("SELECT topic_id, sum(occurrences) as occ FROM tmp GROUP BY topic_id").to_df().set_index("topic_id").occ.to_dict()
            tmp["tot_occ"] = -1*tmp.topic_id.apply(tid_to_occ.get)
            tmp = tmp.sort_values([field, "tot_occ"], ascending=True) 
            name = tmp.topic_group_short_title.iloc[0]
            fig.add_trace(
                go.Bar(name=name, x=tmp[field], y=tmp.occurrences, customdata=tmp.custom_data, marker=dict(color=tmp.color),
                       hovertemplate="<b>%{customdata[2]}</b><br>" +
                       "%{customdata[1]}" +
                       "<extra></extra>"
            ))  
        fig = procFig(fig)
        fig.update_layout(showlegend=False)
        fig.update_layout(barmode='stack')
        plots.append(dcc.Graph(figure=fig, id={"type": "q-bar-plot-lg", "index":field}, clickData=None, config={'displayModeBar':False}))
        plots.append(exportFigButton(fig, filter1, field+"_comparison"))

    return plots

@callback(
        Output("q-plots", "children"),
        url_params,
        Input("q-exact-filter", "data"),
        State(component_id='q-live-sunburst', component_property='figure'),
        suppress_callback_exceptions=True,
)
def getPlots(folder_id, filter_data, figure):
    params = parse_url_params(folder_id)
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    date_field = params.get("date_field")
    trend_date = params.get("trend_date", "true").lower() == "true" and date_field is not None and len(date_field.strip()) > 0
    discrete_fields = params.get("bar_plot_fields", "").split(",")

    numfields = getNumericFields(folder_id, api_key)
    index = init_index(folder_id, api_key) 
    topic_id = filter_data["topic_id"]
    if topic_id is None: return []

    data = figure["data"][0]
    colors = data["marker"]["colors"]
    ids = data["ids"]
    g_to_c_sb = { int(tg.split(DELIMITER)[1]): c for tg, c in zip(ids, colors) if tg.startswith("topic_group") }
    g_to_c_sb = dict() ## Hack maybe just dont us96sb
    g_to_c = {i: v for i, v in enumerate(getgbp()[1][0])} # type: ignore
    g_to_c = {**g_to_c, **g_to_c_sb}

    tdf, _= getTopicDF(folder_id, "", "", "", api_key, 10000)
    tinfo = tdf.loc[tdf.topic_id == topic_id].iloc[0]
    tg = tinfo.topic_group_id
    color = g_to_c[tg]

    discrete_fields = [ x.strip() for x in 
       ([date_field if trend_date else ""] + discrete_fields)
        if len(x.strip()) > 0 ]
    ## TODO sort discrete fields by not filtered on??
    #discrete_fields = sorted(discrete_fields, key=lambda x: (x in filter_data["filter1"].keys(), discrete_fields.index(x)) )


    plots = []
    for field in discrete_fields:
        filter1 = copy.deepcopy(filter_data["filter1"])
        if field in filter1.keys(): filter1.pop(field)
        filter1 = " AND ".join(list(filter1.values())).strip()
        if len(filter1.strip()) > 0: filter1 = " WHERE " + filter1
        ## TODO hack
        filter1 = filter1.replace("search(", "search_doc_meta(doc_id, ")

        if field not in numfields.keys():
            query = f"SElECT {field}, SUM(sum_topic_counts[{topic_id+1}]) as occurrences FROM doc_meta {filter1} GROUP BY {field} ORDER BY {field} asc"
            df = pd.DataFrame(queryMeta(folder_id, api_key, query))
        else:
            quant = duckdbquant(numfields, field) 
            query = f"with t as (SElECT {quant} as {field}, * FROM doc_meta) SELECT {field}, SUM(sum_topic_counts[{topic_id+1}]) as occurrences FROM t {filter1} GROUP BY {field} ORDER BY {field} asc"
            df = pd.DataFrame(queryMeta(folder_id, api_key, query))
            df.columns = [field, "occurrences"]
        if len(df) <= 1: continue
        if field == date_field:
            df, intervals = processDateField(df, date_field)
        df = duckdb.sql(f"""select * FROM df where {field} in 
        ( SElECT {field} FROM ( 
            SELECT {field}, sum(occurrences) as n 
            FROM df
            GROUP BY {field}
            )
        ORDER BY n DESC
        LIMIT 12
        )""").to_df()
        def procFilterData(x):
            ## TODO support pos == 0 better
            if field != date_field: 
                if field not in numfields.keys():
                    return f"({field}='{x[field]}')"
                return numfields[field][QUANTS.index(x[field]) - 1]
            val = f"{field}<='{str(x[field].date())}' "
            pos = intervals.index(x[field])
            if pos > 0: val += f"AND {field} >='{str(intervals[pos-1].date())}'"
            return f"({val})"
        df["custom_data"] = df.apply(lambda x: dict(filter_data={field: procFilterData(x) }, topic_id=topic_id, index=field), axis=1) 
        fig = px.bar(df, x=field, y="occurrences", custom_data="custom_data")
        fig = procFig(fig)
        fig.update_traces(marker_color=color)
        plots.append(dcc.Graph(figure=fig, id={"type": "q-bar-plot-sm", "index":field}, clickData=None, config={'displayModeBar':False}))
        plots.append(exportFigButton(fig, filter1, tinfo.short_title + "_" + field+"_comparison"))
    return plots


@callback(
    Output(component_id='q-exact-filter', component_property='data'),
    url_params,
    Input(component_id='q-exact-filter', component_property='data'),
    Input(component_id='q-search-query', component_property='value'),
    Input(dict(type="q-comp-filter", index=ALL), "id"),
    Input(dict(type="q-comp-filter", index=ALL), "value"),
    Input('q-date-range', 'start_date'),
    Input('q-date-range', 'end_date'),
    State('q-date-range', 'max_date_allowed'),
    State('q-date-range', 'min_date_allowed'),
    Input(component_id='q-live-sunburst', component_property='clickData'),
    Input(dict(type="q-bar-plot-sm", index=ALL), "clickData"),
    Input(dict(type="q-bar-plot-lg", index=ALL), "clickData"),
)
def buildFilter(params, old_filter, queries, ids, values, start_date, end_date, max_date, min_date, clickDataSB, clickDataBarSMs, clickDataBarLGs):
    params = parse_url_params(params)
    date_field = params.get("date_field")
    folder_id = params["folder_id"]
    api_key = params.get("api_key", API_KEY)
    numfields = getNumericFields(folder_id, api_key)

    filter1 = dict()
    ## Process Input Search Queries
    queries = queries or []
    if len(queries) > 0:
        queries = [ q.replace("'", "''") for q in queries ]  
        queries = [ " AND ".join([ f"( search('{q}') )" for q in qs.split(" ") if len(q.strip()) > 0 ]) 
                   for qs in queries ]
        query = " OR ".join([ f"({q})" for q in queries if len(q.strip()) > 0])
        filter1["search_queries"] = f"({query})"

    ## Process Comp Fields
    ids = [ x["index"] for x in ids ]
    values = [ [v] if type(v) == str else v for v in values ]
    values = [ [ val.replace("'", "''") for val in v] if v is not None else v for v in values ]
    filter1.update(
            {field: f"({field} in {tuple(value)} )" if len(value) > 1
            else f"({field}='{value[0]}')" if field not in numfields.keys()
             else value[0]
            for field, value in zip(ids, values)
        if value is not None and len(value) > 0 }
    )

    ## Process Date Input
    ## Hard code the date field so that it is always applied 
    if date_field is not None and len(date_field.strip()) > 0:
        date_filter = f"({date_field} >= '{start_date}' and {date_field} <= '{end_date}')"
        if start_date > min_date or end_date < max_date:
            filter1["hard_coded_"+date_field] = date_filter

    topic_id = None
    if ctx.triggered_id == "q-live-sunburst":
        filter1 = old_filter["filter1"]  
        point = clickDataSB["points"][0]
        pid = point["id"].split(DELIMITER)
        uid = int(pid[1])
        if pid[0] == "topic":
            topic_id = int(uid)
    elif isinstance(ctx.triggered_id, dict):
        tmpClick = []
        if ctx.triggered_id["type"] == "q-bar-plot-sm": tmpClick = clickDataBarSMs 
        if ctx.triggered_id["type"] == "q-bar-plot-lg": tmpClick = clickDataBarLGs
        if len(tmpClick) > 0:
            filter1 = old_filter["filter1"]  
            tmp = [ x["points"][0]["customdata"][0] for x in tmpClick if x is not None ]
            ## This decides whether to stack filters or to only apply the last one
            custom_data = [ x for x in tmp if x["index"] == ctx.triggered_id["index"] ] ## Not sure if this is needed
            if len(custom_data) > 0: 
                custom_datum = custom_data[0]
                filter1 = {**filter1, **custom_datum["filter_data"],}
                topic_id = custom_datum["topic_id"]
    return dict(filter1=filter1, topic_id=topic_id, triggered_id=ctx.triggered_id, triggered_prop_ids=ctx.triggered_prop_ids)


@callback(
    Output("q-gpt_short_title-metadiv-ac", "children"),
    Output("q-big-plots-metadiv-ac", "children"),
    Output("q-plots-metadiv-ac", "children"),
    Output("q-gpt_short_title-metadiv-sb", "children"),
    Output("q-big-plots-metadiv-sb", "children"),
    Output("q-plots-metadiv-sb", "children"),
    Output("q-sunburst-div", "hidden"),
    url_params,
    Input("q-hidden-input12", "id"),
)
def init_bar_plot_divs(urlparams, _):
    params = parse_url_params(urlparams)
    discrete_fields = params.get("bar_plot_fields", "")
    sb = params.get('sb', "true").lower() == 'true'
    bleft = False if sb else True

    bigplots = html.Div(id="q-big-plots")
    title = html.Div(id="q-gpt_short_title")
    smplots = html.Div(id="q-plots")
    res = [title, bigplots, smplots] 
    if bleft: res = ([None] *3) + res
    else: res = res + ([None]*3)
    res.append(not sb) # type: ignore
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6]

################################################################################
## sunburst column

# put the plot in a "nu-full-img" container to absorb the full
# available space
sunburst_plot = \
    html.Div(
        [ dcc.Graph(id="q-live-sunburst", clear_on_unhover=True, config={'displayModeBar':False}),
         html.Div(id="q-live-sunburst-export-div"),
         ],
        className="nu-full-img",
        id="q-sunburst-full-img"
    )

# search bar
search_bar = \
    html.Div([
    dmc.TagsInput(
        id="q-search-query",
        placeholder="Enter or select a search query",
        data=None,)
    ],
        className="nu-search-bar",
        hidden=False,
        id="q-search-query-div",
    )

filters = html.Div(
    [
      search_bar,
      html.Div(id="q-comp_filters", className="nu-sb-segcontroller"),
      html.Div(
          dcc.DatePickerRange(
            id='q-date-range',
           ),
          id="q-date-range-div",
          hidden=True,
       ),
    ],
    id="q-filters-div",
)

sunburst_column = \
    html.Main(
        [ html.Div(id='my-output', hidden=True),
           html.Div([ sunburst_plot, html.Div()],
                   className="sunburst", id="q-sunburst-div"),

          html.Div(id="q-gpt_short_title-metadiv-sb", className="nu-sticky-short-title"),
           html.Div(id="q-big-plots-metadiv-sb"),
           html.Div(id="q-plots-metadiv-sb"),
         ],
        className="nu-column-articles"
    )

tooltip = dbc.Tooltip(
    children="Click on the Graph to Organize the Content",
    id="q-live-sunburst-dd-graph-tooltip",
    target="q-big-plots",
    placement="top-end", ## TODO fix placement or don't use tooltip?
    is_open=True,
)

################################################################################
## articles column

norm_seg_controller = html.Div(
    dmc.SegmentedControl(
        data=[dict(label="Plot Count", value=False), dict(label="Plot Pct", value=True)][::-1],
        value=True,
        orientation="horizontal", 
        id="q-norm-big-plot-bool",
    ),
    className="nu-sb-seg-div",
)

articles_column = \
    html.Aside(
        [ 
          filters,
          html.Div(id="abstract_top", hidden=False),
          norm_seg_controller,
          html.Div(id="q-gpt_short_title-metadiv-ac", className="nu-sticky-short-title"),
          html.Div(id="q-big-plots-metadiv-ac"),
          html.Div([
              html.H4(id="q-gpt_paragraph_title"),
              html.Div(id="q-gpt_paragraph"),
          ],
              className="nu-live-gpt-summary"
          ),
          html.Div(id="q-plots-metadiv-ac", ),
          dcc.Loading(html.Div(id="q-article_excerpts")),
          html.Div(id="q-preload-big-plots-null-output"),
         ],
        className="nu-column-articles",
    )


################################################################################
## page layout

app.layout = \
dmc.MantineProvider(
    html.Div(
        [ html.Div(id="hidden-output-1", hidden=True),
          html.Div(id="q-hidden-input12", hidden=True),
          html.Div(id="hidden-output-search-doc", hidden=True),
          html.Div([ sunburst_column, articles_column],
                   className="nu-content"),
          dcc.Interval(
              id='startup-interval-component',
              interval=10e10, # only call on startup
              n_intervals=0,  # initialize with 0
              max_intervals=1),
            html.Div(id="hiddenInput"),
            dcc.Store(id="q-article_excerpt_prompt", storage_type="memory", data=""),
            dcc.Store(id="q-plot-filter", storage_type="memory", data=dict()),
            dcc.Store(id="cur-topic", storage_type="memory", data=dict()),
            dcc.Store(id="q-exact-filter", storage_type="memory", data=dict()),
            html.Div(id="q-hidden-output-stream-gpt", hidden=True),
            tooltip,
            dcc.Location(
                id="app-location",
            ),
         ],
    )
)

if __name__ == '__main__':
    app.run(debug=True)

