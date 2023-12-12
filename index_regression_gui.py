'''
Dash App to run Multi-Variate Index Regression on S&P 500, NASDAQ and Russell 2000.
Regression types to run: Linear, Lasso and Ridge.
Top N tickers explaining the Index are also calculated using sklearn RandomForestRegressor's feature importance.
'''
import logging
import diskcache
from datetime import date, datetime
import numpy as np
import pandas as pd
from dash import Dash, DiskcacheManager, html, dcc, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.api as sm
from config import DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION, EQ_INDEX_COLLECTION
from database.data_loader import DataLoader
from utils import get_index_components_from_db, get_ticker_data_from_db, load_index_components_in_db
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONTENT_STYLE = {
    'margin-left': '1rem',
    'margin-right': '1rem',
    'padding': '1rem 1rem',
}

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, background_callback_manager=background_callback_manager)

# Layout of Dash App
app.layout = html.Div([
    html.H1('Multi-Variate Index Regression Dashboard', style={'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div(['Select Dates: ', dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=date(2010, 1, 1),
                max_date_allowed=date(2024, 12, 31),
                initial_visible_month=date(2023, 10, 1),
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                calendar_orientation='vertical',
            ), ]),
            html.Br(),
            'Select Index: ',
            dcc.Dropdown([
                {'label': 'S&P 500', 'value': '^GSPC'},
                {'label': 'NASDAQ', 'value': 'NDX'},
                {'label': 'Russell 2000', 'value': '^RUT'}
            ],
                value='', id='index_selection',
                style={'marginLeft': '48px', 'width': '275px'}
            ),
            html.Br(),
            dbc.Button('Load Market Data in Database!', id='load-mktdata',
                       style={'color': 'white', 'background-color': '#0096FF'}),
            dbc.Spinner(html.Div(id='load_mktdata_status',
                                 children=''), color='success'),
            html.Br(),
            'Select Components: ',
            dcc.Dropdown([],
                         value=[], id='index_component',
                         multi=True,
                         style={'marginLeft': '48px', 'width': '275px'}
                         ),
            html.Br(),
            'Regression Type: ',
            dcc.Dropdown(['Linear', 'Lasso', 'Ridge'],
                         value=[], id='regression_type',
                         style={'marginLeft': '48px', 'width': '275px'}
                         ),
            html.Br(),
            dbc.Button('Run Regression!', id='run-regression',
                       style={'color': 'white', 'background-color': '#0096FF'}),
            html.Br(),
            dbc.Spinner(html.Div(id='regression_status',
                                 children=''), color='success'),
            html.Br(),
            html.Br(),
            'Number of top Index Explainers: ',
            dcc.Input(id='num_explainers', type='number', value=10),
            html.Br(),
            html.Br(),
            dbc.Button('Explain Index!', id='explain_index',
                       style={'color': 'white', 'background-color': '#0096FF'}),
            dbc.Spinner(html.Div(id='index_explain_status',
                                 children=''), color='success'),
        ]),
        dbc.Col([
            html.Div(id='explain_index_results', children='')
        ])
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div(id='regression_results', children='')),
        dbc.Col([dcc.Graph(figure={}, id='scatter_plot'), ]),
    ]),
], style=CONTENT_STYLE)


@callback(
    Output('index_component', 'options'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    Input('index_selection', 'value'),
    prevent_initial_call=True
)
def populate_index_components(start_date: str, end_date: str, selected_index: str) -> list:
    '''
    Returns Index Components which have marketdata in the dropdown.
    :param start_date: Period Start Date
    :param end_date: Period End Date
    :param selected_index: Selected Index
    :return: list
    '''
    tickers = get_index_components_from_db(DB_CONNECTION_STR, DB_NAME, EQ_INDEX_COLLECTION, [selected_index])
    # Only show tickers which have data in the database
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_df = get_ticker_data_from_db(DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION, list(tickers), start_date,
                                      end_date)
    return data_df.columns.values


@callback(
    Output('load_mktdata_status', 'children'),
    Input('load-mktdata', 'n_clicks'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('index_selection', 'value'),
    background=True,
    prevent_initial_call=True
)
def load_mktdata(n_clicks, start_date: str, end_date: str, selected_index: str) -> str:
    '''
    Loads marketdata for Index and its components in the database
    :param n_clicks: Button clicks
    :param start_date: Period start date
    :param end_date: Period end date
    :param selected_index: Index selected
    :return: Status string
    '''
    # Load All Index components in mktdata db
    load_index_components_in_db(DB_CONNECTION_STR, DB_NAME)
    data_loader = DataLoader(DB_CONNECTION_STR, DB_NAME)
    # Load data for individual tickers in mktdata db
    tickers = get_index_components_from_db(DB_CONNECTION_STR, DB_NAME, EQ_INDEX_COLLECTION, [selected_index])
    source_config = {'source': 'yfinance', 'tickers': tickers,
                     'start': datetime.strptime(start_date, '%Y-%m-%d'),
                     'end': datetime.strptime(end_date, '%Y-%m-%d')}
    db_data_config = {'collection_name': EQ_DAILY_COLLECTION, 'collection_index': [('Symbol', 1), ('Date', 1)]}
    logger.info('Loading MarketData in database!')
    data_loader.load_data_db(source_config, db_data_config)
    return f'Market Data Loaded from {start_date} to {end_date} for {selected_index}!'


@callback(
    Output('regression_results', 'children'),
    Output('scatter_plot', 'figure'),
    Output('regression_status', 'children'),
    Input('run-regression', 'n_clicks'),
    State('index_selection', 'value'),
    State('index_component', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('regression_type', 'value'),
    prevent_initial_call=True
)
def run_regression(n_clicks, selected_index: str, tickers: list, start_date: str, end_date: str, regression_type: str):
    '''
    User can run Linear, Lasso, or Ridge regression based on regression_type selected.
    :param n_clicks: Button clicks
    :param selected_index: Selected Index
    :param tickers: List of Index tickers selected by the user
    :param start_date: Period start Date
    :param end_date: Period end date
    :param regression_type: Type of regression to run
    :return: html.Div and scatter plot
    '''
    logger.info(f'Running {regression_type} Regression!')
    all_tickers = tickers + [selected_index]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_df = get_ticker_data_from_db(DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION, all_tickers, start_date, end_date)
    x = data_df[tickers]
    y = data_df[selected_index]
    if regression_type == 'Linear':
        x = sm.add_constant(x)
        reg = sm.OLS(y, x)
        res = reg.fit()
        summary = res.summary().as_html()
        summary = dcc.Markdown(summary, dangerously_allow_html=True)
        prediction = res.predict(x)
        fig = px.scatter(x=prediction, y=y, labels={'x': 'Predicted', 'y': 'Actual'}, trendline='ols')
        return summary, fig, 'Regression Run Finished!'
    elif regression_type == 'Lasso':
        reg = LassoCV(cv=5).fit(x, y)
    elif regression_type == 'Ridge':
        reg = RidgeCV(cv=5).fit(x, y)
    r2 = reg.score(x, y)
    coef = np.append(reg.intercept_, reg.coef_)
    predict = reg.predict(x)
    df = pd.DataFrame(zip(np.append('Intercept', x.columns.values), coef), columns=['Variables', 'Coefficients'])
    dash_tbl = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                                    fixed_rows={'headers': True},
                                    style_cell={'padding-left': '20px', 'padding-right': '20px'},
                                    style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                    style_table={'width': '500px', 'height': '400px', 'overflowY': 'auto'},
                                    )
    summary = html.Div([
                'R-Square: ', r2,
                html.Br(),
                dash_tbl
    ])
    fig = px.scatter(x=predict, y=y, labels={'x': 'Predicted', 'y': 'Actual'}, trendline='ols')
    return summary, fig, 'Regression Run Finished!'

@callback(
    Output('explain_index_results', 'children'),
    Output('index_explain_status', 'children'),
    Input('explain_index', 'n_clicks'),
    State('index_selection', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('num_explainers', 'value'),
    prevent_initial_call=True
)
def run_index_explainer(n_clicks, selected_index: str, start_date: str, end_date: str, num_top_explainers: int):
    '''
    Using RandomForest Regression Feature Importance to explain Index variance.
    :param n_clicks: Button click
    :param selected_index: Selected Index to run explain on.
    :param start_date: Period Start Date
    :param end_date: Period End Date
    :param num_top_explainers: Number of top explainer tickers to include in results
    :return: Dash Table
    '''
    logger.info(f'Running Index Explain on {selected_index}!')
    tickers = get_index_components_from_db(DB_CONNECTION_STR, DB_NAME, EQ_INDEX_COLLECTION, [selected_index])
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_df = get_ticker_data_from_db(DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION, list(tickers), start_date,
                                      end_date)
    x = data_df.drop(selected_index, axis=1)
    y = data_df[selected_index]
    reg = RandomForestRegressor(n_estimators=1000, max_depth=2, random_state=0)
    reg.fit(x, y)
    feature_imp = list(zip(x.columns.values, reg.feature_importances_))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    top_explainers = feature_imp[:num_top_explainers]
    df = pd.DataFrame(top_explainers, columns=['Ticker', 'Importance (%)'])
    df['Importance (%)'] = df['Importance (%)']*100
    df = df.round({'Importance (%)': 2})
    dash_tbl = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                                    fixed_rows={'headers': True},
                                    style_cell={'padding-left': '20px', 'padding-right': '20px'},
                                    style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                    style_table={'width': '500px', 'height': '400px', 'overflowY': 'auto'},
                                    )
    return html.Div([dash_tbl]), 'Index Explain Run Finished!'


if __name__ == '__main__':
    app.run(debug=True)
