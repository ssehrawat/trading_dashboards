'''
Dash App implementing Pairs Trading Dashboard.
'''
from datetime import date, datetime
import json
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from config import DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION, INDEX_LIST, EQ_INDEX_COLLECTION
from database.data_loader import DataLoader
from utils import get_index_components_from_db, load_index_components_in_db
from pairs_trading import PairsTrading, BackTesting

CONTENT_STYLE = {
    'margin-left': '1rem',
    'margin-right': '1rem',
    'padding': '1rem 1rem',
}

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Pairs Trading Dashboard', style={'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div(['Train Date: ', dcc.DatePickerRange(
                id='train-date-range',
                min_date_allowed=date(2010, 1, 1),
                max_date_allowed=date(2024, 12, 31),
                initial_visible_month=date(2023, 10, 1),
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                calendar_orientation='vertical',
            ), ], ),
            html.Br(),
            html.Div(['Test Date: ', dcc.DatePickerRange(
                id='test-date-range',
                min_date_allowed=date(2010, 1, 1),
                max_date_allowed=date(2024, 12, 31),
                initial_visible_month=date(2023, 10, 1),
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                calendar_orientation='vertical',
            ), ], ),
            html.Br(),
            'Select Index: ',
            dcc.Dropdown([
                {'label': 'S&P 500', 'value': '^GSPC'},
                {'label': 'NASDAQ', 'value': 'NDX'},
                {'label': 'Russell 2000', 'value': '^RUT'},
                {'label': 'All', 'value': 'All'}
            ],
                value='', id='index_selection',
                style={'marginLeft': '48px', 'width': '275px'}
            ),
            html.Br(),
            dbc.Button('Load Market Data in Database!', id='load-mktdata',
                       style={'color': 'white', 'background-color': '#0096FF'}),
            dbc.Spinner(html.Div(id='load_mktdata_status',
                     children=''), color='success'),
        ], align='start'),
        # html.Br(),
        dbc.Col([
            html.Div([
                'Confidence Level: ',
                dcc.Input(id='confidence_level', type='number', value=0.95, style={'marginLeft': '110px'}),
                html.Br(),
                html.Br(),
                'Number of Top Correlated Pairs: ',
                dcc.Input(id='num_top_corrs', type='number', value=30, style={'marginLeft': '8px'}),
                html.Br(),
                html.Br(),
                'Select Model: ',
                dcc.Dropdown(['OLS', 'Kalman Filter'], 'OLS', id='model_selection',
                             style={'marginLeft': '120px', 'width': '200px'}),
                html.Br(),
                # html.Br(),
                'Rolling Window size: ',
                dcc.Input(id='window_size', type='number', value=5, style={'marginLeft': '90px'}),
                html.Br(),
                'Upper Band Std Dev: ',
                html.Div([dcc.Slider(0, 3, 0.5, value=1, id='upper_std'), ],
                         style={'marginLeft': '150px', 'width': '300px'}),
                html.Br(),
                'Lower Band Std Dev: ',
                html.Div([dcc.Slider(0, 3, 0.5, value=1, id='lower_std'), ],
                         style={'marginLeft': '150px', 'width': '300px'}),
                html.Br(),
                html.Br(),
                dbc.Button('Get Top Correlated and Co-Integrated Pairs!', id='run_strategy',
                           style={'color': 'white', 'background-color': '#0096FF'}),
            ]), ], align='end'), ], align='center'),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Store(id='memory_storage', storage_type='memory'),
            html.Div([
                html.Div(id='corr_pairs_display',
                         children=''),
                dash_table.DataTable(id='top_corrs',
                                     fixed_rows={'headers': True},
                                     style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                     style_table={'height': '400px', 'overflowY': 'auto'},
                                     ),
            ], style={'width': '600px'}), ]),
        dbc.Col([
            html.Div([
                dbc.Spinner(html.Div(id='cointegrated_pairs_display',
                         children=''), color='success'),
                dash_table.DataTable(id='cointegrated_pairs',
                                     row_selectable='single',
                                     selected_rows=[],
                                     fixed_rows={'headers': True},
                                     style_cell={'padding-left': '20px', 'padding-right': '20px'},
                                     style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                                     style_table={'height': '400px', 'overflowY': 'auto'},
                                     ),
            ], style={'width': '600px'}), ]), ]),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([dcc.Graph(figure={}, id='cummulative_returns'), ]),
        dbc.Col([dcc.Graph(figure={}, id='bollinger_bands'), ]),
    ]),
], style=CONTENT_STYLE)


@callback(
    Output('load_mktdata_status', 'children'),
    Input('load-mktdata', 'n_clicks'),
    State('train-date-range', 'start_date'),
    State('train-date-range', 'end_date'),
    State('test-date-range', 'start_date'),
    State('test-date-range', 'end_date'),
    State('index_selection', 'value'),
    prevent_initial_call=True
)
def load_mktdata(n_clicks,
                 train_start_date: str,
                 train_end_date: str,
                 test_start_date: str,
                 test_end_date: str,
                 selected_index: str) -> str:
    '''
    Loads marketdata for Index and its components in the database
    :param n_clicks: Button Clicks
    :param train_start_date: Training Period Start date
    :param train_end_date: Training Period End date
    :param test_start_date: Test Period Start date
    :param test_end_date: Test Period End Ddate
    :param selected_index: Index Selected by the user
    :return: Status string
    '''
    # Load Index components
    data_loader = DataLoader(DB_CONNECTION_STR, DB_NAME)
    # Load All Index components in mktdata db
    load_index_components_in_db(DB_CONNECTION_STR, DB_NAME)

    # Load data for individual tickers
    index_list = INDEX_LIST if selected_index == 'All' else [selected_index]
    tickers = get_index_components_from_db(DB_CONNECTION_STR, DB_NAME, EQ_INDEX_COLLECTION, index_list)
    source_config = {'source': 'yfinance', 'tickers': tickers,
                     'start': datetime.strptime(train_start_date, '%Y-%m-%d'),
                     'end': datetime.strptime(train_end_date, '%Y-%m-%d')}
    db_data_config = {'collection_name': EQ_DAILY_COLLECTION, 'collection_index': [('Symbol', 1), ('Date', 1)]}
    data_loader.load_data_db(source_config, db_data_config)

    # Load test data
    source_config = {'source': 'yfinance', 'tickers': tickers,
                     'start': datetime.strptime(test_start_date, '%Y-%m-%d'),
                     'end': datetime.strptime(test_end_date, '%Y-%m-%d')}
    data_loader.load_data_db(source_config, db_data_config)
    return f'Market Data Loaded from {train_start_date} to {train_end_date} for training and from {test_start_date} to {test_end_date} for testing!'


@callback(
    Output('corr_pairs_display', 'children'),
    Output('top_corrs', 'data'),
    Output('top_corrs', 'columns'),
    Output('cointegrated_pairs_display', 'children'),
    Output('cointegrated_pairs', 'data'),
    Output('cointegrated_pairs', 'columns'),
    Output('memory_storage', 'data'),
    Input('run_strategy', 'n_clicks'),
    State('train-date-range', 'start_date'),
    State('train-date-range', 'end_date'),
    State('test-date-range', 'start_date'),
    State('test-date-range', 'end_date'),
    State('index_selection', 'value'),
    State('confidence_level', 'value'),
    State('num_top_corrs', 'value'),
    State('model_selection', 'value'),
    State('window_size', 'value'),
    State('upper_std', 'value'),
    State('lower_std', 'value'),
    prevent_initial_call=True
)
def run_strategy(n_clicks,
                 train_start_date: str,
                 train_end_date: str,
                 test_start_date: str,
                 test_end_date: str,
                 selected_index: str,
                 confidence_level: float,
                 num_top_corrs: int,
                 model_selection: str,
                 window_size: int,
                 upper_std: float,
                 lower_std: float,
                 ):
    '''
    Runs Pairs Trading Strategy on the selected tickers and backtests using test period.
    :param n_clicks:
    :param train_start_date: Training Period Start date
    :param train_end_date: Training Period End date
    :param test_start_date: Test Period Start date
    :param test_end_date: Test Period End date
    :param selected_index: Index selected
    :param confidence_level: Confidence level
    :param num_top_corrs: Number of top correlated pairs to display
    :param model_selection: Model to use in Pairs trading Strategy run
    :param window_size: Rolling Window size
    :param upper_std: Upper Bollinger band standard deviation
    :param lower_std: Lower Bollinger band standard deviation
    :return: Top correlated pairs, co-integrated pairs and their backtesting results
    '''
    train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d')
    train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d')
    test_start_date = datetime.strptime(test_start_date, '%Y-%m-%d')
    test_end_date = datetime.strptime(test_end_date, '%Y-%m-%d')
    index_list = INDEX_LIST if selected_index == 'All' else [selected_index]
    tickers = get_index_components_from_db(DB_CONNECTION_STR, DB_NAME, EQ_INDEX_COLLECTION, index_list)
    strategy_obj = PairsTrading(train_start_date,
                                train_end_date,
                                tickers=list(tickers),
                                confidence_level=confidence_level,
                                num_top_corrs=num_top_corrs,
                                calc_type=model_selection)
    strategy_results = strategy_obj.run()
    top_corrs = strategy_obj.selected_tickers_corrs.round({'Correlation': 3})
    top_corrs_cols = [{'name': col, 'id': col} for col in top_corrs.columns]
    top_corrs_data = top_corrs.to_dict('records')
    top_corrs_display = 'Top Correlated Pairs: '
    backtest_obj = BackTesting(test_start_date, test_end_date, strategy_results, upper_std, lower_std, window_size)
    backtest_results = backtest_obj.run()
    backtest_result_summary = pd.concat([v[0] for k, v in backtest_results.items()]).round(3)
    backtest_result_summary_cols = [{'name': col, 'id': col} for col in backtest_result_summary.columns]
    backtest_result_summary_data = backtest_result_summary.to_dict('records')
    cointegrated_pairs_display = 'Co-Integrated Pairs! Please select a pair to see performance results: '
    # Data for sharing between callbacks
    memory_data = {json.dumps(k): v[1].reset_index().to_json() for k, v in backtest_results.items()}
    return top_corrs_display, top_corrs_data, top_corrs_cols, cointegrated_pairs_display, backtest_result_summary_data, backtest_result_summary_cols, memory_data


@callback(
    Output('cummulative_returns', 'figure'),
    Output('bollinger_bands', 'figure'),
    Input('cointegrated_pairs', 'selected_rows'),
    State('memory_storage', 'data'),
    prevent_initial_call=True
)
def display_results(selected_row: list, results: dict):
    '''
    Takes the row selected in co-integrated pairs and show their timeseries performance results.
    :param selected_row: Row selected in the co-integrated pairs table
    :param results: backtesting results
    :return: graphs
    '''
    results = {tuple(json.loads(k)): pd.read_json(v) for k, v in results.items()}
    selected_pairs = list(results.keys())[selected_row[0]]
    selected_pairs_data = results[tuple(selected_pairs)]
    fig1 = px.line(selected_pairs_data, x=selected_pairs_data.Date, y=selected_pairs_data.Cum_Returns,
                   title=str(selected_pairs))
    fig1.update_layout(title_text=str(selected_pairs), title_x=0.5, title_font_color='red')
    fig2 = px.line(selected_pairs_data, x=selected_pairs_data.Date,
                   y=['Spread', 'BB_Upper', 'BB_Lower', 'Rolling_Correlation'], title=str(selected_pairs))
    fig2.update_layout(title_text=str(selected_pairs), title_x=0.5, title_font_color='red')
    return fig1, fig2


if __name__ == '__main__':
    app.run(debug=True)
