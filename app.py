import dash
from dash import dcc
from dash import html
import pandas as pd
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dash_table
import numpy as np
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors

bs='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
app=dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],suppress_callback_exceptions=True,
              meta_tags=[{'name': 'viewport',
                          'content': 'width=device-width, initial-scale=1.0'}])

server=app.server
colors = {
    'background': '##cccccc'
}


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),


])



#******************Home Page*********************************
index_page = html.Div([
dbc.NavbarSimple(
    children=[

        dbc.NavItem(dbc.NavLink("PAGE CAR", href="/page-2")),
    ],
    brand="Home",
    brand_href="/",
    color="primary",
    dark=True,
),
    dbc.Container([

            dbc.Row([
                dbc.Col(html.H4("Dashboard and data."), width={'size': 6, 'offset': 1, 'order': 1}),
                dbc.Col(html.H4('''Pour présenter au mieux l'intérêt que je cultive pour la data science, et le pourquoi de ce choix professionnel,
                                 j'ai choisi de mettre en application des cas d'usage et de déployer des dashboard sur différents sujets afin de démontrer l'utilité de ces moyens et technologies.
                                 Visualiser, mettre en évidence des tendances, améliorer la prise de décision, et l'analyse dans le domaine de la business intelligence.''',className='text-left text-primary, mb-6'),
                width={'size': 6, 'offset': 1, 'order': 1}),
            ]),

    ], fluid=True)


])

#**************Home page end********************************




#**************Page Automobile*************************************
#data car and processing
df_car=pd.read_csv('./data/automobile_clean.csv')
#**********for model*****************************
X = df_car.price.values[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    X, df_car.horsepower, random_state=42)

models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}

#*********for car table*****************************
df_test_car=df_car[['drive-wheels', 'body-style', 'price']]
df_grp=df_test_car.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
df_grp=df_grp.round(2)

df_test02=df_car[['horsepower-binned','aspiration','price']]
df_grp02=df_test02.groupby(['horsepower-binned','aspiration' ], as_index=False).mean()
df_grp02=df_grp02.round(2)

df_car_table=df_test_car.columns

fuel_type=df_car['fuel-type'].unique()

page_2_layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("PAGE CAR", href="/page-2")),
        ],
        brand="Home",
        brand_href="/",
        color="primary",
        dark=True,
    ),

    # Place for title
    dbc.Row([
        dbc.Col(html.H4("Web Dashboard Présentation statistique pour l'automobile", className='text-left text-primary, mb-4'),
                width={'size': 4, 'offset': 4, 'order': 1})

    ]),
    # Deuxième ligne
    dbc.Row([
        # Première colonne
        dbc.Col([
            html.Div([

                dash_table.DataTable(
                    id='df_grp',
                    columns=[{"name": i, "id": i}
                             for i in df_grp.columns],
                    data=df_grp.to_dict('records'),
                    style_cell=dict(textAlign='left'),
                    style_header=dict(backgroundColor="cadetblue"),
                    style_data=dict(backgroundColor="lavender")
                )

            ]),

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),

        # Deuxième colonne

        dbc.Col([
            html.Div([

                dash_table.DataTable(
                    id='df_grp02',
                    columns=[{"name": i, "id": i}
                             for i in df_grp02.columns],
                    data=df_grp02.to_dict('records'),
                    style_cell=dict(textAlign='left'),
                    style_header=dict(backgroundColor="cadetblue"),
                    style_data=dict(backgroundColor="lavender")
                )

            ]),

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),
        ]),

        dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id="dropdown_car",
                    options=[{"label": x, "value": x} for x in fuel_type],
                    value=fuel_type[0],
                    clearable=False,
                ),
                dcc.Graph(
                    id='bar-chart-car',

                )
            ])

        ], width={'size': 7, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5),

        #colonne with model
        dbc.Col([
            html.Div([
                html.H3("Horsepower and price."),
                html.P("Select Model:"),
                dcc.Dropdown(
                    id='model-name',
                    options=[{'label': x, 'value': x}
                             for x in models],
                    value='Regression',
                    clearable=False
                ),
                dcc.Graph(id="graph"),
            ])

        ], width={'size': 7, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5),

    ]),


], fluid=True)




# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


#******Callback page car*****************

@app.callback(
    Output("bar-chart-car", "figure"),
    [Input("dropdown_car", "value")])
def update_bar_chart_car(fuel_type):
    mask = df_car["fuel-type"] == fuel_type
    fig = px.histogram(df_car[mask], x="make", color="fuel-type")
    return fig

#***************Callback for model**********************
@app.callback(
    Output("graph", "figure"),
    [Input('model-name', "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')
    ])

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)



