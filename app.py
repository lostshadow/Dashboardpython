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

############Place python##############################
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

#**************************************************
make_name=df_car.make.unique()
make_value=df_car.make.value_counts()
val_car=[]
for i in make_value:
    val_car.append(i)
df_make=pd.DataFrame(data=zip(make_name, make_value), columns=['make', 'number_car'])
fig_pie = px.pie(df_make, values='number_car', names='make', title='''Marques automobile de l'étude''')
################################

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
    color="#20bed8",
    dark=True,
),
    dbc.Container([
            html.Br(),
            dbc.Row([
                html.Br(),
                dbc.Col(html.H4("Le dashboard pourquoi?"),width={'size': 6, 'offset': 1, 'order': 1}),
                dbc.Col(html.P('''Les données organisées, structurées sont utiles pour la stratégie et la prise de décision. Elles représentent de façon numérique ou sous la forme de catégories
                l'information liée à la donnée, les data de votre étude . Trouver le moyen de visualiser ces données est donc important et nécessite une certaine réflexion.
                Aujourd'hui il existe de nombreuses options pour construire un tableau de bord pertinent et lisible.
                Cela commence avec une page Jupyter notebook, mais cela pourrait être un tableau construit sur Streamlit, Power BI, Data Studio de google en passant par QlickView ou Tableau.
            j'ai choisi une façon simple de montrer un tableau de bord, utilisant des données correspondant à l'industrie automobile, ceci à l'aide d' outils simple comme python et Dash le tout déployer sur Heroku.
            Ceci est un simple exemple pour montrer que tout est possible afin de mettre en évidence des tendances dans le domaine de la business intelligence.''',className='text-left text-primary, mb-6'),
                width={'size': 6, 'offset': 1, 'order': 1}),
            ]),
        dbc.Row([
           dbc.Col(

               html.Div([
                   html.H4('Représentation 3D: proportion des automobiles en fonction de leur style'),
                   html.H6('''Body style: convertible, hatchback: voiture à hayon arrière, sedan: berline, wagon: break, hardtop: convertible cabriolet'''),
                   dcc.Graph(id="graph3D"),
                   html.P("Car Width:"),
                   dcc.RangeSlider(
                       id='range-slider',
                       min=0, max=1, step=0.1,
                       marks={0.2: '0.2', 1: '1.0'},
                       value=[0.5, 2]
                   ),
               ]),

            width={'size': 8, 'offset': 1, 'order': 1}),
        ]),

    ], fluid=True)


])

#**************Home page end********************************




#**************Page Automobile*************************************
#
##############################################

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
        dbc.Col(html.H4("Dashboard Présentation statistique pour l'automobile", className='text-left text-primary, mb-4'),
                width={'size': 7, 'offset': 4, 'order': 1})

    ]),

    # Premiere ligne descriptive
    dbc.Row([
        dbc.Col([
            html.Div([
html.H4("Présentation des données et explication", className='text-left text-primary, mb-4'),

                    dcc.Markdown('''
#### Source Kaggle '''),
html.P('''L/100km signifie litre au 100, rpm est une abbreviation qui désigne la mesure de deux choses: le nombre de fois que le vilebrequin du moteur effectue une rotation complète par minute, et simultanément, le nombre de fois que chaque piston monte et descend dans son cylindre. /
    Bore diamètre de chaques cylindres. The stroke ratio, déterminé en divisant the bore par the stroke, indique traditionnellement si un moteur est conçu pour la puissance à des régimes élevés (tr/min)(rpm)
    Curb weight c'est Le poids à vide du véhicule, y compris un réservoir plein de carburant et tout l'équipement standard ''')
            ]),
 dcc.Markdown('''
#### Les Colonnes'''),
html.P(''' symboling, normalized-losses, make, fuel-type,
       aspiration, num-of-doors, body-style, drive-wheels,
       engine-location, wheel-base, length, width, height,
       curb-weight, engine-type, num-of-cylinders,engine-size,
       fuel-system, bore, stroke, compression-ratio, horsepower,
       peak-rpm, city-L/100km, highway-L/100km, price, price-binned,
       horsepower-binned, fuel-type-diesel, fuel-type-gas,
       aspiration-std, aspiration-turbo''')

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5),

dbc.Col([
    html.Div([

        dcc.Graph(
            id='graphpie',
            figure=fig_pie
        )
    ])

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5),

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
                ),
                html.Br(),
                dcc.Dropdown(
                    id="dropdown_car",
                    options=[{"label": x, "value": x} for x in fuel_type],
                    value=fuel_type[0],
                    clearable=False,
                ),
                dcc.Graph(
                    id='bar-chart-car',

                )

            ]),

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),

        # Deuxième colonne

        dbc.Col([
            html.Div([
                html.Br(),
                dash_table.DataTable(
                    id='df_grp02',
                    columns=[{"name": i, "id": i}
                             for i in df_grp02.columns],
                    data=df_grp02.to_dict('records'),
                    style_cell=dict(textAlign='left'),
                    style_header=dict(backgroundColor="cadetblue"),
                    style_data=dict(backgroundColor="lavender")
                ),
                html.Br(),
                ###########Pie chart place ################
                html.H4('Distribution valeurs category'),
                dcc.Graph(id="piegraph"),
                html.P("Names:"),
                dcc.Dropdown(id='names',
                             options=['horsepower', 'highway-L/100km',	'city-L/100km'],
                             value='horsepower', clearable=False
                             ),
                html.P("Values:"),
                dcc.Dropdown(id='values',
                        options=['price', 'engine-size', 'bore', 'stroke', 'peak-rpm'],
                        value='price', clearable=False
                ),
            html.Br(),
            ]),

        ], width={'size': 6, 'offset': 1, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),
        ]),

        dbc.Row([
        dbc.Col([
            html.Div([
            html.H4('Scatter plot Height and width dataset car'),
                    dcc.Graph(id="scatter-plot"),
                    html.P("Height and width"),
                    dcc.RangeSlider(
                        id='range-slider',
                        min=0, max=1, step=0.1,
                        marks={0: '0', 1: '1'},
                        value=[0.5, 1]
            ),

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

    #troisieme ligne

    dbc.Col([
        html.Div([



        ]),

    ], width={'size': 6, 'offset': 1, 'order': 1},
        xs=12, sm=12, md=12, lg=5, xl=5
    ),



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

##########callback scatter
@app.callback(
    Output("scatter-plot", "figure"),
    Input("range-slider", "value"))
def update_bar_chart(slider_range):

    low, high = slider_range
    mask = (df_car['width'] > low) & (df_car['width'] < high)
    fig = px.scatter(
        df_car[mask], x="width", y="height",
        color="aspiration", size='length',
        hover_data=['width'])
    return fig

@app.callback(
    Output("piegraph", "figure"),
    Input("names", "value"),
    Input("values", "value"))
def generate_chart(names, values):
    fig = px.scatter(df_car, x=names, y=values, color="num-of-doors", size='curb-weight')
    return fig

#####Callback 3D#######
@app.callback(
    Output("graph3D", "figure"),
    Input("range-slider", "value"))
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df_car.width > low) & (df_car.width < high)

    fig = px.scatter_3d(df_car[mask],
        x='length', y='width', z='height',
        color="body-style", hover_data=['width'])
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)



