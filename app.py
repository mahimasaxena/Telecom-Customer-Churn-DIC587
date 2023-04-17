from dash import Dash, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

'''
age                       input box for integer  
married                   drop down yes / no 
dependents                drop down yes / no
number_of_referrals       input box for integer
tenure_in_months          input box for integer
internet_service          drop down yes / no
online_security           drop down yes / no
premium_tech_support      drop down yes / no
monthly_charge            input box for float
satisfaction_score        scale of 1 to 5
cltv                      input box for float
payament_method           drop down of credit_card (1) and mailed_check (0)
credit_card               drop down yes / no
mailed_check              drop down yes / no
'''

path = './Models/rf.sav'
with open(path, "rb") as f:
    model = pickle.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Label("Telecom Customer Churn Predictor"),
                className="text-center mt-4 mb-4")
    ], style={'background-color': '#023047', 'color': 'white', "fontSize": "50px"}),

    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Age:"),
                dbc.Input(placeholder='Enter Age', type='number',
                            value=25, style={'background-color': '#BEBEBE'}, id='input_age')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Married:"),
                dbc.Select(options=[{"label": 'Yes', "value": '1'}, {
                    "label": 'No', "value": '0'},], value='1', style={'background-color': '#BEBEBE'}, id='input_married')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Dependents:"),
                dbc.Select(options=[{"label": 'Yes', "value": '1'}, {
                    "label": 'No', "value": '0'},], value='1', style={'background-color': '#BEBEBE'}, id='input_dependents')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Number of Referrals:"),
                dbc.Input(placeholder='Enter # of referrals',
                            type='number', value=1, style={'background-color': '#BEBEBE'}, id='input_referrals')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Tenure in Months:"),
                dbc.Input(placeholder='Enter tenure in months',
                            type='number', value=8, style={'background-color': '#BEBEBE'}, id='input_tenure')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Internet Services:"),
                dbc.Select(options=[{"label": 'Yes', "value": '1'}, {
                    "label": 'No', "value": '0'},], value='1', style={'background-color': '#BEBEBE'}, id='input_internet_services')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Online Security:"),
                dbc.Select(options=[{"label": 'Yes', "value": '1'}, {
                    "label": 'No', "value": '0'},], value='1', style={'background-color': '#BEBEBE'}, id='input_online_security')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Premium Tech Support:"),
                dbc.Select(options=[{"label": 'Yes', "value": '1'}, {
                    "label": 'No', "value": '0'},], value='1', style={'background-color': '#BEBEBE'}, id='input_tech_support')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Monthly Charges:"),
                dbc.Input(placeholder='Enter monthly charges',
                            type='text', value='20', style={'background-color': '#BEBEBE'}, id='input_charges')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Satisfaction Score:"),
                dbc.Select(
                    id="input_satisfaction",
                    options=[{"label": i, "value": i}
                                for i in range(1, 6)],
                    value=3, style={'background-color': '#BEBEBE'}
                ),
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Customer Lifetime Value:"),
                dbc.Input(placeholder="Enter Customer Lifetime Value",
                          type="text", value='2500', style={'background-color': '#BEBEBE'}, id='input_cltv')
            ], className="mb-3"),

            dbc.InputGroup([
                dbc.InputGroupText("Payment Method:"),
                dbc.Select(
                    id="input_payment_method",
                    options=[
                        {"label": "Credit Card", "value": "Credit Card"},
                        {"label": "Mailed Check", "value": "Mailed Check"}
                    ],
                    value="Credit Card", style={'background-color': '#BEBEBE'}
                ),
            ], className="mb-3"),
        ], md=6, style={'background-color': '#234e70', 'padding': '40px', 'height': '743px'}),

        dbc.Col([
            dbc.Button("Predict", id="predict_button",
                       className="mt-4 w-100", color="warning"),
            html.Div(id="output_div", className="text-center mt-4"),
        ], md=6, style={'background-color': '#28666e', 'padding': '18px', 'height': '743px'}),
    ], align="center"),], fluid=True)


@callback(
    Output('output_div', 'children'),
    [Input('predict_button', "n_clicks")],
    [State('input_age', 'value'),
     State('input_married', 'value'),
     State('input_dependents', 'value'),
     State('input_referrals', 'value'),
     State('input_tenure', 'value'),
     State('input_internet_services', 'value'),
     State('input_online_security', 'value'),
     State('input_tech_support', 'value'),
     State('input_charges', 'value'),
     State('input_satisfaction', 'value'),
     State('input_cltv', 'value'),
     State('input_payment_method', 'value')]
)
def predict(n_clicks, input_age, input_married, input_dependents, input_referrals, input_tenure, input_internet_services,
            input_online_security, input_tech_support, input_charges, input_satisfaction, input_cltv, input_payment_method):

    if input_payment_method == 'Credit Card':
        credit_card = 1
        mailed_check = 0
    else:
        credit_card = 0
        mailed_check = 1

    data = {
        'age': int(input_age),
        'married': int(input_married),
        'dependents': int(input_dependents),
        'number_of_referrals': int(input_referrals),
        'tenure_in_months': int(input_tenure),
        'internet_service': int(input_internet_services),
        'online_security': int(input_online_security),
        'premium_tech_support': int(input_tech_support),
        'monthly_charge': float(input_charges),
        'satisfaction_score': int(input_satisfaction),
        'cltv': int(input_cltv),
        'credit_card': int(credit_card),
        'mailed_check': int(mailed_check)
    }

    data = pd.DataFrame(data=data, index=[0])

    if n_clicks:
        churn_label = model.predict(data)
        if churn_label:
            return dbc.Alert('Customer is likely to Churn', color='danger', style={"fontSize": "54px"})
        return dbc.Alert('Customer is likely to Stay', color='success', style={"fontSize": "54px"})


if __name__ == '__main__':
    app.run_server(debug=True)
