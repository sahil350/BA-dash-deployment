from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


### Constants
N = 1000
beta = 1.0
D = 4.0 # infections lasts four days
gamma = 1.0 / D
delta = 1.0 / 3.0  # incubation period of three days

S0 = 999
E0 = 0
I0, R0 = N - S0, 0  # initial conditions: one infected, rest susceptible
y0= S0, E0, I0, R0

def deriv(y,t,N,beta,gamma,delta):
    S,E,I,R = y
    dsdt = -beta * I * S / N
    dedt = beta * I * S / N - delta * E
    didt = delta * E - gamma * I
    drdt = gamma * I
    return dsdt, dedt, didt, drdt

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_daq as daq

app = dash.Dash()


beta_slider = daq.Slider(id='beta', value=1.2, min=0, max=4, step=0.01,
                    marks={0: '0', 2: '2', 4: '4'})
gamma_slider = daq.Slider(id='gamma', value=0.2, min=0, max=1, step=0.01,
                    marks={0: '0', 1: '1'})
delta_slider = daq.Slider(id='delta', value=0.2, min=0, max=1, step=0.01,
                    marks={0: '0', 1: '1'})


app.layout = html.Div(
                      children=[dcc.Graph(id='sir-model'),
                       html.Div([
                       html.Label('Beta'),
                       beta_slider],
                       ),html.Br(),
                                html.Br(),
                        html.Div(
                       [html.Label('Gamma'),
                       gamma_slider]),
                                html.Br(),
                                html.Br(),
                        html.Div(
                       [html.Label('Delta'),
                       delta_slider])]                
                     )


@app.callback(Output('sir-model', 'figure'),
              [Input('beta', 'value'), Input('gamma', 'value'), Input('delta', 'value')])
def make_figure(beta, gamma, delta):

    t = np.linspace(0,100,100)
    ret = odeint(deriv, y0, t, args = (N, beta, gamma, delta))
    S, E, I, R = ret.T
    R_t = beta * S / (S + I + E + R)


    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=t, y=S,\
                             mode='lines', name='Susceptible',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=t, y=E,\
                             mode='lines', name='Exposed',
                            line=dict(color='pink', width=2)))
    fig.add_trace(go.Scatter(x=t, y=R,\
                             mode='lines', name='Recovered',
                            line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=t, y=I,\
                             mode='lines', name='Infected',
                            line=dict(color='darkred', width=2))) 
    

    fig.update_layout(title='SIR-Model',
                       xaxis_title='Time(days)',
                       yaxis_title='Cases',
                     title_x=0.5,
                      width=800,
                      height=600)
    return fig

if __name__ == "__main__":
	app.run_server()
