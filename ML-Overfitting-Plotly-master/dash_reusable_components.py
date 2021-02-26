import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def named_slider(name, **kwargs):
    return html.Div(
        style={'padding': '10px 10px 15px 4px'},
        children=[
            html.P(f'{name}:'),
            html.Div(dcc.Slider(**kwargs), style={'margin-left': '6px'})
        ]
    )


def named_dropdown(name, **kwargs):
    return html.Div([
        html.P(f'{name}:', style={'margin-left': '3px'}),
        dcc.Dropdown(**kwargs)
    ])


def input_area(name, **kwargs):
    return html.Div([
        html.P(f'{name}:', style={'margin-left': '3px'}),
        dcc.Input(**kwargs)
    ])


# Custom Display Components
def card(children, **kwargs):
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def custom_tooltip(text, target, placement='top'):
    return dbc.Tooltip(text, target=target, placement=placement, style={'font-size': 14})
