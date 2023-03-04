"""
Some extra components for the layout of the app.
"""
from typing import Dict

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc


def _omit(omitted_keys, dictionary: Dict):
    """
    Return a copy of dictionary with the keys in omitted_keys dropped
    :param omitted_keys: Iterable of keys
    :param dictionary: dictionary to prune
    :return: subset of given dictionary
    """
    return {k: v for k, v in dictionary.items() if k not in omitted_keys}


def named_slider(name, **kwargs):
    """
    A normal slider but with a name above it
    :param name: The name of the slider
    :param kwargs: To be passed to the Slider object
    :return: HTML div containing the name and slider
    """
    return html.Div(
        style={'padding': '10px 10px 15px 4px'},
        children=[
            html.P(f'{name}:'),
            html.Div(dcc.Slider(**kwargs), style={'margin-left': '6px'})
        ]
    )


def named_dropdown(name, **kwargs):
    """
    A normal dropdown but with a name above it
    :param name: The name of the dropdown
    :param kwargs: to be passed to the dropdown object
    :return: HTML div containing the name and dropdown
    """
    return html.Div([
        html.P(f'{name}:', style={'margin-left': '3px'}),
        dcc.Dropdown(**kwargs)
    ])


def named_input_area(name, **kwargs):
    """
    A normal input, but with a name above it
    :param name: The name of the input area
    :param kwargs: to be passed to the input area
    :return: HTML div containing the name and input area
    """
    return html.Div([
        html.P(f'{name}:', style={'margin-left': '3px'}),
        dcc.Input(**kwargs)
    ])


# Custom Display Components
def card(children, **kwargs):
    """
    Bootstrap card
    :param children: The children to put in the cards
    :param kwargs: to be passed to the Section
    :return: HTML Section containing the card
    """
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def custom_tooltip(text, target, placement='top'):
    """
    Wrapper to make a bootstrap tooltip
    :param text: What to put in the tooltip
    :param target: Object on which to place the tooltip
    :param placement: Where to "pop" the tooltip
    :return: Tooltip
    """
    return dbc.Tooltip(text, target=target, placement=placement, style={'font-size': 14})
