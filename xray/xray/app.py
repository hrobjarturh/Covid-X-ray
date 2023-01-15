import numpy as np
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash_slicer import VolumeSlicer
from dash import callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import subprocess
import random
import subprocess
import time
import pexpect
import sys
import asyncio
from threading import Thread


import logging

from dash.dash import no_update

from pgrace import concar
from datetime import datetime, timedelta
import logging
from dateutil.parser import parse
from datetime import datetime
import time
import ast

import Events
events = Events.EventsManager()


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(
    __name__,
    title="Covid X-Ray",
    update_title=None,
    external_stylesheets=external_stylesheets,
)
server = app.server


images = [
    'radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii',
    'radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii',
]

natural_persons = [
    'Evelyn Salinas',
    'Christopher Russo',
    'Demi Beard',
    'Ilyas Fischer',
    'Casper Hewitt',
    'Carter Osborne',
    'Antony Lambert',
    'Stella Frazier',
    'Rosie Baxter',
    'Cruz Burton',
]

# ------------- I/O and data massaging ---------------------------------------------------

def getNextImage(images):
    imgfile = 'radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii'
    img = image.load_img("assets/"+imgfile)

    mat = img.affine
    img = img.get_data()
    img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]

    spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])

    # Create smoothed image and histogram
    med_img = filters.median(img, selem=np.ones((1, 3, 3), dtype=np.bool))
    hi = exposure.histogram(med_img)

    # Create mesh
    verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=5)
    x, y, z = verts.T
    i, j, k = faces.T
    fig_mesh = go.Figure()
    fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i, color='#3297a5'))

    # Create slicers
    slicer1 = VolumeSlicer(
        app, img, axis=0, spacing=spacing, thumbnail=False, color="#3297a5"
    )
    slicer1.graph.figure.update_layout(
        dragmode="drawclosedpath",
        newshape_line_color="#3297a5",
        plot_bgcolor="rgb(0, 0, 0)",
    )
    slicer1.graph.config.update(
        modeBarButtonsToAdd=[
            "drawclosedpath",
            "eraseshape",
        ]
    )

    slicer2 = VolumeSlicer(
        app, img, axis=1, spacing=spacing, thumbnail=False, color="#e36f32"
    )
    slicer2.graph.figure.update_layout(
        dragmode="drawrect", newshape_line_color="#e36f32", plot_bgcolor="rgb(0, 0, 0)"
    )
    slicer2.graph.config.update(
        modeBarButtonsToAdd=[
            "drawrect",
            "eraseshape",
        ]
    )

    def path_to_coords(path):
        """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
        indices_str = [
            el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
        ]
        return np.array(indices_str, dtype=float)


    def largest_connected_component(mask):
        labels, _ = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())[1:]
        return labels == (np.argmax(sizes) + 1)



    def gen_histogram(hi1, hi0):
        fig = px.bar(
            x=hi1,
            y=hi0,
            # Histogram
            labels={"x": "intensity", "y": "count"},
            template="plotly_white",
            color_continuous_scale=['#3297a5'],
            color_discrete_sequence=['#3297a5']
        )
        return fig
        

    # ------------- Define App Layout ---------------------------------------------------
    axial_card = dbc.Card(
        [
            dbc.CardHeader("Axial view of the lung"),
            dbc.CardBody([slicer1.graph, slicer1.slider, *slicer1.stores]),
            dbc.CardFooter(
                [
                    html.H6(
                        [
                            "Step 1: Draw a rough outline that encompasses all ground glass occlusions across ",
                            html.Span(
                                "all axial slices",
                                id="tooltip-target-1",
                                className="tooltip-target",
                            ),
                            ".",
                        ]
                    ),
                    dbc.Tooltip(
                        "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                        target="tooltip-target-1",
                    ),
                ]
            ),
        ]
    )

    saggital_card = dbc.Card(
        [
            dbc.CardHeader("Sagittal view of the lung"),
            dbc.CardBody([slicer2.graph, slicer2.slider, *slicer2.stores]),
            dbc.CardFooter(
                [
                    html.H6(
                        [
                            "Step 2:\n\nDraw a rectangle to determine the ",
                            html.Span(
                                "min and max height ",
                                id="tooltip-target-2",
                                className="tooltip-target",
                            ),
                            "of the occlusion.",
                        ]
                    ),
                    dbc.Tooltip(
                        "Only the min and max height of the rectangle are used, the width is ignored",
                        target="tooltip-target-2",
                    ),
                ]
            ),
        ]
    )

    histogram_card = dbc.Card(
        [
            dbc.CardHeader("Histogram of intensity values"),
            dbc.CardBody(
                [
                    dcc.Graph(
                        id="graph-histogram",
                        figure=gen_histogram(hi[1], hi[0]),
                        config={
                            "modeBarButtonsToAdd": [
                                "drawline",
                                "drawclosedpath",
                                "drawrect",
                                "eraseshape",
                            ]
                        },
                    ),
                ]
            ),
            dbc.CardFooter(
                [
                    dbc.Toast(
                        [
                            html.P(
                                "Before you can select value ranges in this histogram, you need to define a region"
                                " of interest in the slicer views above (step 1 and 2)!",
                                className="mb-0",
                            )
                        ],
                        id="roi-warning",
                        header="Please select a volume of interest first",
                        icon="danger",
                        is_open=True,
                        dismissable=False,
                    ),
                    "Step 3: Select a range of values to segment the occlusion. Hover on slices to find the typical "
                    "values of the occlusion.",
                ]
            ),
        ]
    )

    mesh_card = dbc.Card(
        [
            dbc.CardHeader("3D mesh representation of the image data and annotation"),
            dbc.CardBody([dcc.Graph(id="graph-helper1")]), #, figure=fig_mesh
        ]
    )
    
    return axial_card, saggital_card, histogram_card, mesh_card 

def getMeshFig(imgfile):
    img = image.load_img("assets/"+imgfile)

    img = img.get_data()
    img = np.copy(np.moveaxis(img, -1, 0))[:, ::-1]


    # Create smoothed image and histogram
    med_img = filters.median(img, selem=np.ones((1, 3, 3), dtype=np.bool))

    # Create mesh
    verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=5)
    x, y, z = verts.T
    i, j, k = faces.T
    fig_mesh = go.Figure()
    fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i, color='#3297a5'))

    return fig_mesh

# Define Modal
with open("assets/modal.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)
button_gh = dbc.Button(
    "Learn more",
    id="howto-open",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)
button_howto = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-covid-xray",
    id="gh-link",
    style={"text-transform": "none"},
)
nav_bar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.A(
                                        html.Img(
                                            src=("assets/grace_logo.svg"),
                                            height="30px",
                                        ),
                                        href="https://plotly.com/dash/",
                                    ),
                                    style={"width": "min-content"},
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H3("Covid X-Ray app"),
                                            html.P(
                                                "Exploration and annotation of CT images"
                                            ),
                                        ],
                                        id="app_title",
                                    )
                                ),
                            ],
                            align="center",
                            style={"display": "inline-flex"},
                        )
                    ),
                    modal_overlay,
                ],
                align="center",
                style={"width": "100%"},
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)
banner = dbc.Row(
    id="banner",
    className="banner",
    children=[html.Img(src="assets/grace_logo.svg")],
)
intro_section = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Row(style={"padding": "1.5rem"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Covid X-Ray App", id="title"),
                                html.H3("Welcome to the Covid X-Ray App"),
                                html.Div(
                                    className="intro",
                                    children=[
                                        html.P(
                                            "Here you can verify remote X-ray images!"
                                        ),
                                    ],
                                ),
                            ],
                            width=4,
                            style={"margin-left": "3rem"},
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        # dbc.NavbarToggler(id="navbar-toggler"),
                                        dbc.Nav(
                                            [
                                                dbc.NavItem(button_gh),
                                            ],
                                            className="ml-auto",
                                            navbar=True,
                                        ),
                                        modal_overlay,
                                    ]
                                )
                            ],
                            align="center",
                        ),
                    ]
                ),
            ]
        ),
    ]
)

#axial_card, saggital_card, histogram_card, mesh_card = getNextImage('imgfile goes here')

imgfile = random.choice(images)
mesh_card_1 = dbc.Card(
        [
            dbc.CardHeader("3D mesh unknown"),
            dbc.CardBody([dcc.Graph(id="graph-helper1")]), #, figure=getMeshFig(imgfile)
        ]
    )

imgfile = random.choice(images)
mesh_card_2 = dbc.Card(
        [
            dbc.CardHeader("3D mesh predicted"),
            dbc.CardBody([dcc.Graph(id="graph-helper2" )]), #
        ]
    )

imgfile = random.choice(images)
mesh_card_3 = dbc.Card(
        [
            dbc.CardHeader("3D mesh unknown"),
            dbc.CardBody([dcc.Graph(id="graph-helper3")]),
        ]
    )

imgfile = random.choice(images)
mesh_card_4 = dbc.Card(
        [
            dbc.CardHeader("3D mesh predicted"),
            dbc.CardBody([dcc.Graph(id="graph-helper4" )]), #
        ]
    )


#list_files = subprocess.run(["ls", "-l"])
#print("The exit code was: %d" % list_files.returncode)

def getmain(username):
    #axial_card, saggital_card, histogram_card, mesh_card = getNextImage('imgfile goes here')

    main_card = dbc.Col([   
                    #dbc.Row([dbc.Col(axial_card), dbc.Col(saggital_card)]), 
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader([
                                        html.Div([
                                            username,
                                        ],id='hidden_user', style= {'display': 'none'}),
                                        html.Div([
                                            '',
                                        ],id='hidden_managerID', style= {'display': 'none'}),
                                        html.Div([
                                            '',
                                        ],id='hidden_meshcard1', style= {'display': 'none'}),
                                        html.Div([
                                            '',
                                        ],id='hidden_meshcard2', style= {'display': 'none'}),
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(label="Use System",
                                                        tab_id="predict_tab"),
                                                dbc.Tab(label="External verificaiton",
                                                        tab_id="external_tab"),
                                            ],
                                            id="card-tabs",
                                            active_tab="predict_tab",
                                        )
                                ]
                                    ),
                                    dbc.CardBody(html.P(id="card-content", className="card-text")),
                                ]
                            )
                        ])
                    ])
        ])
    return main_card

def getLogin():
    login_card = dbc.Card(
        [
            dbc.Col([
                dbc.Row(style={"padding": "1.0rem"}),
                dbc.Input(id="username", placeholder="Type username...", type="text"),
                dbc.Row(style={"padding": "1.0rem"}),
                dbc.Input(id="password", placeholder="Type password...", type="password"),
                dbc.Row(style={"padding": "1.0rem"}),
                dbc.Button(
                    "login", id="login_button", className="ms-auto", n_clicks=0
                )
            ])
        ]
    )
    
    return login_card

card1_body = dbc.Row([
    dbc.Row([dbc.Col(mesh_card_1), dbc.Col(mesh_card_2)]),
    dbc.Row(style={"padding": "1.0rem"}),
    dbc.Row([
        dbc.Col([
            dbc.Button("Next Image", outline=True, color="primary", className="me-1", id='next_button'),
            dbc.Row(style={"padding": "1.0rem"}),
            dbc.Button("Check for ID match", outline=False, color="primary", className="me-1", id='predict_button'),
        ]),
        dbc.Col([
                html.Div('Results are shown here',id = 'first_results'),
                dbc.Row(style={"padding": "1.0rem"}),
                dbc.Button(
                    "Approve", id="approve_button", color="primary", className="ms-auto", n_clicks=0, disabled=True
                )
        ])
    ])
])



def getnewconts(user):
    card1 = dbc.Card(
        [
            dbc.CardHeader(f"{user}",id='user_header'),
            dbc.CardBody(card1_body),
        ]
    )
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    #mr.genLogs()
    return dbc.Row([
        html.Div(f"{user}",id='user_header'),
        dbc.Col(card1_body)]),

def getExternal(user):
    
    df2 = pd.read_csv("sessionData.csv")
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    #mr.genLogs()
    card2_body = dbc.Row([
        html.Div(user,id='loggedin_user'),
        dbc.Row(style={"padding": "1.0rem"}),      
        dbc.Col([
            dbc.Row([dbc.Col(mesh_card_3), dbc.Col(mesh_card_4)]),     
            dbc.Row(style={"padding": "1.0rem"}),
            html.H5('Select result to verify'),
            dcc.Dropdown(df2.managerID.values, id='result_dropdown'),
            html.Div('results are shown here',id = 'external_results'),
            dbc.Row(style={"padding": "1.0rem"}),
            dbc.Button(
                "Approve", id="approve_ext_button", color="primary", className="ms-auto", n_clicks=0, disabled=True
            )
        ])
    ])
    
    
    
    return dbc.Row([dbc.Col(card2_body)]),



app.layout = html.Div(
    [
        banner,
        intro_section,
        dbc.Row([
            dbc.Col([
                html.P('monpoly'),
                html.Div(
                        [
                            dcc.Interval(interval=5 * 1000, id="interval"),
                            dcc.Textarea(
                                id='monpoly-textarea',
                                value= str(0), #subprocess.check_output(['./monpoly/monpoly', '-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-log','test.log','-negate']).decode("utf-8"),
                                style={'width': '100%', 'height': 200},
                                disabled = True,
                            ),   
                        ]
                    ),
            ]),
            dbc.Col([
                html.P('logs'),
                dcc.Textarea(
                    id='logs-textarea',
                    value=open('test.log', "r").read(),
                    style={'width': '100%', 'height': 200},
                    disabled = True,
                ),   
            ]),
        ]),
        #login_modal,
        # nav_bar,
        dbc.Col([
            getLogin(),
        ],id ='mainCol'),
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
    ],
)


'''@app.callback(
    [Output('monpoly-textarea','value'),Output('logs-textarea','value')],
    Input('refresh_logs_button','n_clicks')
)
def update_logs(n_clicks):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    if btn == 'refresh_logs_button':
        print('genLogs()')
        genLogs()
        return (subprocess.check_output(['./monpoly/monpoly', '-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-log','test.log','-negate']).decode("utf-8"),open('test.log', "r").read())
        
    raise PreventUpdate'''

#subprocess.check_output(['./monpoly/monpoly', '-sig','monpoly/art12_app_new/test.sig','-formula','monpoly/art12_app_new/test.mfotl','-log','test.log','-negate']).decode("utf-8")

@app.callback(
    [
        Output('monpoly-textarea','value'),
        Output('logs-textarea','value'), 
    ],
    Input('interval', 'n_intervals')
)
def updateMon(n_intervals):
    print('n_intervals: ',n_intervals)
    #print('str(mr.data): ',str(mr.data))
    mr.genLogs()
    return mr.monpoly, mr.logs #open('test.log', "r").read() #str(mr.data)

@app.callback(
    [Output("external_results", "children"), Output('approve_ext_button','disabled'),Output("graph-helper3", "figure"),Output("graph-helper4", "figure")],
    [State('loggedin_user','children'), State('result_dropdown','value')],
    [Input("approve_ext_button", "n_clicks"), Input("result_dropdown", "value")],
)
def showResults(user,result_dropdown,b_clicks,value):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    if btn == 'result_dropdown':
        df2 = pd.read_csv("sessionData.csv")
        inputData = df2.loc[df2['managerID']==result_dropdown]['inputData'].values[0]
        predData = df2.loc[df2['managerID']==result_dropdown]['predData'].values[0]
        
        return 'new results come here', False, getMeshFig(inputData), getMeshFig(predData)# getMeshFig(inputData), getMeshFig(predData)

    if btn == 'approve_ext_button':
        
        df2 = pd.read_csv("sessionData.csv")
        inputData = df2.loc[df2['managerID']==result_dropdown]['inputData'].values[0]
        predData = df2.loc[df2['managerID']==result_dropdown]['predData'].values[0]
        ver1 = df2.loc[df2['managerID']==result_dropdown].user.values[0]
        
        print('ver1: ',ver1)
        print('user: ',user)
        
        if ver1 == user:
            textoutput = 'Same user cannot do initial verification and external verification'
        else:
            textoutput = 'Succesfully verified'
        
            events.sendExternalVerification(result_dropdown,user)
            df2 = df2.loc[df2['managerID']!=result_dropdown]
            df2.to_csv('sessionData.csv', index=False)
            
            print('genLogs() inside ext')
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
        #mr.genLogs()
        
        return textoutput, False, no_update, no_update# getMeshFig(inputData), getMeshFig(predData)
        
    raise PreventUpdate



@app.callback(
    [Output("mainCol", "children")],
    [State('username','value')],
    [Input("login_button", "n_clicks")],
)
def login(username,n_clicks):
    btn = callback_context.triggered[0]['prop_id']
    events.setUser(username)
    if btn == 'login_button.n_clicks':
        return [getmain(username)]
    raise PreventUpdate

@app.callback(
    [Output("approve_button", "disabled"), Output("hidden_managerID", "children"),Output("graph-helper2", "figure"), Output('first_results','children'),Output("hidden_meshcard2", "children")],
    [State('user_header','children'), State("hidden_managerID", "children"),State("hidden_meshcard1", "children"),State("hidden_meshcard2", "children")],
    [Input("predict_button", "n_clicks"),Input("approve_button", "n_clicks")],
)
def predict_click(user_header,hidden_managerID,hidden_meshcard1,hidden_meshcard2,n_clicks1,n_clicks2):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if btn == 'predict_button':
        if n_clicks1 != None:
            
            print('predict_button: mr', mr.data)
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            result = random.choice(natural_persons)
            ## model is used here
            events.sendDBUsed('db1',{'input1': 'value','input2': 'value'},1.)
            ##
            imgfile = random.choice(images)
            print('predict img', imgfile)
            print('genLogs() inside predict')
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            #mr.genLogs()
            return (False, str('EM.managerID'),getMeshFig(imgfile), result, imgfile) #getMeshFig(imgfile)
    
    elif btn == 'approve_button':
        df2 = pd.read_csv("sessionData.csv")
        df2.loc[len(df2)] = {'managerID': events.managerID, 'user': user_header,'referenceDB': 'refdb', 'inputData':hidden_meshcard1,'predData':hidden_meshcard2, 'results': 'asdasd' }
        df2.to_csv('sessionData.csv', index=False)
        
        events.sendInterntalVerification(user_header)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        #mr.genLogs()
        
        return (True, 'hidden_managerID', None, 'Results are shown here',hidden_meshcard2)
        
    
    raise PreventUpdate


@app.callback(
    [Output("graph-helper1", "figure"),Output("hidden_meshcard1", "children")], [Input("next_button", "n_clicks")], prevent_initial_call=True
)
def next_button_click(next_button):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    print('btn1: ',btn)
    if btn == 'next_button':
        if next_button != None:
            events.startSession()
            events.sendModelUse('startDate')
            print('btn2: ',btn)
            imgfile = random.choice(images)
            print(imgfile)
            events.sendModelUse('endDate')
            return (getMeshFig(imgfile), imgfile)
        
    raise PreventUpdate
        

@app.callback(
    Output("card-content", "children"), State('hidden_user','children'), [Input("card-tabs", "active_tab")]
)
def tab_content(hidden_user,active_tab):
    print('hidden_user: ', hidden_user)
    if active_tab == 'predict_tab':
        return getnewconts(hidden_user[0])
    elif active_tab == 'external_tab':
        return getExternal(hidden_user[0])
    else:
        print('in else ..')
        raise PreventUpdate
    
    
# events

# ** Async Part **


    
# *** Flask Part ***:
    

class Logger():   
    def __init__(self) -> None:
        self.last_datetime = datetime.now() - timedelta(seconds=5)
        self.data = 1
        self.logs = ""
        self.monpoly = ""
        self.child = None#self.start_child()
    
    def genLogs(self, start=False):
        
        # clear log file
        with open('test.log', 'w'):
            pass
        
        
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        new_datetime = datetime.now() - timedelta(seconds=5)
        list_of_events = concar.get_events(
            start_date= self.last_datetime,
            end_date=new_datetime,
            name='art12dev35'
        )
        self.last_datetime = new_datetime

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='test.log', level=logging.INFO, format= '%(message)s')
        
        for e_index in reversed(range(len(list_of_events))):
            e = list_of_events[e_index]
            e_type = e['comment']
            e_datetime = int(time.mktime(parse(e['createdDate']).timetuple()))
            
            def sending(lg, e_datetime):
                self.logs += lg
                self.sendline(lg)
                self.sendline(' '.join(['@'+str(e_datetime+1)+'\n']))
                self.sendline(' '.join(['@'+str(e_datetime+1)+'\n']))
            
            self.sendline(' '.join(['@'+str(e_datetime)+'\n']))
            
            if e_type == 'initialVerification':
                sending(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')\n' ]), e_datetime)
                sending(' '.join(['@'+str(e_datetime+1)+'\n']), e_datetime)
                
            if e_type == 'externalVerification':
                sending(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')\n' ]), e_datetime)
                sending(' '.join(['@'+str(e_datetime+1)+'\n']), e_datetime)
                
            if e_type == 'modelUse':
                sending(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+')\n' ]), e_datetime)
                if str(e['data']['standardized']['customStr1']) == 'endDate':
                    sending(' '.join(['@'+str(e_datetime+1)+'\n']), e_datetime)
                
            if e_type == 'dbUsed':
                sending(' '.join(['@'+str(e_datetime) ,e_type, '('+str(e['govStream'])   +','+'"'+str(e['data']['standardized']['customStr1'])+'"'+','+str(e['data']['standardized']['customNum1'])+', '+'"'+str(ast.literal_eval(e['data']['standardized']['jsonDoc']))+'"'+')\n' ]), e_datetime)
                sending(' '.join(['@'+str(e_datetime+1)+'\n']), e_datetime)
                
        #print('LOG: ',self.logs)
                
                
                
    def sendline(self, monp):
        
        self.child.send(monp)
        self.child.expect('\n',async_=False)
        result = self.child.before.decode("utf-8")
        
        print('sendline result...: ',result)
        if result:
            
            if 'violation:' in result:
                print('in violation....')
                self.monpoly += result + '\n'
            
        
    async def start_child(self):
        print('start_child')
        launchcmd = './monpoly/monpoly -sig monpoly/art12_app_new/test.sig -formula monpoly/art12_app_new/test.mfotl -negate -nofilteremptytp -nofilterrel'
        child = pexpect.spawn(launchcmd)
        #child.logfile_read = sys.stdout.buffer
        self.child  = child
        
    async def some_print_task(self):
        pass
               
    async def async_main(self):
        """Main async function"""
        time.sleep(2)
        await asyncio.gather(self.some_print_task())
                
mr = Logger()

def async_main_wrapper(mr):
    asyncio.run(mr.start_child())
    while True:
        """Not async Wrapper around async_main to run it as target function of Thread"""
        asyncio.run(mr.async_main())
        time.sleep(2)
        
if __name__ == "__main__":
    print('@@@@ main----')
    th = Thread(target=async_main_wrapper, args=(mr,))
    th.start()
    app.run_server(debug=True, dev_tools_props_check=False,dev_tools_silence_routes_logging=False, use_reloader=False,port=8052,)
    th.join()
    

@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
