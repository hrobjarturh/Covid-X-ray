from time import time

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


t1 = time()

# ------------- I/O and data massaging ---------------------------------------------------

def getNextImage(imgfile):
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
            dbc.CardBody([dcc.Graph(id="graph-helper", figure=fig_mesh)]),
        ]
    )
    
    return axial_card, saggital_card, histogram_card, mesh_card #[dbc.Row([dbc.Col(axial_card), dbc.Col(saggital_card)]), dbc.Row([dbc.Col(histogram_card),dbc.Col(mesh_card),])]

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


t2 = time()

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
                                            "Here you can explore and annotate CT images! Just follow the instructions under each image or plot."
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


axial_card, saggital_card, histogram_card, mesh_card  = getNextImage('imgfile')
print('unpack')

def getmain(username):
    axial_card, saggital_card, histogram_card, mesh_card = getNextImage('imgfile goes here')

    main_card = dbc.Col([
                    dbc.Row([dbc.Col(axial_card), dbc.Col(saggital_card)]), 
                    dbc.Row([dbc.Col(histogram_card),dbc.Col(mesh_card),]),
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
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(label="Predict",
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
    dbc.Col([
        dbc.Button("Next Image", outline=True, color="primary", className="me-1", id='next_button'),
        dbc.Row(style={"padding": "1.0rem"}),
        dbc.Button("Predict", outline=False, color="primary", className="me-1", id='predict_button'),
    ]),
    dbc.Col([
            html.P('Results are shown here'),
            dbc.Row(style={"padding": "1.0rem"}),
            dbc.Button(
                "Approve", id="approve_button", color="primary", className="ms-auto", n_clicks=0, disabled=True
            )
    ])
])



def getnewconts(user):
    card1 = dbc.Card(
        [
            dbc.CardHeader(f"{user}",id='user_header'),
            dbc.CardBody(card1_body),
        ]
    )
    
    
    return dbc.Row([dbc.Col(card1)]),

def getExternal(user):
    
    df2 = pd.read_csv("sessionData.csv")
    
    card2_body = dbc.Row([
        html.Div(user,id='loggedin_user'),
        dbc.Col([
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
        #login_modal,
        # nav_bar,
        dbc.Col([
            getLogin(),
        ],id ='mainCol'),
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
    ],
)

t3 = time()

@app.callback(
    [Output("external_results", "children"), Output('approve_ext_button','disabled')],
    [Input("result_dropdown", "value")],
)
def showResults(value):
    if value:
        return 'new results come here', False
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
    [Output("approve_button", "disabled"), Output("hidden_managerID", "children"),],
    [State('user_header','children'), State("hidden_managerID", "children")],
    [Input("predict_button", "n_clicks"),Input("approve_button", "n_clicks")],
)
def predict_click(hidden_managerID,user_header,n_clicks1,n_clicks2):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if btn == 'predict_button':
        if n_clicks1 != None:
            
            events.startSession()
            events.sendModelUse('startDate')
            ## model is used here
            events.sendModelUse('endDate')
            
            return (False, str('EM.managerID'))
    
    elif btn == 'approve_button':
        df2 = pd.read_csv("sessionData.csv")
        df2.loc[len(df2)] = {'managerID': events.managerID, 'user': user_header,'referenceDB': 'refdb', 'inputData': 'img1', 'results': 'asdasd' }
        df2.to_csv('sessionData.csv', index=False)
        
        return (True, 'hidden_managerID')
        
    
    raise PreventUpdate

@app.callback(
    [Output("result_dropdown", "children")],
    [State('loggedin_user','children'),
    State('result_dropdown','value')],
    [Input("approve_ext_button", "n_clicks")],
)
def predict_click(user,result_dropdown,b_clicks):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if btn == 'approve_ext_button':
        
        df2 = pd.read_csv("sessionData.csv")
        ver1 = df2.loc[df2['managerID']==result_dropdown].user.values[0]
        events.sendVerification(result_dropdown,ver1,user)
        
        
    
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
        return getnewconts(hidden_user[0])




# ------------- Define App Interactivity ---------------------------------------------------
'''
@app.callback(
    [Output("graph-histogram", "figure"), Output("roi-warning", "is_open")],
    [Input("annotations", "data")],
)
def update_histo(annotations):
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        return dash.no_update, dash.no_update
    # Horizontal mask for the xy plane (z-axis)
    path = path_to_coords(annotations["z"]["path"])
    rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
    if len(rr) == 0 or len(cc) == 0:
        return dash.no_update, dash.no_update
    mask = np.zeros(img.shape[1:])
    mask[rr, cc] = 1  # TODO: #ERROR: Index 630 for axis 1 with size 630?
    mask = ndimage.binary_fill_holes(mask)
    # top and bottom, the top is a lower number than the bottom because y values
    # increase moving down the figure
    top, bottom = sorted([int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]])
    intensities = med_img[top:bottom, mask].ravel()
    if len(intensities) == 0:
        return dash.no_update, dash.no_update
    hi = exposure.histogram(intensities)
    fig = gen_histogram(hi[1], hi[0])
    
    fig.update_layout(dragmode="select", title_font=dict(size=20))
    return fig, False


@app.callback(
    [
        Output("occlusion-surface", "data"),
        Output(slicer1.overlay_data.id, "data"),
        Output(slicer2.overlay_data.id, "data"),
    ],
    [Input("graph-histogram", "selectedData"), Input("annotations", "data")],
)
def update_segmentation_slices(selected, annotations):
    ctx = dash.callback_context
    # When shape annotations are changed, reset segmentation visualization
    if (
        ctx.triggered[0]["prop_id"] == "annotations.data"
        or annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        mask = np.zeros_like(med_img)
        overlay1 = slicer1.create_overlay_data(mask)
        overlay2 = slicer2.create_overlay_data(mask)
        return go.Mesh3d(), overlay1, overlay2
    elif selected is not None and "range" in selected:
        if len(selected["points"]) == 0:
            return dash.no_update
        v_min, v_max = selected["range"]["x"]
        t_start = time()
        # Horizontal mask
        path = path_to_coords(annotations["z"]["path"])
        rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
        mask = np.zeros(img.shape[1:])
        mask[rr, cc] = 1
        mask = ndimage.binary_fill_holes(mask)
        # top and bottom, the top is a lower number than the bottom because y values
        # increase moving down the figure
        top, bottom = sorted(
            [int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]]
        )
        img_mask = np.logical_and(med_img > v_min, med_img <= v_max)
        img_mask[:top] = False
        img_mask[bottom:] = False
        img_mask[top:bottom, np.logical_not(mask)] = False
        img_mask = largest_connected_component(img_mask)
        # img_mask_color = mask_to_color(img_mask)
        t_end = time()
        print("build the mask", t_end - t_start)
        t_start = time()
        # Update 3d viz
        verts, faces, _, _ = measure.marching_cubes(
            filters.median(img_mask, selem=np.ones((1, 7, 7))), 0.5, step_size=3
        )
        t_end = time()
        print("marching cubes", t_end - t_start)
        x, y, z = verts.T
        i, j, k = faces.T
        trace = go.Mesh3d(x=z, y=y, z=x, color="red", opacity=0.8, i=k, j=j, k=i)
        overlay1 = slicer1.create_overlay_data(img_mask)
        overlay2 = slicer2.create_overlay_data(img_mask)
        # todo: do we need an output to trigger an update?
        return trace, overlay1, overlay2
    else:
        return (dash.no_update,) * 3


@app.callback(
    Output("annotations", "data"),
    [
        Input(slicer1.graph.id, "relayoutData"),
        Input(slicer2.graph.id, "relayoutData"),
    ],
    [State("annotations", "data")],
)
def update_annotations(relayout1, relayout2, annotations):
    if relayout1 is not None and "shapes" in relayout1:
        if len(relayout1["shapes"]) >= 1:
            shape = relayout1["shapes"][-1]
            annotations["z"] = shape
        else:
            annotations.pop("z", None)
    elif relayout1 is not None and "shapes[2].path" in relayout1:
        annotations["z"]["path"] = relayout1["shapes[2].path"]

    if relayout2 is not None and "shapes" in relayout2:
        if len(relayout2["shapes"]) >= 1:
            shape = relayout2["shapes"][-1]
            annotations["x"] = shape
        else:
            annotations.pop("x", None)
    elif relayout2 is not None and (
        "shapes[2].y0" in relayout2 or "shapes[2].y1" in relayout2
    ):
        annotations["x"]["y0"] = relayout2["shapes[2].y0"]
        annotations["x"]["y1"] = relayout2["shapes[2].y1"]
    return annotations

'''
app.clientside_callback(
    """
function(surf, fig){
        let fig_ = {...fig};
        fig_.data[1] = surf;
        return fig_;
    }
""",
    output=Output("graph-helper", "figure"),
    inputs=[
        Input("occlusion-surface", "data"),
    ],
    state=[
        State("graph-helper", "figure"),
    ],
)


@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False,port=8052,)
