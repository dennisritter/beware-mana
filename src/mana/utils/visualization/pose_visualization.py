"""Visualises a skeleton from a Sequence in a webbrowser 3-D canvas."""
import numpy as np
import plotly.graph_objects as go
from kaleido.scopes.plotly import PlotlyScope


def vis_pose(positions: 'np.ndarray',
             name='figure',
             dump_format='png',
             dump_path='.',
             scene=None,
             img_size=(1000, 1000),
             html=False):

    scope = PlotlyScope()
    if not scene:
        scene = dict(
            xaxis=dict(range=[-50, 50]),
            yaxis=dict(range=[-50, 50]),
            zaxis=dict(range=[-50, 50]),
            camera=dict(up=dict(x=0, y=0, z=1.25),
                        eye=dict(x=-1.2, y=-1.2, z=1.2)),
        )

    layout = go.Layout(
        scene=scene,
        scene_aspectmode="cube",
        showlegend=False,
    )

    traces = _make_joint_traces(positions)

    fig = go.Figure(data=traces, layout=layout)
    filename = f'{dump_path}/{name}.{dump_format}'
    with open(filename, "wb") as f:
        f.write(
            scope.transform(fig,
                            format=dump_format,
                            width=img_size[0],
                            height=img_size[1]))

    fig.write_html(f'{name}.html', auto_open=False, auto_play=False)
    # print(f"Plot URL: {py.plot(fig, filename='skeleton', auto_open=False)}")
    # fig.show()


def _make_joint_traces(positions: 'np.ndarray'):
    trace_joints = go.Scatter3d(x=positions[:, 0],
                                y=positions[:, 1],
                                z=positions[:, 2],
                                text=np.arange(len(positions)),
                                textposition='top center',
                                mode="markers+text",
                                marker=dict(color="royalblue", size=5))
    # root_joints = go.Scatter3d(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]), mode="markers", marker=dict(color="red", size=5))
    # hipl = go.Scatter3d(x=np.array(pos[frame, 1, 0]),
    #                     y=np.array(pos[frame, 1, 1]),
    #                     z=np.array(pos[frame, 1, 2]),
    #                     mode="markers",
    #                     marker=dict(color="red", size=10))
    # hipr = go.Scatter3d(x=np.array(pos[frame, 6, 0]),
    #                     y=np.array(pos[frame, 6, 1]),
    #                     z=np.array(pos[frame, 6, 2]),
    #                     mode="markers",
    #                     marker=dict(color="green", size=10))
    # lowerback = go.Scatter3d(x=np.array(pos[frame, 11, 0]),
    #                      y=np.array(pos[frame, 11, 1]),
    #                      z=np.array(pos[frame, 11, 2]),
    #                      mode="markers",
    #                      marker=dict(color="black", size=10))
    return [trace_joints]  # + [hipl, hipr, lowerback]
