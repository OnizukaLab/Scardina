from functools import wraps
import networkx as nx


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from line_profiler import LineProfiler

        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()

    return wrapper


def draw_schema_graph(mg: nx.MultiDiGraph, layout):
    # draw nodes
    nx.draw_networkx_nodes(
        mg,
        pos=layout,
        node_color="None",  # node surface
        edgecolors="k",  # node outline
        linewidths=2,  # node outline
        # bbox=dict(facecolor="#ffffff90", edgecolor="#f0f0f0"),
    )
    nx.draw_networkx_labels(
        mg,
        pos=layout,
        # node_color="None",  # node surface
        # edgecolors="k",  # node outline
        # linewidths=2,  # node outline
        bbox=dict(facecolor="#ffffff90", edgecolor="#f0f0f0"),
    )

    # draw parallel edges
    rad_interval = 0.05
    edges = {}  # {(u, v): [(k, label, rad)]}
    for u, v, d in mg.edges(data=True):
        if (u, v) not in edges:
            edges[(u, v)] = [(0, f"{d[u]}={d[v]}", 0)]
        else:
            curr_edges = edges[(u, v)]
            curr_edges.append((len(curr_edges), f"{d[u]}={d[v]}", -1))
            n_edges = len(curr_edges)
            for i in range(n_edges):
                # keep k and label, update rad
                curr_edges[i] = (curr_edges[i][0], curr_edges[i][1], i * rad_interval)

    edges_each_rad = {}  # {rad: ([(u, v)], {(u, v, k): label})}
    for e, ds in edges.items():
        for d in ds:
            rad = d[2]
            es, labels = edges_each_rad.get(rad, ([], {}))
            es.append(e)
            labels[(e[0], e[1], d[0])] = d[1]
            edges_each_rad[rad] = (es, labels)

    for rad, el in edges_each_rad.items():
        nx.draw_networkx_edges(
            mg,
            edgelist=el[0],
            pos=layout,
            connectionstyle=f"arc3, rad={rad}",  # parallel edges
        )
        for e, l in el[1].items():
            if rad == 0:
                nx.draw_networkx_edge_labels(
                    mg,
                    edge_labels={(e[0], e[1]): l},
                    pos=layout,
                    font_size=8,
                    bbox=dict(facecolor="#ffffff90", edgecolor="#f0f0f0"),
                )
            else:
                draw_networkx_multiedge_labels(
                    mg,
                    edge_labels={(e[0], e[1]): l},
                    pos=layout,
                    font_size=8,
                    bbox=dict(facecolor="#ffffff90", edgecolor="#f0f0f0"),
                )


# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
def draw_networkx_multiedge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
