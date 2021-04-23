#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly as plotly
from matplotlib import pyplot as plt

DEFAULT_CI_LEVEL = 95
__all__ = [
    "add_selctor",
    "add_stats_selector",
    "add_stats_display_sns",
    "line_with_band",
    "add_stats_display_plotly",
    "add_stats_display",
    "load_data",
    "plot_netowrk",
]


@st.cache
def load_data(folder: Path, name: str):
    data = pd.read_csv(folder / f"{name}.csv", index_col=None)
    return data


def add_selector(
    parent,
    title,
    content,
    key,
    all=True,
    none=True,
    some=True,
    one=True,
    default="one",
    check=False,
    check2=False,
    default_check=False,
):
    options = []
    indx = 0
    content = sorted(content)
    for a, v in zip((all, some, one, none), ("all", "some", "one", "none")):
        if a:
            options.append(v)
            if v == default:
                indx = len(options) - 1
    parent.text(title)
    col1, col2 = parent.beta_columns([1, 4])
    if check:
        combine = st.checkbox("Combine", value=default_check, key=f"{key}_sel_check")
    else:
        combine = False
    if check2:
        overlay = st.checkbox("Overlay", value=default_check, key=f"{key}_sel_overlay")
    else:
        overlay = False
    with col1:
        selection_type = st.radio(title, options, index=indx, key=f"{key}_sel_type")
    if selection_type == "none":
        return ([], combine, overlay) if check or overlay else []
    if selection_type == "all":
        return (content, combine, overlay) if check or overlay else content
    with col2:
        if selection_type == "some":
            selector = st.multiselect("", content, key=f"{key}_multi")
            return (selector, combine, overlay) if check or overlay else selector
        selector = st.selectbox("", content, key=f"{key}_sel")
        return ([selector], combine, overlay) if check or overlay else [selector]


def add_stats_selector(
    folder,
    file_name,
    filters,
    xvar,
    label,
    default_selector="one",
    choices=None,
    key="",
):
    if label is None:
        label = file_name.split("_")[0] + " Statistics"
        label = label[0].toupper() + label[1:]
    world_stats = load_data(folder, file_name)
    if filters:
        filtered = None
        for fset in filters:
            x = world_stats.copy()
            for field, values in fset:
                if len(x) < 1:
                    break
                if field.endswith("step") or field.endswith("steps") or field.endswith("relative_time"):
                    x = x.loc[(world_stats[field] >= values[0]) & (world_stats[field] <= values[1]), :]
                    continue
                x = x.loc[world_stats[field].isin(values), :]
            if len(x) > 0:
                x = x.index
                if filtered is None:
                    filtered = x
                else:
                    filtered = filtered.union(x)
        if filtered is not None:
            world_stats = world_stats.loc[filtered, :]
        else:
            world_stats = world_stats.loc[[False]*len(world_stats), :]
        # world_stats = pd.concat(filtered, ignore_index=False)

    if choices is None:
        choices = [_ for _ in world_stats.columns if _ not in ("step", "relative_time")]
    elif isinstance(choices, Callable):
        choices = choices(world_stats)

    world_stats_expander = st.sidebar.beta_expander(label)
    with world_stats_expander:
        selected_world_stats, combine_world_stats, overlay_world_stats = add_selector(
            st,
            "",
            choices,
            key=f"{file_name}_{key}",
            none=True,
            default=default_selector,
            check=True,
            check2=True,
            default_check=xvar == "step",
        )
    return world_stats, selected_world_stats, combine_world_stats, overlay_world_stats


def add_stats_display_sns(
    stats,
    selected,
    combine,
    overlay,
    hue,
    xvar,
    cols=None,
    start_col=0,
    ncols_effective=0,
    ci_level=DEFAULT_CI_LEVEL,
):
    displayed = 0
    if overlay:
        fig, ax = plt.subplots()
    for i, field in enumerate(selected):
        if not overlay:
            fig, ax = plt.subplots()
            plt.title(field)
        sns.lineplot(
            data=stats,
            x=xvar,
            y=field,
            label=field if overlay else None,
            ax=ax,
            hue=hue if not combine else None,
            style=None,
            ci=ci_level,
        )
        if not overlay:
            with cols[(i + start_col) % ncols_effective]:
                displayed += 1
                st.pyplot(fig)
    if overlay:
        with cols[(displayed + start_col) % ncols_effective]:
            displayed += 1
            st.pyplot(fig)
    return cols, displayed + start_col


def line_with_band(fig, stats, xvar, yvar, color, i, color_val=None, ci_level=DEFAULT_CI_LEVEL):
    if color is not None:
        for i, v in enumerate(stats[color].unique()):
            fig = line_with_band(
                fig, stats.loc[stats[color] == v, :], xvar, yvar, None, i, color_val=v, ci_level=ci_level
            )
            fig.update_layout(yaxis_title=yvar)
        return fig
    colors = px.colors.qualitative.Plotly
    if color:
        stats = stats.groupby([xvar, color]).agg(["mean", "std", "count"])
    else:
        stats = stats.groupby([xvar]).agg(["mean", "std", "count"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    for c in stats.columns:
        if not c.endswith("mean"):
            continue
        base = c[: -len("_mean")]
        if not f"{base}_std" in stats.columns or not f"{base}_count" in stats.columns:
            stats[f"{base}"] = stats[c]
            stats = stats.drop([c])
            continue
        stats[f"{base}_ci_hi"] = stats[c]
        stats[f"{base}_ci_lo"] = stats[c]
        indx = stats[f"{base}_count"] > 0
        stats.loc[indx, f"{base}_ci_hi"] = stats.loc[indx, c] + (1 + ci_level / 100.0) * stats.loc[
            indx, f"{base}_std"
        ] / stats.loc[indx, f"{base}_count"].apply(np.sqrt)
        stats.loc[indx, f"{base}_ci_lo"] = stats.loc[indx, c] - (1 + ci_level / 100.0) * stats.loc[
            indx, f"{base}_std"
        ] / stats.loc[indx, f"{base}_count"].apply(np.sqrt)
        stats[f"{base}"] = stats[c]
        stats = stats.drop([c, f"{base}_std", f"{base}_count"], axis=1)
    stats = stats.reset_index()
    x, y, hi, lo = stats[xvar], stats[f"{yvar}"], stats[f"{yvar}_ci_hi"], stats[f"{yvar}_ci_lo"]
    if fig is None:
        fig = go.Figure()

    yname = yvar if not color_val else f"{color_val}:{yvar}" if ":" not in color_val else color_val
    fig.add_trace(go.Scatter(x=x, y=y, name=yname, line_color=colors[i % len(colors)]))
    clr = str(tuple(plotly.colors.hex_to_rgb(colors[i % len(colors)]))).replace(" ", "")
    clr = f"rgba{clr[:-1]},0.2)"
    fig.add_trace(
        go.Scatter(
            x=x.tolist() + x[::-1].tolist(),  # x, then x reversed
            y=(hi).tolist() + (lo[::-1]).tolist(),  # upper, then lower reversed
            fill="toself",
            fillcolor=clr,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=f"{yname}_{ci_level}ci",
        )
    )
    fig.update_layout(xaxis_title=xvar)
    return fig


def add_stats_display_plotly(
    stats,
    selected,
    combine,
    overlay,
    hue,
    xvar,
    cols=None,
    start_col=0,
    ncols_effective=0,
    ci_level=DEFAULT_CI_LEVEL,
):
    displayed, fig = 0, None
    if overlay:
        fig = go.Figure()
    # allcols = [xvar] + ([hue] if hue else []) + selected
    # stats = stats.loc[:, allcols]
    if combine:
        stats = stats.loc[:, [_ for _ in stats.columns if _ != hue]]
    # st.text([xvar, hue, selected])
    # st.table(stats.loc[(stats.step==0) & (stats.agent=="03SyR@1->05Dec@1"), :])
    for i, field in enumerate(selected):
        if not overlay:
            fig = line_with_band(None, stats, xvar, field, color=hue if not combine else None, i=i, ci_level=ci_level)
            fig.update_layout(showlegend=not combine)
            if combine:
                fig.update_layout(yaxis_title=field)
            with cols[(i + start_col) % ncols_effective]:
                displayed += 1
                st.plotly_chart(fig)
            continue
        col_name = "value" if len(selected) > 1 else field.split(":")[-1]
        fig = line_with_band(fig, stats, xvar, field, color=None, i=i, ci_level=ci_level)
        fig.update_layout(yaxis_title=col_name)
        fig.update_layout(showlegend=len(selected) > 1 or not combine)
    if overlay:
        with cols[(displayed + start_col) % ncols_effective]:
            displayed += 1
            st.plotly_chart(fig)
    return cols, displayed + start_col


def add_stats_display(
    stats,
    selected,
    combine,
    overlay,
    hue,
    xvar,
    ncols,
    cols=None,
    start_col=0,
    title=None,
    sectioned=False,
    dynamic=False,
    ci_level=DEFAULT_CI_LEVEL,
):
    if "bankrupt" in stats.columns:
        stats["bankrupt"] = stats.bankrupt.astype(int)
    add_section = False
    if sectioned:
        start_col, cols = 0, None
        if title:
            add_section = True
    if len(selected) < 1:
        return cols, start_col
    allcols = [xvar] + ([hue] if hue else []) + selected
    stats = stats.loc[:, allcols]
    if add_section:
        st.markdown(f"### {title}")
    if sectioned or cols is None:
        ncols_effective = max(1, min(len(stats), ncols))
        cols = st.beta_columns(ncols_effective)
    else:
        ncols_effective = len(cols)
    displayed = 0
    if overlay and not combine:
        data = stats.loc[:, [hue, xvar] + selected]
        data = data.melt(id_vars=[xvar, hue], value_vars=selected)
        data["variable"] = data[hue].astype(str) + ":" + data["variable"]
        if len(selected) == 1:
            col_name = selected[0].split(":")[-1]
            data = data.rename(columns=dict(value=col_name))
        else:
            col_name = "value"
        if dynamic:
            presenter = st.plotly_chart
            fig = line_with_band(None, data, xvar, col_name, color="variable", i=0, ci_level=ci_level)
            fig.update_layout(yaxis_title=col_name)
            fig.update_layout(showlegend=len(selected) > 1 or not combine)
        else:
            presenter = st.pyplot
            fig, ax = plt.subplots()
            sns.lineplot(data=data, x=xvar, y=col_name, hue="variable", ax=ax, style=None)
        with cols[(displayed + start_col) % ncols_effective]:
            displayed += 1
            presenter(fig)
        return cols, displayed + start_col
    runner = add_stats_display_plotly if dynamic else add_stats_display_sns
    cols, end_col = runner(
        stats,
        selected,
        combine,
        overlay,
        hue,
        xvar,
        cols=cols,
        start_col=start_col,
        ncols_effective=ncols_effective,
        ci_level=ci_level,
    )
    return cols, end_col


def plot_network(nodes, node_weights=None, color_title=None, edges=[], edge_weights=None, title=""):
    if not node_weights:
        node_weights = [1] * len(nodes)
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = nodes[edge[0]]['pos']
        x1, y1 = nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_info = []
    for node in nodes:
        x, y = nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_info.append(str(nodes[node]))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'eys' | 'YlBu' | 'eens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale=plotly.colors.col
            reversescale=True,
            color=[],
            size=30,
            colorbar=dict(
                thickness=12,
                title=color_title,
                xanchor='left',
                titleside='right'
            ) if color_title else None,
            line_width=2))

    node_text = []
    for node in nodes:
        node_text.append(str(node))

    node_trace.marker.color = node_weights
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    # annotations=[ dict(
                    #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    #     showarrow=False,
                    #     xref="paper", yref="paper",
                    #     x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig
