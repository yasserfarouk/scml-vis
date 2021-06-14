#!/usr/bin/env python
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from pprint import pformat
import random
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
    "score_distribution",
    "score_factors",
]


@st.cache(allow_output_mutation=True)
def load_data(folder: Path, name: str):
    file = folder / f"{name}.csv"
    if not file.exists():
        return None
    data = pd.read_csv(file, index_col=None)
    if name == "agents":
        data["is-default"] = data["is_default"].astype(int)
        data["is-builtin"] = data["type"].str.startswith("scml")
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
    default_choice=None,
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
        return [], combine, overlay
    if selection_type == "all":
        return content, combine, overlay
    with col2:
        if selection_type == "some":
            selector = st.multiselect("", content, key=f"{key}_multi", default=default_choice)
            return selector, combine, overlay
        if default_choice is not None:
            try:
                indx = content.index(default_choice)
            except ValueError:
                indx = 0
        else:
            indx = 0
        selector = st.selectbox("", content, key=f"{key}_sel", index=indx)
        return [selector], combine, overlay


def add_stats_selector(
    folder,
    file_name,
    filters,
    xvar,
    label,
    default_selector="one",
    choices=None,
    key="",
    default_choice=None,
    combine=True,
    overlay=True,
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
            world_stats = world_stats.loc[[False] * len(world_stats), :]
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
            check=combine,
            check2=overlay,
            default_check=xvar == "step",
            default_choice=default_choice,
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
    if not isinstance(color_val, str):
        color_val = str(color_val)
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
        stats = stats.loc[:, [_ for _ in stats.columns if _ != hue or _ in selected]]
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


def plot_network(fields, nodes, node_weights=None, color_title=None, edges=[], title="", edge_weights=True, edge_colors=True):
    edge_x = []
    edge_y = []
    annotations = []
    min_width, max_width = 1, 7
    weights = []
    colors = []
    for edge in edges:
        # st.text((nodes[edge[0]]["name"], nodes[edge[1]]["name"]))
        x0, y0 = nodes[edge[0]]["pos"]
        x1, y1 = nodes[edge[1]]["pos"]
        if edge_weights or edge_colors:
            edge_x.append([x0, x1, None])
            edge_y.append([y0, y1, None])
        else:
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        fraction = random.random() * 0.4 + 0.2
        slope = (y1-y0) / (x1 - x0)
        dx = fraction * (x1 - x0)
        x = x0 +dx
        y = slope * dx + y0
        # x = min(x0, x1) + fraction * (max(x0, x1) - min(x0, x1))
        # y = min(y0, y1) + fraction * (max(y0, y1) - min(y0, y1))
        if edge_colors:
            clr = tuple([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            clr = f"rgb{clr}"
        else:
            clr = "#888"
        annotations.append((x, y, edge[2]))
        weights.append(edge[2])
        colors.append(clr)

    if weights and len(weights):
        mn, mx = min(weights), max(weights)
        if mx == mn:
            weights = [1]*len(weights)
        else:
            weights = [(_-mn) *(max_width - min_width)/ (mx-mn) + min_width for _ in weights]

    edge_traces = []
    if edge_weights or edge_colors:
        for x, y, w, clr in zip(edge_x, edge_y, weights, colors):
            if np.isnan(w):
                continue
            # "#888"
            line = dict()
            if edge_colors:
                line["color"] = clr
            else:
                line["color"] = "#888"
            if edge_weights:
                line["width"] = w
            edge_traces.append(go.Scatter(x=x, y=y, line=line, hoverinfo="text", mode="lines"))
    else:
        edge_traces.append(go.Scatter(x=edge_x, y=edge_y, hoverinfo="text", mode="lines"))

    node_x = []
    node_y = []
    node_info = []
    node_w = []
    for node in nodes.keys():
        n= nodes[node]
        x, y = n["pos"]
        node_x.append(x)
        node_y.append(y)
        node_info.append(tuple(n.values()))
        node_w.append(n.get(node_weights, 1))

    hovertemplate=""
    if len(nodes):
        for i, k in enumerate(fields):
            if k in ("id", "name", "pos", "is_default"):
                continue
            hovertemplate += f"<b>{k}</b>:%{{customdata[{i}]}}<br>"
    node_trace = go.Scatter(
        name="",
        x=node_x,
        y=node_y,
        mode="markers+text",
        customdata=node_info,
        textposition="top center",
        hovertemplate=hovertemplate,
        marker=dict(
            showscale=True,
            # colorscale options
            #'eys' | 'YlBu' | 'eens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale=plotly.colors.col
            colorscale = "plotly3",
            reversescale=True,
            color=node_w,
            size=30,
            colorbar=dict(thickness=12, title=color_title, xanchor="left", titleside="right") if color_title else None,
            line_width=2,
        ),
    )

    node_text = []
    for node in nodes:
        node_text.append(str(node))
    node_trace.marker.color = node_w
    node_trace.text = node_text

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            # annotations=[ dict(
            #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
            #     showarrow=False,
            #     xref="paper", yref="paper",
            #     x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    for ((x, y, txt), clr) in zip(annotations, colors):
        if edge_colors:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, yshift=10, font=dict(color=clr))
        else:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, yshift=10)
    # st.write(st.get_option("theme"))
    return fig


def score_distribution(selected_worlds, selected_agents, selected_types, data, parent=st.sidebar):
    # st.write(data["a"])
    # st.write(data["a"].groupby(["world", "type", "input_product"])["final_score"].count())
    expander = st.beta_expander("Score Distribution")
    col1, col2, col3, col4 = expander.beta_columns(4)
    is_type = col1.checkbox("Agent Types", value=True, key=f"is_type_check")
    independent_levels = col2.checkbox("Independent Production Levels", value=True, key=f"is_independent_levels")
    no_default = col3.checkbox("No Default Agents", value=True, key=f"no_default_agents")
    selected = selected_types if is_type else selected_agents
    col = "type" if is_type else "name"
    data = data["a"].loc[data["a"].world.isin(selected_worlds) , [col, "world", "tournament", "final_score", "input_product", "is_default"]]
    if no_default:
        data = data.loc[~data.is_default, :]
    data = data.drop("is_default", axis=1)
    data.columns = ["agent", "world", "tournament", "final_score", "level"]
    data = data.loc[data.agent.isin(selected), :]
    if len(data.tournament.unique()) > 0:
        data["world"] = data.tournament + "-" + data.world
    if independent_levels:
        data["agent"] = data.agent + "@" + data.level.astype(str)
    data.drop("tournament", axis=1)
    data.drop("level", axis=1)
    agnts = sorted(data["agent"].unique().tolist())
    n = len(agnts)
    map = dict(zip(agnts, range(n)))
    img = np.zeros((n, n))
    count_img = np.zeros((n, n))
    world_agents = defaultdict(list)
    scores = defaultdict(list)
    counts = defaultdict(list)
    for indx, x in data.iterrows():
        world_agents[x.world].append(x.agent)
        scores[(x.world, x.agent)] = x.final_score
        counts[(x.world, x.agent)] = x.final_score
    for (world, agent), score in scores.items():
        for opponent in world_agents[world]:
            if opponent == agent:
                continue
            img[map[agent], map[opponent]] += score
    for (world, agent), score in counts.items():
        for opponent in world_agents[world]:
            if opponent == agent:
                continue
            count_img[map[agent], map[opponent]] += 1

    expander.write("## Scores")
    col1, col2 = expander.beta_columns(2)
    fig = px.imshow(img, x = agnts, y = agnts)
    col1.plotly_chart(fig)
    col2.plotly_chart(px.bar(data, x="agent", y="final_score"))

    expander.write("## Counts")
    col1, col2 = expander.beta_columns(2)
    fig = px.imshow(count_img, x = agnts, y = agnts)
    col1.plotly_chart(fig)
    scores = data.groupby("agent")["final_score"].count().reset_index()
    scores = scores.rename(columns=dict(final_score="count"))
    col2.plotly_chart(px.bar(scores, x="agent", y="count"))

def score_factors(selected_worlds, selected_agents, selected_types, data, parent=st.sidebar):
    expander = st.beta_expander("Final Score Factors")
    col1, col2, col3 = expander.beta_columns(3)
    show_counts = col2.checkbox("Show counts only", value=False)
    is_type = col1.checkbox("Agent Types", value=True, key=f"is_type_check_factors")
    no_default = col3.checkbox("Ignore Default Agents", value=True, key=f"no_default_agents_factors")
    selected = selected_types if is_type else selected_agents
    col = "type" if is_type else "name"
    data = data["a"].loc[data["a"].world.isin(selected_worlds), :]
    data["config"] = data["world"].str.split("_").str[0]
    data["n"] = 1
    target = "final_score" if not show_counts else "n"
    if no_default:
        data = data.loc[~data.is_default, :]
        data = data.drop("is_default", axis=1)
    data = data.drop("id", axis=1)
    if is_type:
        data = data.drop("name", axis=1)
    else:
        data = data.drop("type", axis=1)
    data = data.rename(columns=dict(name="agent", type="agent"))
    data = data.loc[data.agent.isin(selected), :]
    cols = [_ for _ in data.columns if _ not in ("agent", "id", target)]
    agents = expander.multiselect("Agents", data.agent.unique())
    if len(agents) > 0:
        data = data.loc[data["agent"].isin(agents)]
    facet_col = expander.selectbox("Facet Columns", ["none"] + cols, index=0)
    if facet_col == "none":
        facet_col = None
    else:
        data = data.sort_values(facet_col)
    factors = expander.selectbox("Factors", cols)
    expander.write(f"**Final score vs {factors}**")
    tbl = data.groupby(([] if not facet_col else [facet_col]) + ["agent"] + [factors])[target].describe().reset_index().set_index("agent" if not facet_col else ["agent", facet_col])
    tbl.reset_index(inplace=True)
    graph_type = expander.selectbox("Graph type", ["Scatter", "Bar", "Box", "Line"])

    fig = None
    if graph_type == "Scatter":
        fig = px.scatter(data, x=factors, y=target, color="agent", facet_col=facet_col, facet_col_wrap=3)
    elif graph_type == "Bar":
        fig = px.bar(data, x=factors, y=target, color="agent", facet_col=facet_col, facet_col_wrap=3)
    elif graph_type == "Box":
        fig = px.box(data, x=factors, y=target, color="agent", facet_col=facet_col, facet_col_wrap=3)
    elif graph_type == "Line":
        fig = px.line(tbl, x=factors, y="count" if show_counts else "mean", color="agent", facet_col=facet_col, facet_col_wrap=3, facet_row_spacing=0.01)

    if fig:
        expander.plotly_chart(fig)
    expander.dataframe(tbl)
