#!/usr/bin/env python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Callable
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly as plotly
from matplotlib import pyplot as plt
from streamlit import cli as stcli
from scml_vis.compiler import VISDATA_FOLDER

__all__ = ["main"]


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
                x = x.loc[world_stats[field].isin(values), :]
            if len(x) > 0:
                x = x.index
                if filtered is None:
                    filtered = x
                else:
                    filtered = filtered.union(x)
        world_stats = world_stats.loc[filtered, :]
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

def add_line_with_band(fig, stats, xvar, yvar, color, i):
    colors = px.colors.qualitative.Plotly
    stats = stats.groupby([xvar]).agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats = stats.reset_index()
    x, y, s = stats[xvar], stats[f"{yvar}_mean"], stats[f"{yvar}_std"]
    fig.add_trace(go.Scatter(x=x, y=y, name=field, line_color=colors[i]))
    x, y, s = stats[xvar], stats[f"{yvar}_mean"], stats[f"{yvar}_std"]
    clr = str(tuple(plotly.colors.hex_to_rgb(colors[i]))).replace(" ", "")
    clr = f"rgba{clr[:-1]},0.2)"
    fig.add_trace(
        go.Scatter(
            x=x.tolist() + x[::-1].tolist(),  # x, then x reversed
            y=(y + s).tolist() + (y[::-1] - s[::-1]).tolist(),  # upper, then lower reversed
            fill="toself",
            fillcolor=clr,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig

def line_with_band(fig, stats, xvar, yvar, color, i, color_val=None, ci_level=90):
    if color is not None:
        for i, v in enumerate(stats[color].unique()):
            fig = line_with_band(fig, stats.loc[stats[color]==v, :], xvar, yvar, None, i, color_val=v)
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
        base = c[:-len("_mean")]
        if not f"{base}_std" in stats.columns or not f"{base}_count" in stats.columns:
            stats[f"{base}"] = stats[c]
            stats = stats.drop([c])
            continue
        stats[f"{base}_ci_hi"] = stats[c]
        stats[f"{base}_ci_lo"] = stats[c]
        indx = stats[f"{base}_count"] > 0
        stats.loc[indx, f"{base}_ci_hi"] = stats.loc[indx, c] + (1+ci_level / 100.0) * stats.loc[indx, f"{base}_std"] / stats.loc[indx, f"{base}_count"].apply(np.sqrt)
        stats.loc[indx, f"{base}_ci_lo"] = stats.loc[indx, c] - (1+ci_level / 100.0) * stats.loc[indx, f"{base}_std"] / stats.loc[indx, f"{base}_count"].apply(np.sqrt)
        stats[f"{base}"] = stats[c]
        stats = stats.drop([c, f"{base}_std", f"{base}_count"], axis=1)
    stats = stats.reset_index()
    x, y, hi, lo = stats[xvar], stats[f"{yvar}"], stats[f"{yvar}_ci_hi"], stats[f"{yvar}_ci_lo"]
    if fig is None:
        fig = go.Figure()

    yname = yvar if not color_val else f"{color_val}:{yvar}" if ":" not in color_val else color_val
    fig.add_trace(go.Scatter(x=x, y=y, name=yname, line_color=colors[i % len(colors)]))
    #     fig = px.line(
    #         stats,
    #         x=xvar,
    #         y=f"{yvar}_mean",
    #         color=color,
    #     )
    # else:
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
            name=f"{yname}_{ci_level}ci"
        )
    )
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
):
    displayed, fig = 0, None
    if overlay:
        fig = go.Figure()
    # allcols = [xvar] + ([hue] if hue else []) + selected
    # stats = stats.loc[:, allcols]
    if combine:
        stats = stats.loc[:, [_ for _ in stats.columns if _!=hue]]
    # st.text([xvar, hue, selected])
    # st.table(stats.loc[(stats.step==0) & (stats.agent=="03SyR@1->05Dec@1"), :])
    for i, field in enumerate(selected):
        if not overlay:
            fig = line_with_band(None, stats, xvar, field, color=hue if not combine else None, i=i)
            fig.update_layout(showlegend=not combine)
            with cols[(i + start_col) % ncols_effective]:
                displayed += 1
                st.plotly_chart(fig)
            continue
        col_name = "value" if len(selected) > 1 else field.split(":")[-1]
        fig = line_with_band(fig, stats, xvar, field, color=None, i=i) 
        fig.update_layout(xaxis_title=xvar)
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
            fig = line_with_band(None, data, xvar, col_name, color="variable", i=0)
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
    )
    return cols, end_col


def main(folder: Path):
    folder = Path(folder)
    if folder.name != VISDATA_FOLDER:
        folder = folder / VISDATA_FOLDER
        if not folder.exists():
            st.write(
                f"## SCML Visualizer\nError: No {VISDATA_FOLDER} folder found with visualization data at {str(folder)}"
            )
            return

    st.write(f"## SCML Visualizer\n{str(folder.parent)}")
    col1, col2, col3, col4 = st.beta_columns([1, 2, 2, 2])
    ncols = col1.number_input("N. Columns", min_value=1, max_value=6)
    xvar = col2.selectbox("x-variable", ["step", "relative_time"])
    dynamic = col3.checkbox("Dynamic Figures", value=True)
    sectioned = col3.checkbox("Figure Sections")

    st.sidebar.markdown("## Data Selection")
    tournaments = load_data(folder, "tournaments")
    tournament_expander = st.sidebar.beta_expander("Tournament Selection")
    with tournament_expander:
        selected_tournaments = add_selector(
            st,
            "",
            tournaments.name.unique(),
            key="tournaments",
            none=False,
            default="one",
        )

    worlds = load_data(folder, "worlds")
    worlds = worlds.loc[worlds.tournament.isin(selected_tournaments), :]
    world_expander = st.sidebar.beta_expander("World Selection")
    with world_expander:
        selected_worlds = add_selector(st, "", worlds.name, key="worlds", none=False, default="one")
    worlds = worlds.loc[(worlds.name.isin(selected_worlds)), :]

    agents = load_data(folder, "agents")
    type_expander = st.sidebar.beta_expander("Type Selection")
    with type_expander:
        selected_types = add_selector(st, "", agents.type.unique(), key="types", none=False, default="all")
    agents = agents.loc[(agents.type.isin(selected_types)), :]

    agent_expander = st.sidebar.beta_expander("Agent Selection")
    with agent_expander:
        selected_agents = add_selector(st, "", agents.name.unique(), key="agents", none=False, default="all")

    products = load_data(folder, "product_stats")
    product_expander = st.sidebar.beta_expander("Product Selection")
    with product_expander:
        selected_products = add_selector(st, "", products["product"].unique(), key="products", none=False, default="all")

    agents = agents.loc[(agents.type.isin(selected_types)), :]

    st.sidebar.markdown("## Figure Selection")

    world_stats, selected_world_stats, combine_world_stats, overlay_world_stats = add_stats_selector(
        folder,
        "world_stats",
        [[("world", selected_worlds)]],
        xvar=xvar,
        label="World Statistics",
        default_selector="none",
    )

    product_stats, selected_product_stats, combine_product_stats, overlay_product_stats = add_stats_selector(
        folder,
        "product_stats",
        [[("product", selected_products)]],
        xvar=xvar,
        label="Product Statistics",
        default_selector="none",
    )

    type_stats, selected_type_stats, combine_type_stats, overlay_type_stats = add_stats_selector(
        folder,
        "agent_stats",
        [[("world", selected_worlds), ("type", selected_types)]],
        xvar=xvar,
        label="Type Statistics",
        choices=lambda x: [
            _ for _ in x.columns if _ not in ("name", "world", "name", "tournament", "type", "step", "relative_time")
        ],
        default_selector="none",
        key="type",
    )

    agent_stats, selected_agent_stats, combine_agent_stats, overlay_agent_stats = add_stats_selector(
        folder,
        "agent_stats",
        [[("world", selected_worlds), ("name", selected_agents)]],
        xvar=xvar,
        label="Agent Statistics",
        choices=lambda x: [
            _ for _ in x.columns if _ not in ("name", "world", "name", "tournament", "type", "step", "relative_time")
        ],
        default_selector="none",
    )

    (
        contract_stats_world,
        selected_contract_stats_world,
        combine_contract_stats_world,
        overlay_contract_stats_world,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (World)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")],
        key="world",
    )

    (
        contract_stats_type,
        selected_contract_stats_type,
        combine_contract_stats_type,
        overlay_contract_stats_type,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Types)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count")  or _.endswith("price")],
        key="type",
    )
    (
        contract_stats_agent,
        selected_contract_stats_agent,
        combine_contract_stats_agent,
        overlay_contract_stats_agent,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Agents)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")],
        key="name",
    )

    def aggregate_contract_stats(stats, ignored_cols):
        cols = [_ for _ in stats.columns if not any(_.endswith(x) for x in ["price", "quantity", "count"])]
        ignored_cols = [_ for _ in cols if _.startswith(ignored_cols)]
        cols = [_ for _ in cols if not _ in ignored_cols]
        allcols = [_ for _ in stats.columns if not _ in ignored_cols]
        # st.text(stats.columns)
        # st.text(allcols)
        # st.text(cols)
        # st.text(len(stats))
        stats = stats.loc[:, allcols].groupby(cols).sum()
        # st.text(len(stats))
        for c in stats.columns:
            if c.endswith("unit_price"):
                base = "_".join(c.split("_")[:-2])
                stats[c] = stats[f"{base}_total_price"] / stats[f"{base}_quantity"]
                stats[c] = stats[c].fillna(0)
        # st.text(len(stats))
        return stats.reset_index()

    (
        contract_stats_buyer_type,
        selected_contract_stats_buyer_type,
        combine_contract_stats_buyer_type,
        overlay_contract_stats_buyer_type,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            # [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Buyer Types)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count")  or _.endswith("price")],
        key="buyer_type",
    )
    (
        contract_stats_seller_type,
        selected_contract_stats_seller_type,
        combine_contract_stats_seller_type,
        overlay_contract_stats_seller_type,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            # [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Seller Types)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count")  or _.endswith("price")],
        key="seller_type",
    )
    (
        contract_stats_buyer,
        selected_contract_stats_buyer,
        combine_contract_stats_buyer,
        overlay_contract_stats_buyer,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            # [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Buyer)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")],
        key="buyer",
    )

    (
        contract_stats_seller,
        selected_contract_stats_seller,
        combine_contract_stats_seller,
        overlay_contract_stats_seller,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            # [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Seller)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")],
        key="seller",
    )

    contract_stats_buyer = aggregate_contract_stats(contract_stats_buyer, "seller")
    contract_stats_seller = aggregate_contract_stats(contract_stats_seller, "buyer")
    contract_stats_buyer_type = aggregate_contract_stats(contract_stats_buyer, "seller_type")
    contract_stats_seller_type = aggregate_contract_stats(contract_stats_seller, "buyer_type")

    contract_stats_agent["agent"] = contract_stats_agent["seller"] + "->" + contract_stats_agent["buyer"]
    contract_stats_agent["agent_type"] = contract_stats_agent["seller_type"] + "->" + contract_stats_agent["buyer_type"]
    contract_stats_type["agent"] = contract_stats_type["seller"] + "->" + contract_stats_type["buyer"]
    contract_stats_type["agent_type"] = contract_stats_type["seller_type"] + "->" + contract_stats_type["buyer_type"]

    (
        contract_stats_product,
        selected_contract_stats_product,
        combine_contract_stats_product,
        overlay_contract_stats_product,
    ) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Product)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")],
        key="product",
    )

    cols, start_col = add_stats_display(
        world_stats,
        selected_world_stats,
        combine_world_stats,
        overlay_world_stats,
        ncols=ncols,
        xvar=xvar,
        hue="world",
        title="World Figures",
        sectioned=sectioned,
        cols=None,
        start_col=0,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        product_stats,
        selected_product_stats,
        combine_product_stats,
        overlay_product_stats,
        ncols=ncols,
        xvar=xvar,
        hue="product",
        title="product Figures",
        sectioned=sectioned,
        cols=None,
        start_col=0,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        type_stats,
        selected_type_stats,
        combine_type_stats,
        overlay_type_stats,
        ncols=ncols,
        xvar=xvar,
        hue="type",
        title="Agent Type Figures",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        agent_stats,
        selected_agent_stats,
        combine_agent_stats,
        overlay_agent_stats,
        ncols=ncols,
        xvar=xvar,
        hue="name",
        title="Agent Instance Figures",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        contract_stats_world,
        selected_contract_stats_world,
        combine_contract_stats_world,
        overlay_contract_stats_world,
        ncols=ncols,
        xvar=xvar,
        hue="world",
        title="Trade Figures (World)",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        contract_stats_type,
        selected_contract_stats_type,
        combine_contract_stats_type,
        overlay_contract_stats_type,
        ncols=ncols,
        xvar=xvar,
        hue="agent_type",
        title="Trade Figures (Agent Type)",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )

    cols, start_col = add_stats_display(
        contract_stats_buyer_type,
        selected_contract_stats_buyer_type,
        combine_contract_stats_buyer_type,
        overlay_contract_stats_buyer_type,
        ncols=ncols,
        xvar=xvar,
        hue="buyer_type",
        title="Trade Figures (Buyer Type)",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        contract_stats_seller_type,
        selected_contract_stats_seller_type,
        combine_contract_stats_seller_type,
        overlay_contract_stats_seller_type,
        ncols=ncols,
        xvar=xvar,
        hue="seller_type",
        cols=cols,
        start_col=start_col,
        title="Trade Figures (Seller Type)",
        sectioned=sectioned,
        dynamic=dynamic,
    )

    cols, start_col = add_stats_display(
        contract_stats_agent,
        selected_contract_stats_agent,
        combine_contract_stats_agent,
        overlay_contract_stats_agent,
        ncols=ncols,
        xvar=xvar,
        hue="agent",
        cols=cols,
        start_col=start_col,
        title="Trade Figures (Agent Instance)",
        sectioned=sectioned,
        dynamic=dynamic,
    )

    cols, start_col = add_stats_display(
        contract_stats_buyer,
        selected_contract_stats_buyer,
        combine_contract_stats_buyer,
        overlay_contract_stats_buyer,
        ncols=ncols,
        xvar=xvar,
        hue="buyer",
        cols=cols,
        start_col=start_col,
        title="Trade Figures (Buyer Instance)",
        sectioned=sectioned,
        dynamic=dynamic,
    )
    cols, start_col = add_stats_display(
        contract_stats_seller,
        selected_contract_stats_seller,
        combine_contract_stats_seller,
        overlay_contract_stats_seller,
        ncols=ncols,
        xvar=xvar,
        hue="seller",
        title="Trade Figures (Seller Instance)",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )

    cols, start_col = add_stats_display(
        contract_stats_product,
        selected_contract_stats_product,
        combine_contract_stats_product,
        overlay_contract_stats_product,
        ncols=ncols,
        xvar=xvar,
        hue="product",
        title="Trade Figures (Product)",
        sectioned=sectioned,
        cols=cols,
        start_col=start_col,
        dynamic=dynamic,
    )


if __name__ == "__main__":
    from streamlit import cli as stcli
    import sys

    folder = Path(sys.argv[1])

    if st._is_running_with_streamlit:
        main(folder)
    else:
        sys.argv = ["streamlit", "run"] + sys.argv
        sys.exit(stcli.main())
