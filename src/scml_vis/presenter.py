#!/usr/bin/env python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Callable
from operator import index
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from streamlit import cli as stcli
from scml_vis.compiler import VISDATA_FOLDER

__all__ = ["main"]


@st.cache
def load(folder, name):
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
    world_stats = load(folder, file_name)
    if filters:
        filtered = []
        for fset in filters:
            x = world_stats.copy()
            for field, values in fset:
                x = x.loc[world_stats[field].isin(values), :]
            if len(x) > 0:
                filtered.append(x)
        world_stats = pd.concat(filtered, ignore_index=True)

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

def add_stats_display_sns(stats, selected, combine, overlay, hue, xvar, ncols, cols=None, start_col=0, title=None, sectioned=False):
    pass
def add_stats_display_plotly(stats, selected, combine, overlay, hue, xvar, ncols, cols=None, start_col=0, title=None, sectioned=False):
    pass

def add_stats_display(stats, selected, combine, overlay, hue, xvar, ncols, cols=None, start_col=0, title=None, sectioned=False, dynamic=False):
    add_section = False
    if sectioned:
        start_col, cols = 0, None
        if title:
            add_section = True
    if len(selected) < 1:
        return cols, start_col
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
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x=xvar, y="value", hue="variable", ax=ax, style=None)
        with cols[(displayed + start_col) % ncols_effective]:
            displayed += 1
            st.pyplot(fig)
        return cols, displayed + start_col

    if overlay:
        fig, ax = plt.subplots()
    for i, field in enumerate(selected):
        if not overlay:
            fig, ax = plt.subplots()
            plt.title(field)
        sns.lineplot(data=stats, x=xvar, y=field, label=field if overlay else None, ax=ax, hue=hue if not combine else None, style=None)
        if not overlay:
            with cols[(i + start_col) % ncols_effective]:
                displayed += 1
                st.pyplot(fig)
    if overlay:
        with cols[(displayed + start_col) % ncols_effective]:
            displayed += 1
            st.pyplot(fig)
    return cols, displayed + start_col



def main(folder: Path):
    folder = Path(folder)
    if folder.name != VISDATA_FOLDER:
        folder = folder / VISDATA_FOLDER
        if not folder.exists():
            st.write(f"## SCML Visualizer\nError: No {VISDATA_FOLDER} folder found with visualization data at {str(folder)}") 
            return

    st.write(f"## SCML Visualizer\n{str(folder.parent)}")
    col1, col2, col3, col4 = st.beta_columns([1, 2, 2, 2])
    ncols = col1.number_input("N. Columns", min_value=1, max_value=6)
    xvar = col2.selectbox("x-variable", ["step", "relative_time"])
    dynamic = col3.checkbox("Dynamic Figures")
    sectioned = col3.checkbox("Figure Sections")

    st.sidebar.markdown("## Data Selection")
    tournaments = load(folder, "tournaments")
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

    worlds = load(folder, "worlds")
    worlds = worlds.loc[worlds.tournament.isin(selected_tournaments), :]
    world_expander = st.sidebar.beta_expander("World Selection")
    with world_expander:
        selected_worlds = add_selector(st, "", worlds.name, key="worlds", none=False, default="one")
    worlds = worlds.loc[(worlds.name.isin(selected_worlds)), :]

    agents = load(folder, "agents")
    type_expander = st.sidebar.beta_expander("Type Selection")
    with type_expander:
        selected_types = add_selector(st, "", agents.type.unique(), key="types", none=False, default="all")
    agents = agents.loc[(agents.type.isin(selected_types)), :]

    agent_expander = st.sidebar.beta_expander("Agent Selection")
    with agent_expander:
        selected_agents = add_selector(st, "", agents.name.unique(), key="agents", none=False, default="all")

    st.sidebar.markdown("## Figures Selection")
    world_stats, selected_world_stats, combine_world_stats, overlay_world_stats = add_stats_selector(
        folder,
        "world_stats",
        [[("world", selected_worlds)]],
        xvar=xvar,
        label="World Statistics",
        default_selector="none",
    )

    type_stats, selected_type_stats, combine_type_stats, overlay_type_stats = add_stats_selector(
        folder,
        "agent_stats",
        [[("world", selected_worlds), ("type", selected_types)]],
        xvar=xvar,
        label="Type Statistics",
        choices=lambda x: [_ for _ in x.columns if _ not in ("agent", "world", "name", "tournament", "type", "step", "relative_time")],
        default_selector="none",
        key="type",
    )

    agent_stats, selected_agent_stats, combine_agent_stats, overlay_agent_stats = add_stats_selector(
        folder,
        "agent_stats",
        [[("world", selected_worlds), ("agent", selected_agents)]],
        xvar=xvar,
        label="Agent Statistics",
        choices=lambda x: [_ for _ in x.columns if _ not in ("agent", "world", "name", "tournament", "type", "step", "relative_time")],
        default_selector="none",
    )

    (contract_stats_world, selected_contract_stats_world, combine_contract_stats_world, overlay_contract_stats_world,) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (World)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("price")],
        key="world",
    )

    (contract_stats_type, selected_contract_stats_type, combine_contract_stats_type, overlay_contract_stats_type,) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Type)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("price")],
        key="type",
    )
    (contract_stats_agent, selected_contract_stats_agent, combine_contract_stats_agent, overlay_contract_stats_agent,) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Agent)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("price")],
        key="agent",
    )

    (contract_stats_product, selected_contract_stats_product, combine_contract_stats_product, overlay_contract_stats_product,) = add_stats_selector(
        folder,
        "contract_stats",
        [
            [("world", selected_worlds), ("buyer", selected_agents)],
            [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Product)",
        default_selector="none",
        choices=lambda x: [_ for _ in x.columns if _.endswith("quantity") or _.endswith("price")],
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
        cols=None, start_col=0,
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
        hue="agent",
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
        hue="buyer_type",
        title="Trade Figures (Buyer Type)",
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
        hue="buyer",
        cols=cols,
        start_col=start_col,
        title="Trade Figures (Buyer Instance)",
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
