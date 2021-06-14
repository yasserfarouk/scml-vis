#!/usr/bin/env python
import itertools
import random
import sys
import traceback
from pathlib import Path

import altair as alt
import pandas as pd
import plotly as plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_numeric_dtype
from plotly.validators.scatter.marker import SymbolValidator
from streamlit import cli as stcli

import scml_vis.compiler as compiler
from scml_vis.compiler import VISDATA_FOLDER
from scml_vis.utils import (
    add_selector,
    add_stats_display,
    add_stats_selector,
    load_data,
    plot_network,
    score_distribution,
    score_factors,
)

__all__ = ["main"]


MARKERS = SymbolValidator().values[2::3]
MARKERS = [_ for _ in MARKERS if not any(_.startswith(x) for x in ("star", "circle", "square"))]
random.shuffle(MARKERS)
MARKERS = ["circle", "square"] + MARKERS

DB_FOLDER = Path.home() / "negmas" / "runsdb"
DB_NAME = "rundb.csv"
BASE_FOLDERS = [
    Path.home() / "negmas" / "logs" / "scml" / "scml2020",
    Path.home() / "negmas" / "logs" / "scml" / "scml2020oneshot",
    Path.home() / "negmas" / "logs" / "scml" / "scml2021oneshot",
    Path.home() / "negmas" / "logs" / "scml" / "scml2021",
    Path.home() / "negmas" / "logs" / "tournaments",
]


def main(folder: Path):
    st.set_page_config(layout="wide")
    if folder is None:
        options = dict(none="none")
        if (DB_FOLDER / DB_NAME).exists():
            data = pd.read_csv(DB_FOLDER / DB_NAME, index_col=None, header=None)
            data: pd.DataFrame
            data = data.iloc[::-1]
            data.columns = ["name", "type", "path"]
            for _, x in data.iterrows():
                options[x["path"]] = f"{x['type'][0]}:{x['name']}"
        add_base = st.sidebar.checkbox("Add Default paths", True)
        if add_base:
            for base in BASE_FOLDERS:
                type_ = base.name == "tournaments"
                for child in base.glob("*"):
                    if not child.is_dir() or not compiler.has_logs(child):
                        continue
                    options[child] = f"{'t' if type_ else 'w'}:{child.name}"

        folder = st.sidebar.selectbox("Select a run", list(options.keys()), format_func=lambda x: options[x])
    if not folder or (isinstance(folder, str) and folder == "none"):
        st.text(
            "Cannot find any folders with logs.\nTry looking in default paths by checking 'Add Default paths' \nin the side bar or start the app with a folder containing log data using -f"
        )
        return
    folder = Path(folder)
    if folder.name != VISDATA_FOLDER:
        folder = folder / VISDATA_FOLDER
    if not folder.exists():
        try:
            do_compile = st.sidebar.button("Compile visualization data?")
            if do_compile:
                try:
                    compiler.main(folder.parent, max_worlds=None)
                except Exception as e:
                    st.write(f"*Failed to compile visualization data for {folder}*\n### Exception:\n{str(e)}")
                    st.write(f"\n### Traceback:\n```\n{traceback.format_exc()}```")
            else:
                st.text("Either press 'Compile visualization data' to view logs of this folder or choose another one.")
                return
        except:
            st.write(f"Folder {folder} contains no logs to use")
        # folder = folder / VISDATA_FOLDER
        # if not folder.exists():
        #     st.write(
        #         f"## SCML Visualizer\nError: No {VISDATA_FOLDER} folder found with visualization data at {str(folder)}"
        #     )
        #     return
    if folder.name != VISDATA_FOLDER:
        folder = folder / VISDATA_FOLDER
    if not folder.exists():
        st.write("Cannot find visualiation data")
        return

    st.write(f"## SCML Visualizer\n{str(folder.parent)}")

    st.sidebar.markdown("## Data Selection")
    tournaments = load_data(folder, "tournaments")
    tournament_expander = st.sidebar.beta_expander("Tournament Selection")
    with tournament_expander:
        selected_tournaments, _, _ = add_selector(
            st,
            "",
            tournaments["name"].unique(),
            key="tournaments",
            none=False,
            default="one",
        )
    worlds = None
    configs = load_data(folder, "configs")
    if configs is None:
        worlds = load_data(folder, "worlds")
        config_names = worlds.loc[:, "name"].str.split("_").str[0].unique()
        configs = pd.DataFrame(data=config_names, columns=["id"])
    config_expander = st.sidebar.beta_expander("Config Selection")
    with config_expander:
        selected_configs, _, _ = add_selector(
            st,
            "",
            configs["id"].unique(),
            key="configs",
            none=False,
            default="all",
        )

    if worlds is None:
        worlds = load_data(folder, "worlds")
    if "config" not in worlds.columns:
        worlds["config"] = worlds.loc[:, "name"].str.split("_").str[0]
    worlds = worlds.loc[worlds.tournament.isin(selected_tournaments) & worlds.config.isin(selected_configs), :]
    world_expander = st.sidebar.beta_expander("World Selection")
    with world_expander:
        selected_worlds, _, _ = add_selector(st, "", worlds.name, key="worlds", none=False, default="all")
    worlds = worlds.loc[(worlds.name.isin(selected_worlds)), :]

    agents = load_data(folder, "agents")
    type_expander = st.sidebar.beta_expander("Type Selection")
    with type_expander:
        selected_types, _, _ = add_selector(st, "", agents.type.unique(), key="types", none=False, default="all")
    agents = agents.loc[(agents.type.isin(selected_types)), :]

    agent_expander = st.sidebar.beta_expander("Agent Selection")
    with agent_expander:
        selected_agents, _, _ = add_selector(st, "", agents.name.unique(), key="agents", none=False, default="all")

    products = load_data(folder, "product_stats")
    product_expander = st.sidebar.beta_expander("Product Selection")
    with product_expander:
        selected_products, _, _ = add_selector(
            st, "", products["product"].unique(), key="products", none=False, default="all"
        )

    agents = agents.loc[(agents.type.isin(selected_types)), :]

    nsteps = worlds.loc[worlds.name.isin(selected_worlds), "n_steps"].max()
    nsteps = int(nsteps)
    selected_steps = st.sidebar.slider("Steps", 0, nsteps, (0, nsteps))
    selected_times = st.sidebar.slider("Relative Times", 0.0, 1.0, (0.0, 1.0))

    st.sidebar.markdown("## Figure Selection")
    # ts_figs = st.sidebar.beta_expander("Time Series")
    # net_figs = st.sidebar.beta_expander("Networks")
    # tbl_figs = st.sidebar.beta_expander("Tables")
    # other_figs = st.sidebar.beta_expander("Others")
    #     if len(selected_worlds) == 1:
    #         fig_type = st.sidebar.selectbox(label="", options=["Time-series", "Networks", "Tables", "Others"], index=1)
    #     else:
    #         fig_type = st.sidebar.selectbox(label="", options=["Time-series", "Tables", "Others"], index=1)
    #
    #     if fig_type == "Time-series":
    #         runner = display_time_series
    #     elif fig_type == "Networks":
    #         runner = display_networks
    #     elif fig_type == "Tables":
    #         runner = display_tables
    #     elif fig_type == "Others":
    #         runner = display_others
    #     else:
    #         st.text("Please choose what type of figures are you interested in")
    #         return
    products_summary = (
        products.loc[:, [_ for _ in products.columns if _ not in ("step", "relative_time")]]
        .groupby(["tournament", "world", "product"])
        .agg(["min", "max", "mean", "std"])
    )
    products_summary.columns = [f"{a}_{b}" for a, b in products_summary.columns]
    products_summary = products_summary.reset_index()
    data = dict(t=tournaments, w=worlds, a=agents, p=products_summary)

    def filter(x, agent_field_sets):
        if x is None:
            return x
        x = x.loc[(x.world.isin(selected_worlds)), :]
        indx = None
        for fields in agent_field_sets:
            if not fields:
                continue
            indx = x[fields[0]].isin(selected_agents)
            for f in fields[1:]:
                indx = (indx) | (x[f].isin(selected_agents))
        if indx is None:
            return x
        return x.loc[indx, :]

    data["con"] = load_data(folder, "configs")
    data["a"] = load_data(folder, "agents")
    data["t"] = load_data(folder, "types")
    data["c"] = filter(load_data(folder, "contracts"), [["buyer", "seller"]])
    data["n"] = filter(load_data(folder, "negotiations"), [["buyer", "seller"]])
    data["o"] = filter(load_data(folder, "offers"), [["sender", "receiver"]])
    for runner, section_name in [
        (display_networks, "Networks"),
        (display_others, "Overview"),
        (display_tables, "Tables"),
        (display_time_series, "Time Series"),
    ]:
        if section_name != "Time Series":
            expander = st.sidebar.beta_expander(section_name, section_name == "Networks")
            do_expand = expander.checkbox(f"Show {section_name}", section_name == "Networks")
        else:
            expander = st.sidebar
            do_expand = st.sidebar.checkbox(section_name, True)
        if do_expand:
            runner(
                folder,
                selected_worlds,
                selected_products,
                selected_agents,
                selected_types,
                selected_steps,
                selected_times,
                data,
                parent=expander,
            )
            # st.sidebar.markdown("""---""")


def filter_by_time(x, cols, selected_steps, selected_times):
    indx = None
    for k in cols:
        step_col, time_col = f"{k}step", f"{k}relative_time"
        i = (x[step_col] >= selected_steps[0]) & (x[step_col] <= selected_steps[1])
        i &= (x[time_col] >= selected_times[0]) & (x[time_col] <= selected_times[1])
        if indx is None:
            indx = i
        else:
            indx |= i
    if indx is not None:
        return x.loc[indx, :]
    return x


def show_a_world(
    world,
    selected_steps,
    selected_times,
    data,
    parent,
    weight_field,
    edge_weights,
    edge_colors,
    node_weight,
    condition_field,
    x,
    src,
    gallery,
):
    nodes = data["a"].loc[data["a"].world == world, :]
    nodes["score*cost"] = nodes["final_score"] * nodes["cost"]
    fields = [_ for _ in nodes.columns]
    nodes = nodes.to_dict("records")
    added = -data["a"].input_product.min()
    nlevels = data["a"].input_product.max() + 1 + added

    level_max = [0] * (nlevels)
    dx, dy = 10, 10
    for node in nodes:
        l = node["input_product"] + added
        node["pos"] = ((l + 1) * dx, level_max[l] * dy)
        level_max[l] += 1

    nodes = {n["name"]: n for n in nodes}
    seller_dict = dict(zip(fields, itertools.repeat(float("nan"))))
    buyer_dict = dict(zip(fields, itertools.repeat(float("nan"))))
    nodes["SELLER"] = {**seller_dict, **dict(pos=(0, dy * (level_max[0] // 2)), name="Seller", type="System")}
    nodes["BUYER"] = {
        **buyer_dict,
        **dict(pos=((nlevels + 1) * dx, dy * (level_max[-1] // 2)), name="Buyer", type="System"),
    }
    edges, weights = [], []
    weight_field_name = "quantity" if weight_field == "count" else weight_field
    time_cols = (
        [condition_field + "_step", condition_field + "_relative_time"]
        if condition_field != "step"
        else ["step", "relative_time"]
    )
    x = x.loc[x.world == world, [weight_field_name, "seller", "buyer"] + time_cols]
    x = filter_by_time(x, [condition_field + "_" if condition_field != "step" else ""], selected_steps, selected_times)
    x.drop(time_cols, axis=1, inplace=True)
    if weight_field == "unit_price":
        x = x.groupby(["seller", "buyer"]).mean().reset_index()
        x["unit_price"].fillna(0.0, inplace=True)
    elif weight_field == "count":
        x = x.groupby(["seller", "buyer"]).count().reset_index()
        x.rename(columns=dict(quantity="count"), inplace=True)
    else:
        x = x.groupby(["seller", "buyer"]).sum().reset_index()
    for _, d in x.iterrows():
        edges.append((d["seller"], d["buyer"], d[weight_field]))
    parent.plotly_chart(
        plot_network(
            fields, nodes, edges=edges, node_weights=node_weight, edge_colors=edge_colors, edge_weights=edge_weights
        )
    )
    if gallery:
        return

    col1, col2 = parent.beta_columns(2)
    mydata = data[src]
    myselected = mydata.loc[(mydata.world == world), :]
    myselected = filter_by_time(
        myselected, [condition_field + "_" if condition_field != "step" else ""], selected_steps, selected_times
    )
    seller = col1.selectbox("Seller", [""] + sorted(x["seller"].unique()), key=f"seller-{world}")
    buyer = col2.selectbox("Buyer", [""] + sorted(x["buyer"].unique()), key=f"seller-{world}")
    if seller:
        myselected = myselected.loc[(myselected.seller == seller), :]
    if buyer:
        myselected = myselected.loc[(myselected.buyer == buyer), :]
    myselected = myselected.reset_index()
    options = myselected
    if src == "n":
        col1, col2 = parent.beta_columns(2)
        broken = col1.checkbox("Broken", False, key=f"broken-{world}")
        timedout = col2.checkbox("Timedout", False, key=f"timedout-{world}")
        if not broken:
            options = options.loc[~options.broken, :]
        if not timedout:
            options = options.loc[~options.timedout, :]

    # options = options.loc[(options["seller"]==seller) & (options["buyer"]==buyer) & (options.world == world) & (options[f"{condition_field}_step"]<= selected_steps[1]) & (options[f"{condition_field}_step"]>= selected_steps[0]) , :]

    if src == "c":
        displayed_cols = (
            [
                "id",
                "delivery_step",
                "quantity",
                "unit_price",
                "total_price",
                "n_neg_steps",
                "concluded_step",
                "signed_step",
                "executed_step",
                "negotiation",
            ]
            + (["buyer"] if not buyer else [])
            + (["seller"] if not seller else [])
        )
    elif src == "n":
        displayed_cols = ["id", "delivery_step", "quantity", "unit_price", "timedout", "broken", "step", "rounds"]
    else:
        return
    parent.dataframe(
        myselected.loc[:, displayed_cols].sort_values(
            ["signed_step", "delivery_step"] if src == "c" else ["step", "delivery_step"]
        )
    )
    contract = None

    options = filter_by_time(
        options, [condition_field + "_" if condition_field != "step" else ""], selected_steps, selected_times
    )
    if parent.checkbox("Ignore Exogenous", key=f"ignore-exogenous-{world}"):
        options = options.loc[(options["buyer"] != "BUYER") & (options["seller"] != "SELLER"), :]
    if src == "n":
        options = options.loc[:, "id"].values
        if len(options) < 1:
            return
        neg = parent.selectbox(label="Negotiation", options=options, key=f"negotiationselect-{world}")
    elif src == "c":
        options = options.loc[:, "id"].values
        if len(options) < 1:
            return
        elif len(options) == 1:
            contract = options[0]
        else:
            contract = parent.selectbox(label="Contract", options=options, key=f"contractselect-{world}")
        neg = myselected.loc[myselected["id"] == contract, "negotiation"]
        if len(neg) > 0:
            neg = neg.values[0]
        else:
            neg = None
    else:
        return
    if contract is not None:
        parent.write(data["c"].loc[data["c"]["id"] == contract, :])
    if not neg or data["n"] is None or len(data["n"]) == 0:
        return
    neg_info = data["n"].loc[data["n"]["id"] == neg]
    offers = data["o"]
    offers = offers.loc[offers.negotiation == neg, :].sort_values(["round", "sender"])
    # if len(offers) >= 2:
    #     offers = offers.loc[offers["sender"].shift(1) != offers["sender"],:]
    offers.index = range(len(offers))

    parent.write(neg_info)
    if len(neg_info) < 1:
        return
    neg_info = neg_info.to_dict("records")[0]
    if not neg_info["broken"] and not neg_info["timedout"]:
        agreement = dict(
            quantity=neg_info["quantity"],
            delivery_step=neg_info["delivery_step"],
            unit_price=neg_info["unit_price"],
            total_price=neg_info["unit_price"] * neg_info["quantity"],
        )
    else:
        agreement = None
    parent.markdown(f"**Agreement**: {agreement}")

    trange = (neg_info["min_delivery_step"], neg_info["max_delivery_step"])
    c1, c2 = parent.beta_columns(2)
    if trange[1] > trange[0]:
        is_3d = c2.checkbox("3D Graph", key=f"threed-{world}")
    else:
        is_3d = False
    use_ranges = c1.checkbox("Use issue ranges to set axes", True, key=f"useissueranges-{world}")
    qrange = (neg_info["min_quantity"] - 1, neg_info["max_quantity"] + 1)
    urange = (neg_info["min_unit_price"] - 1, neg_info["max_unit_price"] + 1)
    if is_3d:
        fig = go.Figure()
        for i, sender in enumerate(offers["sender"].unique()):
            myoffers = offers.loc[offers["sender"] == sender, :]
            fig.add_trace(
                go.Scatter3d(
                    x=myoffers["quantity"],
                    y=myoffers["unit_price"],
                    z=myoffers["delivery_step"],
                    name=sender,
                    mode="lines+markers",
                    marker=dict(size=10),
                    marker_symbol=MARKERS[i],
                )
            )
        if agreement:
            fig.add_trace(
                go.Scatter3d(
                    x=[agreement["quantity"]],
                    y=[agreement["unit_price"]],
                    z=[agreement["delivery_step"]],
                    mode="markers",
                    marker=dict(size=20),
                    name="Agreement",
                    marker_symbol="diamond",
                )
            )
        fig.update_layout(xaxis_title="quantity", yaxis_title="unit_price")
    else:
        fig = go.Figure()
        for i, sender in enumerate(offers["sender"].unique()):
            myoffers = offers.loc[offers["sender"] == sender, :]
            fig.add_trace(
                go.Scatter(
                    x=myoffers["quantity"],
                    y=myoffers["unit_price"],
                    name=sender,
                    mode="lines+markers",
                    marker=dict(size=10),
                    marker_symbol=MARKERS[i],
                )
            )
        if agreement:
            fig.add_trace(
                go.Scatter(
                    x=[agreement["quantity"]],
                    y=[agreement["unit_price"]],
                    mode="markers",
                    marker=dict(size=20),
                    name="Agreement",
                    marker_symbol="star",
                )
            )
        fig.update_layout(xaxis_title="quantity", yaxis_title="unit_price")
        if use_ranges:
            fig.update_layout(xaxis_range=qrange, yaxis_range=urange)
    col1, col2 = parent.beta_columns(2)

    def fig_1d(y):
        fig = go.Figure()
        for i, sender in enumerate(offers["sender"].unique()):
            myoffers = offers.loc[offers["sender"] == sender, :]
            fig.add_trace(
                go.Scatter(
                    x=myoffers["round"],
                    y=myoffers[y],
                    name=sender,
                    mode="lines+markers",
                    marker=dict(size=15),
                    marker_symbol=MARKERS[i],
                )
            )
        if agreement:
            fig.add_trace(
                go.Scatter(
                    x=[offers["round"].max()],
                    y=[agreement[y]],
                    mode="markers",
                    marker=dict(size=20),
                    name="Agreement",
                    marker_symbol="star",
                )
            )
        fig.update_layout(xaxis_title="Round", yaxis_title=y)
        fig.update_layout(yaxis_range=urange if y == "unit_price" else qrange if y == "quantity" else trange)
        return fig

    col1.plotly_chart(fig_1d("quantity"))
    col1.plotly_chart(fig)
    col2.plotly_chart(fig_1d("unit_price"))
    if trange[1] > trange[0]:
        col2.plotly_chart(fig_1d("delivery_step"))

    parent.dataframe(offers)


WORLD_INDEX = 0


def display_networks(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
    parent=st.sidebar,
):
    global WORLD_INDEX
    max_worlds = parent.number_input("Max. Worlds", 1, None, 4)

    if len(selected_worlds) < 1:
        st.write("No worlds selected. Cannot show any networks")
        return
    if len(selected_worlds) > max_worlds:
        st.write(f"More than {max_worlds} world selected ({len(selected_worlds)}). Will show the first {max_worlds}")
        cols = st.beta_columns([1, 5, 1, 3])
        # prev = cols[0].button("<")
        # next = cols[2].button(">")
        # if prev:
        #     WORLD_INDEX = (WORLD_INDEX - max_worlds) % len(selected_worlds)
        # if next:
        #     WORLD_INDEX = (WORLD_INDEX + max_worlds) % len(selected_worlds)
        WORLD_INDEX = cols[1].slider("", 0, len(selected_worlds) - 1, WORLD_INDEX)
        randomize = cols[3].button("Randomize worlds")
        if randomize:
            random.shuffle(selected_worlds)
        selected_worlds = selected_worlds[WORLD_INDEX : WORLD_INDEX + max_worlds]
    what = parent.selectbox("Category", ["Contracts", "Negotiations"])
    if what == "Contracts":
        src = "c"
    elif what == "Negotiations":
        src = "n"
    else:
        src = "o"
    x = data[src]
    if x is None:
        st.markdown(f"**{what}** data is **not** available in the logs.")
        return
    gallery = parent.checkbox("Gallery Mode", len(selected_worlds) > 1)
    node_weight_options = sorted(
        [_ for _ in data["a"].columns if is_numeric_dtype(data["a"][_]) and _ not in ("id", "is_default")]
    )
    default_node_weight = node_weight_options.index("final_score")
    if default_node_weight is None:
        default_node_weight = 0
    with st.beta_expander("Networks Settings"):
        cols = st.beta_columns(5 + int(gallery))
        weight_field = cols[2].selectbox("Edge Weight", ["total_price", "unit_price", "quantity", "count"])
        node_weight = cols[3].selectbox("Node Weight", ["none"] + node_weight_options, default_node_weight + 1)
        per_step = cols[0].checkbox("Show one step only")
        edge_weights = cols[0].checkbox("Variable Edge Width", True)
        edge_colors = cols[0].checkbox("Variable Edge Colors", True)
        if per_step:
            selected_step = cols[1].number_input("Step", selected_steps[0], selected_steps[1], selected_steps[0])
            selected_steps = [selected_step] * 2
        x["total_price"] = x.quantity * x.unit_price
        options = [_[: -len("_step")] for _ in x.columns if _.endswith("_step")]
        if src != "c":
            options.append("step")
        condition_field = cols[4].selectbox("Condition", options, 0 if src != "n" else options.index("step"))
    if gallery:
        n_cols = cols[5].number_input("Columns", 1, 5, 2)
        cols = st.beta_columns(n_cols)
    else:
        n_cols, cols = 1, [st]

    for i, world in enumerate(selected_worlds):
        show_a_world(
            world,
            selected_steps=selected_steps,
            selected_times=selected_times,
            data=data,
            parent=cols[i % n_cols],
            weight_field=weight_field,
            edge_weights=edge_weights,
            edge_colors=edge_colors,
            node_weight=node_weight,
            condition_field=condition_field,
            x=x,
            src=src,
            gallery=gallery,
        )


def display_tables(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
    parent=st.sidebar,
):
    remove_single = parent.checkbox("Remove fields with a single value", True)

    def order_columns(x):
        cols = sorted(x.columns)
        for c in ["buyer_type", "seller_type", "delivery_step", "quantity", "unit_price", "total_price", "buyer", "seller", "name", "id"]:
            if c in cols:
                cols = [c] + [_ for _ in cols if _ != c]
        for c in ["world", "config", "group", "tournament"]:
            if c in cols:
                cols = [_ for _ in cols if _ != c] + [c]
        return x.loc[:, cols]

    def remove_singletons(x):
        selected = []
        for c in x.columns:
            if len(x[c].unique()) < 2:
                continue
            selected.append(c)
        return x.loc[:, selected]

    def show_table(x, must_choose=False):
        x = order_columns(x)
        if remove_single:
            x = remove_singletons(x)
        selected_cols = st.multiselect(label="Columns", options=x.columns)
        if selected_cols or must_choose:
            st.dataframe(x.loc[:, selected_cols])
        else:
            st.dataframe(x)

    def create_chart(df, type):
        if type == "Scatter":
            return alt.Chart(df).mark_point()
        if type == "Bar":
            return alt.Chart(df).mark_bar()
        if type == "Box":
            return alt.Chart(df).mark_boxplot()
        if type == "Line":
            return alt.Chart(df).mark_line()
        raise ValueError(f"Unknown marker type {type}")

    for lbl, k, has_step in (
        ("Tournaments", "t", False),
        ("Configs", "con", False),
        ("Worlds", "w", False),
        ("Products", "p", False),
        ("Agents", "a", False),
        ("Contracts", "c", True),
        ("Negotiations", "n", True),
        ("Offers", "o", True),
    ):

        if data[k] is None or not len(data[k]):
            continue
        if not parent.checkbox(label=lbl):
            continue
        if has_step:
            df = filter_by_time(
                data[k], ["signed_", "concluded_"] if k == "c" else [""], selected_steps, selected_times
            )
        else:
            df = data[k]
        if lbl == "Agents":
            if st.checkbox("Ignore Default Agents", True):
                df = df.loc[~df["is_default"], :]
        elif lbl == "Contracts":
            if st.checkbox("Ignore Exogenous Contracts", True):
                df = df.loc[df["n_neg_steps"] < 1, :]
        show_table(df)
        st.text(f"{len(df)} records found")
        cols = st.beta_columns(6)
        type_ = cols[0].selectbox("Chart", ["Scatter", "Line", "Bar", "Box"], 0)
        x = cols[1].selectbox("x", ["none"] + list(df.columns))
        y = m = c = s = "none"
        if x != "none":
            y = cols[2].selectbox("y", ["none"] + list(df.columns))
            if y != "none":
                m = cols[3].selectbox("Mark", ["none"] + list(df.columns))
                c = cols[4].selectbox("Color", ["none"] + list(df.columns))
                s = cols[5].selectbox("Size", ["none"] + list(df.columns))
                kwargs = dict(x=x, y=y)
                if m != "none": kwargs["shape"] = m
                if s != "none": kwargs["size"] = s
                if c != "none": kwargs["color"] = c
            else:
                kwargs = dict(x=x, y=alt.X(x, bin=True))
            chart = create_chart(df, type_ if y != "none" else "Bar").encode(**kwargs)
            st.altair_chart(chart, use_container_width=True)


def display_time_series(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
    parent=st.sidebar,
):
    settings = st.beta_expander("Time Series Settings")
    ncols = settings.number_input("N. Columns", min_value=1, max_value=6)
    xvar = settings.selectbox("x-variable", ["step", "relative_time"], 1 - int(len(selected_worlds) == 1))
    dynamic = settings.checkbox("Dynamic Figures", value=True)
    sectioned = settings.checkbox("Figure Sections", True)
    ci_level = settings.selectbox(options=[80, 90, 95], label="CI Level", index=2)
    world_stats, selected_world_stats, combine_world_stats, overlay_world_stats = add_stats_selector(
        folder,
        "world_stats",
        [[("world", selected_worlds), ("step", selected_steps), ("relative_time", selected_times)]],
        xvar=xvar,
        label="World Statistics",
        choices=lambda x: [
            _ for _ in x.columns if _ not in ("name", "world", "name", "tournament", "type", "step", "relative_time")
        ],
        default_selector="one",
    )

    product_stats, selected_product_stats, combine_product_stats, overlay_product_stats = add_stats_selector(
        folder,
        "product_stats",
        [[("product", selected_products), ("step", selected_steps), ("relative_time", selected_times)]],
        xvar=xvar,
        label="Product Statistics",
        choices=lambda x: [
            _
            for _ in x.columns
            if _ not in ("name", "world", "name", "tournament", "type", "step", "product", "relative_time")
        ],
        default_selector="some",
        default_choice=["trading_price"],
        combine=False,
    )

    type_stats, selected_type_stats, combine_type_stats, overlay_type_stats = add_stats_selector(
        folder,
        "agent_stats",
        [
            [
                ("world", selected_worlds),
                ("type", selected_types),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ]
        ],
        xvar=xvar,
        label="Type Statistics",
        choices=lambda x: [
            _ for _ in x.columns if _ not in ("name", "world", "name", "tournament", "type", "step", "relative_time")
        ],
        key="type",
        default_selector="some" if len(selected_worlds) != 1 else "none",
        default_choice=["score"] if len(selected_worlds) != 1 else None,
        combine=False,
    )

    agent_stats, selected_agent_stats, combine_agent_stats, overlay_agent_stats = add_stats_selector(
        folder,
        "agent_stats",
        [
            [
                ("world", selected_worlds),
                ("name", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ]
        ],
        xvar=xvar,
        label="Agent Statistics",
        choices=lambda x: [
            _ for _ in x.columns if _ not in ("name", "world", "name", "tournament", "type", "step", "relative_time")
        ],
        default_selector="some" if len(selected_worlds) == 1 else "none",
        default_choice=["score"] if len(selected_worlds) == 1 else None,
        combine=False,
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (World)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (Types)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (Agents)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            # [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Buyer Types)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (Seller Types)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            # [("world", selected_worlds), ("seller", selected_agents)],
        ],
        xvar=xvar,
        label="Contract Statistics (Buyer)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (Seller)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
            [
                ("world", selected_worlds),
                ("buyer", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
            [
                ("world", selected_worlds),
                ("seller", selected_agents),
                ("step", selected_steps),
                ("relative_time", selected_times),
            ],
        ],
        xvar=xvar,
        label="Contract Statistics (Product)",
        default_selector="none",
        choices=lambda x: [
            _ for _ in x.columns if _.endswith("quantity") or _.endswith("count") or _.endswith("price")
        ],
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
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
        ci_level=ci_level,
    )


def display_others(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
    parent=st.sidebar,
):
    # settings = parent.beta_expander("Settings")
    # ncols = settings.number_input("N. Columns", min_value=1, max_value=6)
    if parent.checkbox("Score Distribution", False):
        score_distribution(selected_worlds, selected_agents, selected_types, data, parent=parent)
    if parent.checkbox("Final Score Factors", False):
        score_factors(selected_worlds, selected_agents, selected_types, data, parent=parent)


if __name__ == "__main__":
    import sys

    from streamlit import cli as stcli

    folder = None
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])

    if st._is_running_with_streamlit:
        main(folder)
    else:
        sys.argv = ["streamlit", "run"] + sys.argv
        sys.exit(stcli.main())
