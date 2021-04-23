#!/usr/bin/env python
import sys
from pathlib import Path

import streamlit as st
from streamlit import cli as stcli

from scml_vis.compiler import VISDATA_FOLDER
from scml_vis.utils import add_stats_display, add_stats_selector, load_data, add_selector, plot_network

__all__ = ["main"]


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
        selected_products = add_selector(
            st, "", products["product"].unique(), key="products", none=False, default="all"
        )

    agents = agents.loc[(agents.type.isin(selected_types)), :]

    nsteps = int(worlds.loc[worlds.name.isin(selected_worlds), "n_steps"].max())
    selected_steps = st.sidebar.slider("Steps", 0, nsteps, (0, nsteps))
    selected_times = st.sidebar.slider("Relative Times", 0.0, 1.0, (0.0, 1.0))

    st.sidebar.markdown("## Figure Selection")
    if len(selected_worlds) == 1:
        fig_type = st.sidebar.selectbox(label="", options=["Time-series", "Networks", "Tables"])
    else:
        fig_type = "Time-series"

    if fig_type == "Time-series":
        runner = display_time_series
    elif fig_type == "Networks":
        runner = display_networks
    elif fig_type == "Tables":
        runner = display_tables
    else:
        st.text("Please choose what type of figures are you interested in")
        return
    products_summary = (
        products.loc[:, [_ for _ in products.columns if _ not in ("step", "relative_time")]]
        .groupby(["tournament", "world", "product"])
        .agg(["min", "max", "mean", "std"])
    )
    products_summary.columns = [f"{a}_{b}" for a, b in products_summary.columns]
    products_summary = products_summary.reset_index()
    data = dict(t=tournaments, w=worlds, a=agents, p=products_summary)

    def filter(x, agent_field_sets):
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

    data["c"] = filter(load_data(folder, "contracts"), [["buyer", "seller"]])
    data["n"] = filter(load_data(folder, "negotiations"), [["buyer", "seller"]])
    data["o"] = filter(load_data(folder, "offers"), [["sender", "receiver"]])
    runner(
        folder,
        selected_worlds,
        selected_products,
        selected_agents,
        selected_types,
        selected_steps,
        selected_times,
        data,
    )


def display_networks(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
):
    nodes = data["a"].to_dict("records")
    nlevels = data["a"].input_product.max() + 1
    level_max = [0] * (nlevels)
    dx, dy = 10, 10
    for node in nodes:
        l = node["input_product"]
        node["pos"] = ((l + 1) * dx, level_max[l] * dy)
        level_max[l] += 1
    nodes = {n["name"]: n for n in nodes}
    nodes["SELLER"] = dict(pos=(0, dy * (level_max[0] // 2)), name="Seller", type="System")
    nodes["BUYER"] = dict(pos=((nlevels + 1) * dx, dy * (level_max[-1] // 2)), name="Seller", type="System")
    what = st.sidebar.selectbox("Category", ["Contracts", "Negotiations", "Offers"])

    st.plotly_chart(plot_network(nodes))


def display_tables(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
):
    def show_table(x, must_choose=False):
        selected_cols = st.multiselect(label="Columns", options=x.columns)
        if selected_cols or must_choose:
            st.dataframe(x.loc[:, selected_cols])
        else:
            st.dataframe(x)

    def filtered(x, cols):
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

    for lbl, k in (
        ("Tournaments", "t"),
        ("Worlds", "w"),
        ("Products", "p"),
        ("Agents", "a"),
        ("Contracts", "c"),
        ("Negotiations", "n"),
        ("Offers", "o"),
    ):
        if data[k] is None or not len(data[k]):
            continue
        if st.sidebar.checkbox(label=lbl):
            show_table(filtered(data[k], ["signed_", "concluded_"] if k == "c" else [""]))


def display_time_series(
    folder,
    selected_worlds,
    selected_products,
    selected_agents,
    selected_types,
    selected_steps,
    selected_times,
    data,
):
    settings = st.sidebar.beta_expander("Settings")
    ncols = settings.number_input("N. Columns", min_value=1, max_value=6)
    xvar = settings.selectbox("x-variable", ["step", "relative_time"])
    dynamic = settings.checkbox("Dynamic Figures", value=True)
    sectioned = settings.checkbox("Figure Sections")
    ci_level = settings.selectbox(options=[80, 90, 95], label="CI Level", index=2)
    world_stats, selected_world_stats, combine_world_stats, overlay_world_stats = add_stats_selector(
        folder,
        "world_stats",
        [[("world", selected_worlds), ("step", selected_steps), ("relative_time", selected_times)]],
        xvar=xvar,
        label="World Statistics",
        default_selector="none",
    )

    product_stats, selected_product_stats, combine_product_stats, overlay_product_stats = add_stats_selector(
        folder,
        "product_stats",
        [[("product", selected_products), ("step", selected_steps), ("relative_time", selected_times)]],
        xvar=xvar,
        label="Product Statistics",
        default_selector="none",
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
        default_selector="none",
        key="type",
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


if __name__ == "__main__":
    import sys

    from streamlit import cli as stcli

    folder = Path(sys.argv[1])

    if st._is_running_with_streamlit:
        main(folder)
    else:
        sys.argv = ["streamlit", "run"] + sys.argv
        sys.exit(stcli.main())
