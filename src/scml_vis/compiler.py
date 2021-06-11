#!/usr/bin/env python
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["has_visdata", "has_logs", "main", "VISDATA_FOLDER"]

BASE_IGNORE_SET = {"agent_types", "agent_params", "non_competitors", "non_competitor_params"}
BASE_WORLD_PARAMS_IGNORE_SET = {
    "__dir_name",
    "agent_params",
    "agent_types",
    "name",
    "catalog_prices",
    "profiles",
    "exogenous_contracts",
    "active_lines",
    "input_processes",
    "output_processes",
    "expected_income",
    "expected_income_per_process",
    "expected_income_per_step",
    "expected_n_products",
    "input_quantities",
    "output_quantities",
    "product_prices",
    "process_inputs",
    "process_outputs",
    "profit_basis",
}
ASSIGNED_IGNORE_SET = BASE_IGNORE_SET
ASSIGNED_WORLD_PARAMS_IGNORE_SET = BASE_WORLD_PARAMS_IGNORE_SET
VISDATA_FOLDER = "_visdata"
# tournament files
SCORES_FILE = "scores.csv"
ASSIGNED_CONFIGS_FILE = "assigned_configs.json"  # has n_steps, __dir_name
BASE_CONFIGS_FILE = "base_configs.json"  # has n_steps, __dir_name
# WORLD_STATS_FILE = "stats.csv" # has path and world

TOURNAMENT_REQUIRED = [SCORES_FILE, ASSIGNED_CONFIGS_FILE]

# single world files
CONTRACTS_FILE = ("contracts.csv", "contracts_full_info.csv")
NEGOTIATIONS_FILE = "negotiations.csv"
INFO_FILE = "info.json"
BREACHES_FILE = "breaches.csv"
AGENTS_FILE = "agents.csv"
PARAMS_FILE = "params.csv"
STATS_FILE = "stats.csv"
AGENTS_JSON_FILE = "agents.json"
WORLD_REQUIRED = [STATS_FILE]
MAXWORLDS = float("inf")

PATHMAP = dict()


def has_logs(folder: Path, check_tournament=True, check_world=True) -> bool:
    if check_world:
        for f in WORLD_REQUIRED:
            if not (folder / f).exists():
                return False
        return True
    if check_tournament:
        for f in TOURNAMENT_REQUIRED:
            if not (folder / f).exists():
                return False
        return True
    return True


def adjust_type_names(df):
    if df is None or len(df) == 0:
        return df
    for c in df.columns:
        if not c.endswith("type"):
            continue
        df[c] = df[c].str.split(".").str[0] + "." + df[c].str.split(".").str[-1]
    return df

def adjust_column_names(df):
    if df is None or len(df) == 0:
        return df
    return df.rename(dict(config_id="config", tournament_id="tournament", world_id="world", group_id="group"), axis=1)


def nonzero(f):
    return f.exists() and f.stat().st_size > 0


def get_folders(base_folder, main_file, required, ignore=None):
    return [
        _.parent
        for _ in base_folder.glob(f"**/{main_file}")
        if (not ignore or not re.match(ignore, _.name)) and all(nonzero(f) for f in [_.parent / n for n in required])
    ]


def is_tournament(base_folder):
    return all(nonzero(f) for f in [base_folder / n for n in TOURNAMENT_REQUIRED])


def is_world(base_folder):
    return all(nonzero(f) for f in [base_folder / n for n in WORLD_REQUIRED])


def get_torunaments(base_folder, ignore=None):
    return get_folders(base_folder, main_file=ASSIGNED_CONFIGS_FILE, required=TOURNAMENT_REQUIRED, ignore=ignore)


def get_worlds(base_folder, ignore=None):
    return get_folders(base_folder, main_file=AGENTS_FILE, required=WORLD_REQUIRED, ignore=ignore)


def parse_tournament(
    path,
    t_indx,
    base_indx,
    config_ignore_set=BASE_IGNORE_SET,
    assigned_config_ignore_set=ASSIGNED_IGNORE_SET,
    base_world_ignore_set=BASE_WORLD_PARAMS_IGNORE_SET,
    assigned_world_ignore_set=ASSIGNED_WORLD_PARAMS_IGNORE_SET,
):
    base_configs = json.load(open(path / BASE_CONFIGS_FILE))
    if not base_configs:
        return None, None, None, None
    configs = []
    config_groups = []
    group_map = dict()
    def copy_with_ignore(d, rename_dict, ignore_set, recurse_set):
        c = dict()
        for k, v in d.items():
            if k in ignore_set:
                continue
            if k in recurse_set:
                c = {**c, **copy_with_ignore(v, dict(), ignore_set, recurse_set)}
                continue
            if k in rename_dict.keys():
                c[rename_dict[k]] = v
                continue
            c[k] = v
        return c

    recurse_set = ["world_params", "__exact_params", "info"]
    for aconfig in base_configs:
        conf_group_id = ""
        config_set = []
        for c in aconfig:
            conf_group_id = (conf_group_id + ":" + c["config_id"]) if conf_group_id else c["config_id"]
            d = copy_with_ignore(c, dict(config_id="id"), config_ignore_set.union(base_world_ignore_set), recurse_set)
            config_set.append(d)
        config_groups.append(dict(id=conf_group_id))
        for c in config_set:
            c["group"] = conf_group_id
            configs.append(c)
            group_map[c["id"]] = conf_group_id

    assigned_configs = json.load(open(path / ASSIGNED_CONFIGS_FILE))
    if not assigned_configs:
        return None, None, None, None
    scores = pd.read_csv(path / SCORES_FILE).to_dict("records")  # typing: none
    if not scores:
        return None, None, None, None
    worlds = []
    world_indx = dict()
    world_names = set()
    agents = []
    for i, aconfig in enumerate(assigned_configs):
        if i > MAXWORLDS or not aconfig:
            break
        for c in aconfig:
            world_names.add(c["world_params"]["name"])
            p = c["__dir_name"]
            for k, v in PATHMAP.items():
                p = p.replace(k, v)
            d = dict(
                id=c["world_params"]["name"],
                path=p,
                n_agents=len(c["is_default"]),
                tournament=path.name,
                tournament_indx=t_indx,
            )
            d = {
                **copy_with_ignore(c, dict(), assigned_config_ignore_set.union(assigned_world_ignore_set), recurse_set),
                **d,
            }
            d["name"] = c["world_params"]["name"]
            worlds.append(d)
            world_indx[worlds[-1]["id"]] = worlds[-1]["name"]
            _, wa = get_basic_world_info(Path(p), path.name, worlds[-1]["config_id"], group_map.get(worlds[-1]["config_id"], "none"))
            if not _:
                continue
            agents.append(wa)

    agents = pd.concat(agents)

    return config_groups, configs, worlds, agents


def parse_world(path, tname, wname, nsteps, agents, w_indx, base_indx):
    stats = pd.read_csv(path / STATS_FILE, index_col=0)
    # try:
    #     winfo = json.load(open(path / INFO_FILE))
    # except:
    #     winfo = None
    if (path / AGENTS_JSON_FILE).exists():
        scored_agents = set(agents["name"].to_list())
        ag_dict = json.load(open(path / AGENTS_JSON_FILE))
        for k, v in ag_dict.items():
            if k in ("BUYER", "SELLER"):
                v["final_score"] = float("nan")
            elif k in scored_agents:
                a, b = stats[f"score_{k}"].values[-1], float(agents.loc[agents["name"] == k, "final_score"].values)
                assert a == b
                v["final_score"] = float(agents.loc[agents["name"] == k, "final_score"].values)
            else:
                v["final_score"] = stats[f"score_{k}"].values[-1]
        world_agents = pd.DataFrame.from_records(list(ag_dict.values()))
        world_agents["official_score"] = world_agents["name"].isin(scored_agents)
    else:
        world_agents = agents
        world_agents["official_score"] = True
    contracts = pd.DataFrame(
        data=[],
        columns=[
            "id",
            "seller_name",
            "buyer_name",
            "seller_type",
            "buyer_type",
            "delivery_time",
            "quantity",
            "unit_price",
            "signed_at",
            "nullified_at",
            "concluded_at",
            "signatures",
            "issues",
            "seller",
            "buyer",
            "product_name",
            "n_neg_steps",
            "negotiation_id",
            "executed_at",
            "breaches",
            "erred_at",
            "dropped_at",
        ],
    )
    negotiations = pd.DataFrame(
        data=[],
        columns=[
            "id",
            "seller",
            "buyer",
            "seller_type",
            "buyer_type",
            "delivery_step",
            "delivery_relative_time",
            "quantity",
            "unit_price",
            "step",
            "relative_time",
            "timedout",
            "broken",
            "issues",
            "caller",
            "product",
            "rounds",
            "world",
            "tournament",
            "min_quantity",
            "max_quantity",
            "min_delivery_step",
            "max_delivery_step",
            "min_delivery_relative_time",
            "max_delivery_relative_time",
            "min_unit_price",
            "max_unit_price",
        ],
    )

    offers_list = []
    offers = pd.DataFrame(
        data=[],
        columns=[
            "sender",
            "receiver",
            "sender_type",
            "receiver_type",
            "delivery_step",
            "quantity",
            "unit_price",
            "step",
            "negotiation",
            "relative_time",
            "timedout",
            "broken",
            "caller",
            "product",
            "world",
            "tournament",
            "round",
        ],
    )
    for cname in CONTRACTS_FILE:
        if nonzero(path / cname):
            contracts = pd.read_csv(path / cname, index_col=0)
    if nonzero(path / NEGOTIATIONS_FILE):
        negotiations = pd.read_csv(path / NEGOTIATIONS_FILE, index_col=0)
        negotiations = negotiations.loc[
            :,
            [
                "id",
                "agreement",
                "step",
                "timedout",
                "broken",
                "issues",
                "buyer",
                "seller",
                "caller",
                "product",
                "step",
                "offers",
                "ended_at",
            ],
        ]
        negotiations = negotiations.rename(columns=dict(step="rounds"))
        negotiations = negotiations.rename(columns=dict(ended_at="step"))
        negotiations["offers"] = negotiations.offers.apply(lambda x: eval(x) if isinstance(x, str) else x)
        negotiations["rounds"] = negotiations["rounds"] - 1
        for indx, neg in negotiations.iterrows():
            d = dict(
                sender=None,
                receiver=None,
                delivery_step=None,
                quantity=None,
                unit_price=None,
                round=None,
                index=None,
                step=neg["step"],
                timedout=neg["timedout"],
                broken=neg["broken"],
                caller=neg["caller"],
                product=neg["product"],
                world=wname,
                tournament=tname,
            )
            if not neg["offers"]:
                continue
            offers = eval(neg["offers"]) if isinstance(neg["offers"], str) else neg["offers"]
            current_offers = []
            for k, vs in offers.items():
                for r, v in enumerate(vs):
                    dd = {**d}
                    dd["sender"] = k
                    dd["negotiation"] = neg["id"]
                    dd["receiver"] = neg["seller"] if k == neg["buyer"] else neg["buyer"]
                    dd["round"] = r
                    if v:
                        dd["quantity"] = v[0]
                        dd["delivery_step"] = v[1]
                        dd["unit_price"] = v[2]
                    else:
                        dd["quantity"] = None
                        dd["delivery_step"] = None
                        dd["unit_price"] = None

                    current_offers.append(dd)
            for indx, v in enumerate(current_offers):
                v["index"] = indx
            offers_list += current_offers

        atmap = dict(zip(world_agents["name"].to_list(), world_agents["type"].to_list()))
        # atmap = dict(zip(all_agents["name"].to_list(), all_agents["type"].to_list()))
        # negotiations.agreement[negotiations.agreement.isna(), "agreement"] = None
        for c in ["quantity", "delivery_step", "unit_price"]:
            negotiations[c] = None

        def lst(x):
            return (
                [-1, -1, -1]
                if not x or (isinstance(x, str) and x.lower() == "none")
                else list(eval(x))
                if isinstance(x, str)
                else list(x)
            )

        negotiations.agreement = negotiations.agreement.apply(lst)
        agreements = pd.DataFrame(negotiations.agreement.tolist(), index=negotiations.index)
        agreements.columns = ["quantity", "delivery_step", "unit_price"]
        for c in ["quantity", "delivery_step", "unit_price"]:
            negotiations[c] = agreements[c]
            negotiations.loc[negotiations.loc[:, c] < 0, c] = float("nan")

        def do_map(x):
            return atmap[x]

        negotiations["buyer_type"] = negotiations.buyer.apply(do_map)
        negotiations["seller_type"] = negotiations.seller.apply(do_map)
        negotiations["world"] = wname
        negotiations["tournament"] = tname
        negotiations = negotiations.drop("agreement", axis=1)

        def get_issue_ranges(x):
            if isinstance(x, str):
                x = eval(x)
            if not isinstance(x, Iterable):
                return x
            y = []
            for s in x:
                mn, mx = eval(s.split(":")[-1].strip())
                y += [mn, mx]
            return y

        negotiations = negotiations.reset_index(None)
        issues = negotiations.issues.apply(get_issue_ranges)
        issues = pd.DataFrame(
            issues.to_list(),
            columns=[
                "min_quantity",
                "max_quantity",
                "min_delivery_step",
                "max_delivery_step",
                "min_unit_price",
                "max_unit_price",
            ],
        )
        negotiations = pd.concat((negotiations, issues), axis=1)
        if len(negotiations):
            negotiations.drop("issues", axis=1, inplace=True)
            for c in negotiations.columns:
                if not c.endswith("step") and not c.endswith("steps"):
                    continue
                negotiations.loc[negotiations[c] < 0, c] = float("nan")
                negotiations[c.replace("step", "relative_time")] = negotiations[c] / nsteps if nsteps else 0.0
        offers = pd.DataFrame.from_records(offers_list)
        if len(offers):
            offers["receiver_type"] = offers.receiver.apply(do_map)
            offers["sender_type"] = offers.sender.apply(do_map)
            for c in offers.columns:
                if not c.endswith("step") and not c.endswith("steps"):
                    continue
                offers.loc[offers[c] < 0, c] = float("nan")
                offers[c.replace("step", "relative_time")] = offers[c] / nsteps if nsteps else 0.0

    contracts = contracts.loc[
        :,
        [
            "id",
            "seller_type",
            "buyer_type",
            "delivery_time",
            "quantity",
            "unit_price",
            "signed_at",
            "nullified_at",
            "concluded_at",
            # "issues",
            "seller",
            "buyer",
            "product_name",
            "n_neg_steps",
            "negotiation_id",
            "executed_at",
            "breaches",
            "erred_at",
            "dropped_at",
        ],
    ]
    contracts = contracts.rename(
        columns=dict(
            delivery_time="delivery_step",
            negotiation_id="negotiation",
            product_name="product",
        )
    )
    contracts["product"] = contracts["product"].str[1:].astype(int)
    contracts.columns = [_.replace("_at", "_step") if _.endswith("_at") else _ for _ in contracts.columns]
    # contracts["step"] = contracts["concluded_step"]
    contracts["breached_step"] = contracts["executed_step"]
    contracts["breaches"] = contracts.breaches.astype(str)
    contracts.loc[(contracts.signed_step >= 0) & (contracts.executed_step < 0), "breached_step"] = -1
    contracts.loc[
        (contracts.signed_step >= 0) & (contracts.breaches.str.len() > 0),
        "breached_step",
    ] = -1
    contracts["world"] = wname
    contracts["tournament"] = tname
    # step_cols = (
    #     "",
    #     "concluded_",
    #     "signed_",
    #     "executed_",
    #     "erred_",
    #     "dropped_",
    #     "nullified_",
    #     "breached_",
    # )
    for c in contracts.columns:
        if not c.endswith("step") and not c.endswith("steps"):
            continue
        contracts.loc[contracts[c] < 0, c] = float("nan")
        contracts[c.replace("step", "relative_time")] = contracts[c] / nsteps if nsteps else 0.0

    contract_stats, negotiation_stats = [], []
    non_step_cols = [
        "unit_price",
        "quantity",
        "product",
        "buyer",
        "seller",
        "buyer_type",
        "seller_type",
    ]

    def calc_qp(x):
        c = len(x)
        s = (x["quantity"] * x["unit_price"]).sum()
        q = x["quantity"].sum()
        if q:
            return q, c, s / q
        return 0, 0, 0.0

    for step in range(nsteps):
        s = None
        for c in contracts.columns:
            if not c.endswith("step"):
                continue
            base = c.replace("step", "").replace("_", "")
            qp = contracts.loc[contracts[c] == step, non_step_cols].groupby(non_step_cols[2:]).apply(calc_qp)
            if not isinstance(qp, pd.DataFrame):
                qp = pd.DataFrame(data=qp, index=qp.index)
            if len(qp) < 1:
                continue
            # breakpoint()
            qp.columns = ["qp"]
            # breakpoint()
            qp = pd.DataFrame(qp["qp"].tolist(), index=qp.index)
            qp = qp.rename(columns={0: base + "_quantity", 1: base + "_count", 2: base + "_unit_price"})
            # qp = qp.reset_index()
            qp["step"] = step
            qp.set_index(["step"], append=True, inplace=True)
            qp = qp.melt(
                value_vars=[base + "_quantity", base + "_count", base + "_unit_price"],
                ignore_index=False,
            )
            if s is None:
                s = qp
            else:
                s = pd.concat((s, qp), ignore_index=False, axis=0)
        if s is not None:
            contract_stats.append(s.reset_index())
    # contract_stats = pd.DataFrame.from_records(contract_stats)
    contract_stats = pd.concat(contract_stats, ignore_index=True)
    contract_stats = contract_stats.pivot(
        index=[_ for _ in contract_stats.columns if _ not in ("variable", "value")],
        columns="variable",
        values="value",
    ).reset_index()
    contract_stats["relative_time"] = (contract_stats["step"] / nsteps) if nsteps else 0.0
    contract_stats = contract_stats.fillna(0)
    contract_stats["world"] = wname
    contract_stats["tournament"] = tname
    for col in [_ for _ in contract_stats.columns if _.endswith("unit_price")]:
        field = col.split("_")[0]
        if f"{field}_quantity" in contract_stats.columns:
            contract_stats[f"{field}_total_price"] = (
                contract_stats[f"{field}_quantity"] * contract_stats[f"{field}_unit_price"]
            )

    for c in contract_stats.columns:
        if c.endswith("quantity"):
            contract_stats[c] = contract_stats[c].astype(int)
        if c.endswith("count"):
            contract_stats[c] = contract_stats[c].astype(int)
        if c.endswith("unit_price"):
            contract_stats[c] = contract_stats[c].astype(float)

    # contract_stats.drop_duplicates(inplace=True, subset=["buyer", "seller", "step"])

    if nonzero(path / BREACHES_FILE):
        breaches = pd.read_csv(path / BREACHES_FILE, index_col=0)
    else:
        breaches = pd.DataFrame(
            data=[],
            columns=[
                "contract",
                "contract_id",
                "type",
                "level",
                "id",
                "perpetrator",
                "perpetrator_type",
                "victims",
                "step",
                "resolved",
            ],
        )
    breaches = breaches.loc[
        :,
        [
            "step",
            "contract_id",
            "type",
            "level",
            "id",
            "perpetrator",
            "perpetrator_type",
            "victims",
            "resolved",
        ],
    ]
    breaches = breaches.rename(columns=dict(contract_id="contract"))
    breaches["victim"] = breaches["victims"].apply(lambda x: x[0] if not isinstance(x, str) else eval(x)[0])
    # contracts["relative_time"] = breaches["step"] / nsteps
    breaches.drop(["victims"], axis=1)
    agent_names = world_agents["name"].unique()
    # inventory_{}_input", output
    product_stat_names = ["trading_price", "sold_quantity", "unit_price"]
    products = set([_.split("_")[-1] for _ in stats.columns if any(_.startswith(p) for p in product_stat_names)])
    agent_stat_names = [
        "balance",
        "productivity",
        "bankrupt",
        "score",
        "assets",
        "inventory-input",
        "inventory-output",
        "spot_market_loss",
        "spot_market_quantity",
    ]
    agents_info, product_info = [], []
    world_stat_names = list(
        str(_)
        for _ in stats.columns
        if not any(_.startswith(n.split("-")[0]) for n in (agent_stat_names + product_stat_names))
    )
    for aname in agent_names:
        colnames = []
        ainfo = world_agents.loc[world_agents["name"] == aname, ["type", "final_score"]].to_dict("records")[0]
        for n in agent_stat_names:
            ns = n.split("-")
            if len(ns) > 1:
                col_name = f"{ns[0]}_{aname}_{ns[-1]}"
            else:
                col_name = f"{n}_{aname}"
            if col_name not in stats.columns:
                continue
            colnames.append(col_name)
        x = stats.loc[:, colnames].reset_index().rename(columns=dict(index="step"))
        x.columns = [_ if aname not in _ else _.replace(f"_{aname}", "").replace("-", "_") for _ in x.columns]
        if len(x):
            x["relative_time"] = x["step"] / nsteps
        else:
            x["relative_time"] = 0.0
        x["name"] = aname
        x["world"] = wname
        x["tournament"] = tname
        for k, v in ainfo.items():
            x[k] = v
        agents_info.append(x)

    for p in products:
        colnames = []
        for n in product_stat_names:
            col_name = f"{n}_{p}"
            colnames.append(col_name)
        x = stats.loc[:, colnames].reset_index().rename(columns=dict(index="step"))
        x.columns = ["_".join(_.split("_")[:-1]) if _.endswith(f"_{p}") else _ for _ in x.columns]
        if len(x):
            x["relative_time"] = x["step"] / nsteps
        else:
            x["relative_time"] = 0.0
        x["product"] = p
        x["world"] = wname
        x["tournament"] = tname
        product_info.append(x)

    world_info = stats.loc[:, world_stat_names].reset_index().rename(columns=dict(index="step"))
    if len(world_info):
        world_info["relative_time"] = world_info["step"] / nsteps
    else:
        world_info["relative_time"] = 0.0
    world_info["world"] = wname
    world_info["tournament"] = tname

    return (
        pd.concat(agents_info, ignore_index=True),
        pd.concat(product_info, ignore_index=True),
        world_info,
        contracts,
        contract_stats,
        negotiations,
        offers,
        negotiation_stats,
        breaches,
    )


def get_basic_world_info(path, tname, gname, cname):
    try:
        stats = pd.read_csv(path / STATS_FILE, index_col=0).to_dict("list")
        adata = json.load(open(path / AGENTS_JSON_FILE))
        winfo = json.load(open(path / INFO_FILE))
    except:
        print(f"FAILED {path.name}", flush=True)
        return [], None
    worlds = [dict(name=path.name, tournament=tname, config=cname, group=gname, tournament_indx=0, path=path, n_steps=winfo["n_steps"])]
    agents = []
    definfo = winfo.get("is_default", None)
    agent_key = None
    for k in ("agent_initial_balances", "agent_profiles", "agent_inputs", "agent_outputs", "agent_processes"):
        if k in winfo.keys():
            agent_key = k
            break
    if definfo:
        is_default = dict(zip(winfo[agent_key].keys(), definfo))
    else:
        is_default = dict(zip(winfo[agent_key].keys(), itertools.repeat(False)))
    for i, (aname, info) in enumerate(adata.items()):
        if f"score_{aname}" not in stats.keys():
            continue
        score = stats[f"score_{aname}"][-1]
        aginfo = winfo["agent_profiles"][aname]
        aginfo["initial_balance"] = winfo.get("agent_initial_balances", dict()).get(aname, float("nan"))
        aginfo["is_default"] = is_default.get(aname, True)
        if "costs" in aginfo.keys():
            aginfo["cost"] = float(np.asarray(aginfo["costs"]).min())
            del aginfo["costs"]
        dd = dict(id=i, name=aname, world=worlds[0]["name"], tournament=tname, group=gname, config=cname, final_score=score, type=info["type"])
        dd = {**dd, **aginfo}
        agents.append(dd)
    return worlds, pd.DataFrame.from_records(agents)


def get_data(base_folder, ignore: Optional[str] = None):
    base_folder = Path(base_folder)
    tournaments, worlds, agents, agent_stats, product_stats, world_stats = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    groups, configs = [], []
    contracts, contract_stats, breaches = [], [], []
    negotiations, offers, neg_stats = [], [], []
    if is_tournament(base_folder):
        paths = get_torunaments(base_folder, ignore)
    elif is_world(base_folder):
        paths = [None]
    else:
        paths = get_torunaments(base_folder, ignore)
        paths += get_worlds(base_folder, ignore)
        # raise ValueError(f"Folder {str(base_folder)} contains neither tournament nor world logs")
    none_tournament = dict(id="none", path=base_folder.parent, name="none")
    none_group = dict(id="none")
    none_config = dict(id="none", name="none")
    none_added = False
    for i, t in enumerate(paths):
        indx = i + 1
        base_indx = (i + 1) * 1_000_000
        if t is not None:
            if is_tournament(t):
                print(f"Tournament {t.name} [{i} of {len(paths)}]", flush=True)
                tournaments.append(dict(id=indx, path=t, name=t.name))
                g, con, w, a = parse_tournament(t, indx, base_indx)
                tname = t.name
            else:
                print(f"World {t.name} [{i} of {len(paths)}]", flush=True)
                if not none_added:
                    tournaments.append(none_tournament)
                    none_added = True
                w, a = get_basic_world_info(base_folder, "none", "none", "none")
                g = [{k: v for k, v in none_group.items()}]
                con = [{k: v for k, v in none_config.items()}]
                tname = "none"
        else:
            print(f"No tournament found ... reading world info")
            tname = "none"
            tournaments.append(dict(id=tname, path=base_folder.parent, name=tname))
            g = [{k: v for k, v in none_group.items()}]
            con = [{k: v for k, v in none_config.items()}]
            w, a = get_basic_world_info(base_folder, tname, "none", "none")
        for gg in g:
            gg["tournament"] = tname
        for cc in con:
            cc["tournament"] = tname
        groups += g
        configs += con
        if not w:
            continue
        for j, world in enumerate(w):
            wagents = a.loc[a.world == world["name"]]
            try:
                ag, pr, wo, co, cs, ng, of, ns, br = parse_world(
                    Path(world["path"]),
                    tname,
                    world["name"],
                    world["n_steps"],
                    wagents,
                    base_indx + j + 1,
                    base_indx + j + 1,
                )
            except:
                print(f"\tParse Error {world['name']}", flush=True)
                continue
            print(f"\tWorld {world['name']} [{j} of {len(w)}]", flush=True)
            if ag is not None and len(ag):
                agent_stats.append(ag)
            if pr is not None and len(pr):
                product_stats.append(pr)
            if wo is not None and len(wo):
                world_stats.append(wo)
            if co is not None and len(co):
                contracts.append(co)
            if cs is not None and len(cs):
                contract_stats.append(cs)
            if ng is not None and len(ng):
                negotiations.append(ng)
            if of is not None and len(of):
                offers.append(of)
            if ns is not None and len(ns):
                neg_stats.append(ns)
            if br is not None and len(br):
                breaches.append(br)
        if w is not None and len(w):
            worlds.append(pd.DataFrame.from_records(w))
        if a is not None and len(a):
            agents.append(a)

    tournaments = pd.DataFrame.from_records(tournaments)
    groups = pd.DataFrame.from_records(groups)
    configs = pd.DataFrame.from_records(configs)
    if worlds is not None and len(worlds):
        worlds = pd.concat(worlds, ignore_index=True)
    if agents is not None and len(agents):
        agents = pd.concat(agents, ignore_index=True)
    if agent_stats is not None and len(agent_stats):
        agent_stats = pd.concat(agent_stats, ignore_index=True)
    if product_stats is not None and len(product_stats):
        product_stats = pd.concat(product_stats, ignore_index=True)
    if world_stats is not None and len(world_stats):
        world_stats = pd.concat(world_stats, ignore_index=True)
    if contracts is not None and len(contracts):
        contracts = pd.concat(contracts, ignore_index=True)
    if contract_stats is not None and len(contract_stats):
        contract_stats = pd.concat(contract_stats, ignore_index=True)
    if negotiations is not None and len(negotiations):
        negotiations = pd.concat(negotiations, ignore_index=True)
    if offers is not None and len(offers):
        offers = pd.concat(offers, ignore_index=True)
    if neg_stats is not None and len(neg_stats):
        neg_stats = pd.concat(neg_stats, ignore_index=True)
    if breaches is not None and len(breaches):
        breaches = pd.concat(breaches, ignore_index=True)
    return (
        tournaments,
        groups,
        configs,
        worlds,
        agents,
        agent_stats,
        product_stats,
        world_stats,
        contracts,
        contract_stats,
        negotiations,
        offers,
        neg_stats,
        breaches,
    )


def map_paths(folder: Path, m: Dict[str, str]):
    """Maps paths inside all files within the given folder"""

    for fname in folder.glob("**/*"):
        if fname.is_dir():
            continue
        name = fname.name
        if not name.endswith("json") and not name.endswith("csv"):
            continue
        with open(fname, "r") as infile:
            s = infile.read()
        for k, v in m.items():
            s = s.replace(k, v)
        with open(fname, "w") as outfile:
            outfile.write(s)


def main(folder: Path, max_worlds: Optional[int], ignore: Optional[str] = None, pathmap: Optional[str] = None):
    folder = Path(folder)
    if max_worlds is None:
        max_worlds = float("inf")
    dst_folder = folder / VISDATA_FOLDER
    if dst_folder.exists():
        raise ValueError(
            f"Destiantion folder {dst_folder} exists. Delete it if you want to recompile visualization data"
        )
    folder = Path(folder)

    if pathmap:
        m = pathmap.split(":")
        map_paths(folder, dict(zip(m[0::2], m[1::2])))

    (
        tournaments,
        groups,
        configs,
        worlds,
        agents,
        agent_stats,
        product_stats,
        world_stats,
        contracts,
        contract_stats,
        negotiations,
        offers,
        neg_stats,
        breaches,
    ) = get_data(folder, ignore=ignore)
    dst_folder = folder / VISDATA_FOLDER
    dst_folder.mkdir(parents=True, exist_ok=True)
    for df, name in zip(
        (
            tournaments,
            groups,
            configs,
            worlds,
            agents,
            agent_stats,
            product_stats,
            world_stats,
            contracts,
            contract_stats,
            negotiations,
            offers,
            neg_stats,
            breaches,
        ),
        (
            "tournaments",
            "groups",
            "configs",
            "worlds",
            "agents",
            "agent_stats",
            "product_stats",
            "world_stats",
            "contracts",
            "contract_stats",
            "negotiations",
            "offers",
            "neg_stats",
            "breaches",
        ),
    ):
        if df is None or len(df) == 0:
            print(f"\tDid not find {name}")
            continue
        df = adjust_type_names(df)
        df = adjust_column_names(df)
        df.to_csv(dst_folder / f"{name}.csv", index=False)


def has_visdata(folder: Path):
    folder = Path(folder)
    files = (
        "tournaments",
        "worlds",
        "agents",
        "agent_stats",
        "product_stats",
        "world_stats",
    )
    if folder.name != VISDATA_FOLDER:
        folder /= VISDATA_FOLDER
    if not folder.exists():
        return False
    for file in files:
        if not nonzero(folder / f"{file}.csv"):
            return False
    return True


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        if arg == "--map":
            PATHMAP = {"/export/home": "/Users"}
        elif arg.startswith("n="):
            MAXWORLDS = int(arg.split("=")[-1])
        else:
            path = arg
    main(Path(path), MAXWORLDS)
