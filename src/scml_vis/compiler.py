#!/usr/bin/env python

import pandas as pd
import numpy as np
from typing import Iterable
from pathlib import Path
import json
from pathlib import Path
import sys

__all__ = ["has_visdata", "has_logs", "main", "VISDATA_FOLDER"]

VISDATA_FOLDER = "_visdata"
# tournament files
SCORES_FILE = "scores.csv"
CONFIGS_FILE = "assigned_configs.json"  # has n_steps, __dir_name
# WORLD_STATS_FILE = "stats.csv" # has path and world

TOURNAMENT_REQUIRED = [SCORES_FILE, CONFIGS_FILE]

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
            if not (folder/ f).exists():
                return False
        return True
    if check_tournament:
        for f in TOURNAMENT_REQUIRED:
            if not (folder/ f).exists():
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


def nonzero(f):
    return f.exists() and f.stat().st_size > 0


def get_folders(base_folder, main_file, required):
    return [
        _.parent
        for _ in base_folder.glob(f"**/{main_file}")
        if all(nonzero(f) for f in [_.parent / n for n in required])
    ]


def is_tournament(base_folder):
    return all(nonzero(f) for f in [base_folder / n for n in TOURNAMENT_REQUIRED])


def is_world(base_folder):
    return all(nonzero(f) for f in [base_folder / n for n in WORLD_REQUIRED])


def get_torunaments(base_folder):
    return get_folders(base_folder, main_file=CONFIGS_FILE, required=TOURNAMENT_REQUIRED)


def get_worlds(base_folder):
    return get_folders(base_folder, main_file=AGENTS_FILE, required=WORLD_REQUIRED)


def parse_tournament(path, t_indx, base_indx):
    configs = json.load(open(path / CONFIGS_FILE))
    if not configs:
        return None, None, None
    scores = pd.read_csv(path / SCORES_FILE).to_dict("records") # typing: none
    if not scores:
        return None, None, None
    worlds = []
    world_indx = dict()
    world_names = set()
    agents = []
    for i, _ in enumerate(configs):
        if i > MAXWORLDS:
            break
        c = _[0]
        world_names.add(c["world_params"]["name"])
        p = c["__dir_name"]
        for k, v in PATHMAP.items():
            p = p.replace(k, v)
        worlds.append(
            dict(
                id=i + base_indx,
                path=p,
                name=c["world_params"]["name"],
                n_steps=c["world_params"]["n_steps"],
                n_processes=c["world_params"]["n_processes"],
                n_agents=len(c["is_default"]),
                tournament=path.name,
                tournament_indx=t_indx,
            )
        )
        world_indx[worlds[-1]["id"]] = worlds[-1]["name"]
        _, wa = get_basic_world_info(Path(p), path.name)
        agents.append(wa)

    agents = pd.concat(agents)

    # for i, s in enumerate(scores):
    #     if s["world"] not in (world_names):
    #         continue
    #     agents.append(
    #         dict(
    #             id=i + base_indx,
    #             name=s["agent_id"],
    #             type=s["agent_type"],
    #             # id=s["agent_id"],
    #             final_score=s["score"],
    #             world=s["world"],
    #             world_id=world_indx.get(s["world"], None),
    #         )
    #     )
    # agents = pd.DataFrame.from_records(agents)
    return worlds, agents


def parse_world(path, tname, wname, nsteps, agents, w_indx, base_indx):
    stats = pd.read_csv(path / STATS_FILE, index_col=0)
    if (path / AGENTS_JSON_FILE).exists():
            scored_agents = set(agents["name"].to_list())
            ag_dict = json.load(open(path / AGENTS_JSON_FILE))
            for k, v in ag_dict.items():
                if k in ("BUYER", "SELLER"):
                    v["final_score"] = float("nan")
                elif k in scored_agents:
                    a, b = stats[f"score_{k}"].values[-1], float(agents.loc[agents["name"]==k, "final_score"].values)
                    assert a == b
                    v["final_score"] = float(agents.loc[agents["name"]==k, "final_score"].values)
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
        x.columns = ["_".join(_.split("_")[:-1]) if _.endswith(f"_{p}") else _ for _ in x.columns ]
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


def get_basic_world_info(path, tname):
    stats = pd.read_csv(path / STATS_FILE, index_col=0).to_dict("list")
    adata = json.load(open(path / AGENTS_JSON_FILE))
    winfo = json.load(open(path / INFO_FILE))
    worlds = [dict(name=path.name, tournament=tname, tournament_indx=0
    , path=path, n_steps=winfo["n_steps"])]
    agents = []
    for i, (aname, info) in enumerate(adata.items()):
        if f"score_{aname}" not in stats.keys():
            continue
        score = stats[f"score_{aname}"][-1]
        aginfo = winfo["agent_profiles"][aname]
        if "costs" in aginfo.keys():
            aginfo["cost"] = float(np.asarray(aginfo["costs"]).min())
            del aginfo["costs"]
        dd = dict(id=i, name=aname, world=worlds[0]["name"], tournament=tname, final_score=score, type=info["type"])
        dd = {**dd, **aginfo}
        agents.append( dd)
    return worlds, pd.DataFrame.from_records(agents)


def get_data(base_folder):
    base_folder = Path(base_folder)
    tournaments, worlds, agents, agent_stats, product_stats, world_stats = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    contracts, contract_stats, breaches = [], [], []
    negotiations, offers, neg_stats = [], [], []
    if is_tournament(base_folder):
        paths = get_torunaments(base_folder)
    elif is_world(base_folder):
        paths = [None]
    else:
        raise ValueError(f"Folder {str(base_folder)} contains not tournament or world logs")
    for i, t in enumerate(paths):
        indx = i + 1
        base_indx = (i + 1) * 1_000_000
        if t is not None:
            print(f"Processing {t.name} [{i} of {len(paths)}]", flush=True)
            tournaments.append(dict(id=indx, path=t, name=t.name))
            w, a = parse_tournament(t, indx, base_indx)
            tname = t.name
        else:
            print(f"No tournament found ... reading world info")
            tname = "none"
            tournaments.append(dict(id=tname, path=base_folder.parent, name=tname))
            w, a = get_basic_world_info(base_folder, tname)
        for j, world in enumerate(w):
            print(f"\tWorld {world['name']} [{j} of {len(w)}]", flush=True)
            wagents = a.loc[a.world == world["name"]]
            ag, pr, wo, co, cs, ng, of, ns, br = parse_world(
                Path(world["path"]),
                tname,
                world["name"],
                world["n_steps"],
                wagents,
                base_indx + j + 1,
                base_indx + j + 1,
            )
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


def main(folder: Path, max_worlds: int):
    if max_worlds is None:
        max_worlds = float("inf")
    dst_folder = folder / VISDATA_FOLDER
    if dst_folder.exists():
        print(f"Destiantion folder {dst_folder} exists. Delete it if you want to recompile visualization data")
        return
    folder = Path(folder)
    (
        tournaments,
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
    ) = get_data(folder)
    dst_folder = folder / VISDATA_FOLDER
    dst_folder.mkdir(parents=True, exist_ok=True)
    for df, name in zip(
        (
            tournaments,
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
            continue
        df = adjust_type_names(df)
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
