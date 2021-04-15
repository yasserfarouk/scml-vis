# scml-vis

[![ci](https://github.com/yasserfarouk/scml-vis/actions/workflows/main.yml/badge.svg)](https://github.com/yasserfarouk/scml-vis/actions/workflows/main.yml)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://scml-vis.github.io/scml-vis/)
[![pypi version](https://img.shields.io/pypi/v/scml-vis.svg)](https://pypi.org/project/scml-vis/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/scml-vis/community)

A simple visualiser for SCML worlds and tournaments

## Screenshots
![Screen Shot 1](docs/shot1.png)
![Screen Shot 2](docs/shot2.png)

## Main Features

- Displays any world/tournament run using the [SCML package](https://www.github.com/yasserfarouk/scml)
- Allows filtering using worlds, agent types, and agent instances
- Shows world statistics, agent type and instance statistics and contract 
  statistics as functions of simulation step/time

## TODO List (Good Ideas for PRs)

- Show negotiation logs (i.e. negotiation results)
- Display all contracts (i.e. in a table) based on selection criteria
- Zoom on negotiation details (i.e. exchanged offers)
- Add dynamic figures using plotly/altair
- Add networkx like graphs of contracts / negotiations / offers
- Add new figure types that do not have time/step in the x-axis
- Allow starting the app without specifying a folder. In this case, it should 
  have a dropdown list showing the latest tournament/world runs (requires 
  changes to [scml](https://www.github.com/yasserfarouk/scml))

## Requirements

scml-vis requires Python 3.8 or above.

## Installation

With `pip`:
```bash
python3 -m pip install scml-vis
```

With [`pipx`](https://github.com/pipxproject/pipx):
```bash
python3 -m pip install --user pipx

pipx install scml-vis
```
