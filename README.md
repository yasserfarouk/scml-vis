# scml-vis

[![ci](https://github.com/scml-vis/scml-vis/workflows/ci/badge.svg)](https://github.com/scml-vis/scml-vis/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://scml-vis.github.io/scml-vis/)
[![pypi version](https://img.shields.io/pypi/v/scml-vis.svg)](https://pypi.org/project/scml-vis/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/scml-vis/community)

A simple visualiser for SCML worlds and tournaments

## Requirements

scml-vis requires Python 3.8 or above.

<details>
<summary>To install Python 3.8, I recommend using <a href="https://github.com/pyenv/pyenv"><code>pyenv</code></a>.</summary>

```bash
# install pyenv
git clone https://github.com/pyenv/pyenv ~/.pyenv

# setup pyenv (you should also put these three lines in .bashrc or similar)
export PATH="${HOME}/.pyenv/bin:${PATH}"
export PYENV_ROOT="${HOME}/.pyenv"
eval "$(pyenv init -)"

# install Python 3.8
pyenv install 3.8.12

# make it available globally
pyenv global system 3.8.12
```
</details>

## Installation

With `pip`:
```bash
python3.8 -m pip install scml-vis
```

With [`pipx`](https://github.com/pipxproject/pipx):
```bash
python3.8 -m pip install --user pipx

pipx install --python python3.8 scml-vis
```
