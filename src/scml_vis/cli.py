#!/usr/bin/env python
import sys
from functools import partial
from pathlib import Path

import click
import click_config_file
from streamlit import cli as stcli

try:
    from scml_vis.vendor.quick.quick import gui_option
except:

    def gui_option(x):
        return x


import scml_vis.compiler as compiler
import scml_vis.presenter as presenter
from scml_vis.compiler import has_visdata

click.option = partial(click.option, show_default=True)


@gui_option
@click.group()
def main():
    pass


@main.command(help="Opens the visualizer")
@click.argument("folder", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "-w",
    "--max-worlds",
    default=None,
    type=int,
    help="Maximum number of worlds to keep in the compiled visualization data",
)
def show(folder: Path, max_worlds: int):
    folder = Path(folder)
    if not compiler.has_visdata(folder):
        compiler.main(folder, max_worlds)

    sys.argv = ["streamlit", "run", str(Path(__file__).parent / "presenter.py"), str(folder)]
    sys.exit(stcli.main())


@main.command(help="Compiles the data needed for visualization from a given log folder")
@click.argument("folder", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "-w",
    "--max-worlds",
    default=None,
    type=int,
    help="Maximum number of worlds to keep in the compiled visualization data",
)
def compile(folder: Path, max_worlds):
    return compiler.main(folder, max_worlds)


if __name__ == "__main__":
    main()
