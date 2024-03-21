#!/usr/bin/env python
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import click
from streamlit.web import cli as stcli

import scml_vis.compiler as compiler
from scml_vis.compiler import VISDATA_FOLDER, has_visdata

click.option = partial(click.option, show_default=True)


@click.group()
def main():
    pass


@main.command(help="Opens the visualizer")  # type: ignore
@click.option(
    "-f",
    "--folder",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Folder containing logs of a world or tournament to open. If not given, last runs from SCML or a list of predefined locations will be used",
)
@click.option(
    "-w",
    "--max-worlds",
    default=None,
    type=int,
    help="Maximum number of worlds to keep in the compiled visualization data",
)
def show(folder: Path, max_worlds: int):
    # folder = Path(folder) if folder is not None else None
    if folder and not has_visdata(folder):
        try:
            compiler.main(folder, max_worlds)
        except Exception as e:
            print(
                f"Failed to compile visualization data for {folder}:\nException: {str(e)}"
            )
    if folder:
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).parent / "presenter.py"),
            str(folder),
        ]
    else:
        sys.argv = ["streamlit", "run", str(Path(__file__).parent / "presenter.py")]
    sys.exit(stcli.main())


@main.command(help="Compiles the data needed for visualization from a given log folder")  # type: ignore
@click.argument("folder", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "-i",
    "--ignore",
    default="",
    type=str,
    help="Pattern of foldernames to ignore (uses regex)",
)
@click.option(
    "-w",
    "--max-worlds",
    default=None,
    type=int,
    help="Maximum number of worlds to keep in the compiled visualization data",
)
@click.option(
    "-m",
    "--pathmap",
    default=None,
    type=str,
    help="path maps to apply to all files in all files before compiling in before:after format.\n"
    "For example abc/def:xyz/123 will map all abc/def mentions in all files to xyz/123 before compiling\n"
    "Useful when the logs are moved from where they were created",
)
def compile(folder: Path, max_worlds: int, ignore: str, pathmap: Optional[str] = None):
    return compiler.main(folder, max_worlds, ignore=ignore, pathmap=pathmap)


@main.command(help="Creates an SQLite dataset and explore it using Datasette")  # type: ignore
@click.argument("folder", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "-w",
    "--max-worlds",
    default=None,
    type=int,
    help="Maximum number of worlds to keep in the compiled visualization data",
)
def explore(folder: Path, max_worlds):
    folder = Path(folder)
    if not compiler.has_visdata(folder):
        compiler.main(folder, max_worlds)
    dst = str(folder / VISDATA_FOLDER / "dataset.sqlite")
    files = [str(_.absolute()) for _ in (folder / VISDATA_FOLDER).glob("*.csv")]
    subprocess.run(["csvs-to-sqlite"] + files + [dst])
    subprocess.run(["datasette", dst])


if __name__ == "__main__":
    main()
