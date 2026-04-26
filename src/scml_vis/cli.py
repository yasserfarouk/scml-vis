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
import os
import sqlite3
import pandas as pd
from rich import print


def read_dfs(dir_name, verbose=False):
    dfs = dict()
    for currfile in os.listdir(dir_name):
        path = os.path.join(dir_name, currfile)
        if os.path.isfile(path) and path[-4:] == ".csv":
            fname = currfile[:-4]
            if verbose:
                print(f"Reading {fname}")
            currdf = pd.read_csv(os.path.join(dir_name, currfile), header=0)
            dfs[fname] = currdf
    return dfs


def make_connection(path):
    conn = None
    try:
        conn = sqlite3.connect(str(path))
    except Exception as e:
        print(f"[red]Error creating connection[/red] {e}")
    return conn


def describe_table(cursor, table_name, ncols=6):
    """Describes an SQLite table, listing column names and data types.

    Args:
        db_name (str): The name of the SQLite database file.
        table_name (str): The name of the table to describe.
    """

    cursor.execute(f"PRAGMA table_info({table_name})")
    results = cursor.fetchall()

    # Get the number of records
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    record_count = cursor.fetchone()[0]

    # Print description
    print(f"Table: {table_name} (Records: {record_count})")
    print("---------------------------------")
    # Output in rows of ncols columns
    for i in range(0, len(results), ncols):
        row = results[i : i + ncols]
        for col in row:
            print(f"{col[1]}: {col[2]}  ", end="")  # Note the extra space for alignment
        print()  # Move to the next line after 4 columns


def df2sql(conn, c, df, df_name, verbose=False):
    # Determine the appropriate SQLite data types for each column
    sqlite_types = {
        "int64": "INTEGER",
        "float64": "REAL",
        "object": "TEXT",  # For strings
        "datetime64[ns]": "TIMESTAMP",  # For dates and times
        # ... add more type mappings if needed
    }

    column_types = df.dtypes.apply(lambda x: sqlite_types.get(x.name, "TEXT")).to_dict()

    # Create the table with correct data types
    create_table_query = f"CREATE TABLE IF NOT EXISTS {df_name} ("
    create_table_query += ", ".join(
        [f"{col} {column_types[col]}" for col in df.columns]
    )
    create_table_query += ")"
    try:
        c.execute(create_table_query)
    except Exception as e:
        print(f"[red]Error executing [/red]{create_table_query}\n{e}")
    df.to_sql(df_name, conn, if_exists="replace", index=False)

    c.execute("SELECT * FROM " + df_name)
    if verbose:
        print(f"[green]{df_name}[/green] created with ")
        describe_table(c, df_name)
    # dfOut = pd.DataFrame(c.fetchall())
    # dfOut.columns = [i[0] for i in c.description]


def dfs2csv(path: Path, db: Path, verbose: bool = False):
    dfs = read_dfs(path, verbose=verbose)
    conn = make_connection(db)
    if conn is None:
        return
    c = conn.cursor()

    for df_name, df in dfs.items():
        if verbose:
            print(f"Converting {df_name}")
        df2sql(conn, c, df, df_name, verbose)


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
@click.option(
    "--verbose/--silent",
    default=True,
    type=bool,
)
def explore(folder: Path, max_worlds, verbose):
    folder = Path(folder)
    if not compiler.has_visdata(folder):
        compiler.main(folder, max_worlds)
    dst = folder / VISDATA_FOLDER / "dataset.sqlite"
    dfs2csv(folder / VISDATA_FOLDER, dst, verbose=verbose)
    # dst = str(folder / VISDATA_FOLDER / "dataset.sqlite")
    # files = [str(_.absolute()) for _ in (folder / VISDATA_FOLDER).glob("*.csv")]
    # print(f"Running: {' '.join(['csvs-to-sqlite'] + files + [dst])}")
    # subprocess.run(["csvs-to-sqlite"] + files + [dst])
    print(f"Running: {' '.join(['datasette', str(dst)])}")
    subprocess.run(["datasette", str(dst)])


if __name__ == "__main__":
    main()
