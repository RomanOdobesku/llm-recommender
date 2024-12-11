"""
Module with functional to concat csvs
"""

import os
import pandas as pd


def concat_csvs_in_dir(
    dir_path: str,
    output_path: str,
) -> None:
    """
    Function to concat all csvs in directory
    Args:
        dir_path (str): path to dirictory with csvs
        output_path (str): path to output file
    Returns:
        (None):
    """

    df = pd.DataFrame()

    directory = os.fsencode(dir_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df = pd.concat(
                [
                    df,
                    pd.read_csv(
                        dir_path + filename,
                        sep=";",
                    ),
                ],
                axis=0,
            )
        else:
            continue

    df.to_csv(
        output_path,
        sep=";",
        index=False,
        header=True,
        encoding="utf-8",
    )
