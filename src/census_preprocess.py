from os import getcwd
import logging
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def str_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the list of column names from a given Pandas dataframe where the columns are of type object (i.e.
    they contain strings).
    :param df: The given dataframe.
    :type df: pandas.DataFrame
    :return: The requested list of column names.
    :rtype: list[str]
    """
    str_column_names = [name for name in df.columns if pd.api.types.is_string_dtype(df[name])]
    return str_column_names


def pre_process(raw_datafile: str, cleaned_datafile: str):
    """
    Clean-up the census dataset, and save the result in a file.
    Clean-up does the following:
     - remove any trailing space from the column names (in the csv file header)
     - for every column that doesn't contain numeric data, remove any trailing space from its values
     - the `salary` column is turned to numerical, replacing its values `>50K` and `<=50K` with 1 and 0 respectively
     - every `?` value (in non-numerical columns) is replaced with `null`
    :param raw_datafile: the file containing the census dataset to be cleaned-up
    :type raw_datafile: string
    :param cleaned_datafile: the file that will contain the result; if such a file exists already, it will be overwritten
    :type cleaned_datafile: string
    """

    # Load it
    df = pd.read_csv(raw_datafile)
    initial_shape = df.shape

    # Remove any leading/trailing blank from column names
    mapper = {key: key.lstrip() for key in df.columns}
    df.rename(mapper=mapper, axis='columns', inplace=True)

    # Get the list of columns containing strings (of type object)
    str_column_names = str_columns(df)
    # Remove any leading/trailing blank from strings in the dataframe, and replace unknwon values ('?') with a None
    for col_name in str_column_names:
        df[col_name] = df[col_name].apply(lambda x: x.lstrip())
        df[col_name].replace('?', None, inplace=True)

    # Replace the two categories in the `salary` column with 0 and 1
    df['salary'] = df['salary'].map({'>50K': 1, '<=50K': 0})

    # Dataframe should have unchanged shape after its cleaning
    assert df.shape == initial_shape

    # Save the cleaned dataset
    if Path(cleaned_datafile).exists():
        logging.info(f'Overwriting existing {cleaned_datafile}')
    else:
        logging.info(f'Writing {cleaned_datafile}')
    df.to_csv(cleaned_datafile, index=False, na_rep='null')


@hydra.main(version_base=None, config_path=".", config_name="params.yaml")
def main(params: DictConfig) -> None:
    logging.info(f'Working directory is {getcwd()}')
    raw_datafile = params['raw_data']
    cleaned_datafile = params['cleaned_data']
    pre_process(raw_datafile, cleaned_datafile)


if __name__ == '__main__':
    main()
