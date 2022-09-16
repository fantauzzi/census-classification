import pytest
from pathlib import Path
import pandas as pd
from census_preprocess import str_columns, pre_process

raw_datafile = '../data/census.csv'
test_cleaned_datafile = '../data/test_census_cleaned.csv'


def test_str_columns():
    source = {'A': 'a1', 'B': 'b1', 'C': 0, 'D': 'd1', 'E': -1}
    df = pd.DataFrame(source, index=[0])
    res = str_columns(df)
    assert len(res) == 3
    assert 'A' in res
    assert 'B' in res
    assert 'D' in res


def test_pre_process():
    if Path(test_cleaned_datafile).exists():
        Path(test_cleaned_datafile).unlink()
    pre_process(raw_datafile, test_cleaned_datafile)
    assert Path(test_cleaned_datafile).exists()
    Path(test_cleaned_datafile).unlink()
