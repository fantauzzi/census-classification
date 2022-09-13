import pandas as pd


def main():
    raw_datafile = '../data/census.csv'
    cleaned_datafile = '../data/census_cleaned.csv'
    # categorical_idx = (1, 3, 4, 5, 6, 7, 8, 9, 13)

    df = pd.read_csv(raw_datafile)
    initial_shape = df.shape

    # Remove any leading/trailing blank from column names
    mapper = {key: key.strip() for key in df.columns}
    df.rename(mapper=mapper, axis='columns', inplace=True)

    # Get the list of columns containing strings (of type object)
    str_column_names = [name for name in df.columns if pd.api.types.is_string_dtype(df[name])]
    # Remove any leading/trailing blank from strings in the dataframe, and replace unknwon values ('?') with a None
    for col_name in str_column_names:
        df[col_name] = df[col_name].apply(lambda x: x.strip())
        df[col_name].replace('?', None, inplace=True)

    # Replace the two categories in the `salary` column with 0 and 1
    df['salary'] = df['salary'].map({'>50K': 1, '<=50K': 0})

    # Dataframe should have unchanged shape after its cleaning
    assert df.shape == initial_shape

    # Save the cleaned dataset
    df.to_csv(cleaned_datafile, index=False, na_rep='null')


if __name__ == '__main__':
    main()
