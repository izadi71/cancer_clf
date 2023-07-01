import time
from functools import wraps

import pandas as pd


def summary_statistics_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary statistics for numerical features in dataframe for the input dataframe
    """
    feature_info = pd.DataFrame({
        'feature': df.columns,
        'dtype': df.dtypes,
        'num_unique': df.nunique(),
        'num_Nan': df.isna().sum(),
        'count': df.count()
    })

    feature_describe = df.describe().T.drop(columns='count').reset_index().rename(columns={'index': 'feature'})

    summary_statistics_result = pd.merge(feature_info, feature_describe, on='feature')

    return summary_statistics_result



def summary_statistics_obj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary statistics for non-numerical features in dataframe for the input dataframe
    """
    feature_info = pd.DataFrame({
        'feature': df.columns,
        'dtype': df.dtypes,
        'num_unique': df.nunique(),
        'num_nan': df.isna().sum(),
    })

    non_numeric_features = df.select_dtypes(include=['object', 'bool']).columns
    feature_describe_non_numeric = df[non_numeric_features].describe().T.reset_index().rename(columns={'index': 'feature'})

    summary_statistics_result = pd.merge(feature_info, feature_describe_non_numeric, on='feature', how='right',)

    return summary_statistics_result


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper