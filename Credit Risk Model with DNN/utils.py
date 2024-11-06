import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt 
import seaborn as sns
import os, re, gc, platform


def read_file(path:str):
    ''' 
    Read a csv file and cast memory-efficient data types.
    '''

    df = pl.read_csv(path)

    for col in df.columns:
        suffix, dtype0 = col[-1], df[col].dtype
        if col in ["case_id", "WEEK_NUM", "MONTH", "target", "num_group1", "num_group2"] or dtype0 == pl.Int64:
            dtype = pl.Int32
        elif col == "date_decision" or suffix == "D":
            dtype = pl.Date
        elif suffix in ("P", "A") or dtype0 == pl.Float64:
            dtype = pl.Float32
        elif suffix == "M":
            dtype = pl.String
        else:
            continue
        df = df.with_columns(pl.col(col).cast(dtype))

    return df


def read_table(table:str, train_test:str, data_dir:str, verbose:int=False):
    ''' 
    Read all the csv files under the same table name.
    '''

    folder_path = data_dir + "/csv_files/{}".format(train_test)
    file_list = np.array(os.listdir(folder_path))
    table_filter = [table in file for file in file_list]
    file_to_read = file_list[table_filter]

    df_list = []
    for file in file_to_read:
        if verbose == True: print("Reading {}".format(file))
        path = folder_path + "/" + file
        df = read_file(path)
        df_list.append(df)

    if len(df_list) == 1:
        df = df_list[0]
    else:
        df = pl.concat(df_list, how="vertical_relaxed")

    # Drop a column if all values are null
    df = df[[s.name for s in df if not (s.null_count() == df.height)]]

    # Try to change string columns into numeric; otherwise into pl.Categorical 
    is_str = np.array(df.dtypes) == pl.String
    str_col = np.array(df.columns)[is_str]
    for col in str_col:
        try:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
        except pl.exceptions.ComputeError:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
            except pl.exceptions.ComputeError:
                df = df.with_columns(pl.col(col).cast(pl.Categorical))

    return df


def read_all_table(table_list:list, train_test:str, data_dir:str, verbose:int=False):
    ''' 
    Read and cocatenate all required tables by passing a required table list (base table must be the first one).
    Return a dictionary of tables.
    '''

    my_dict = {}
    for table in table_list:
        my_dict[table] = read_table(table, train_test, data_dir, verbose)
        if table != "base":
            my_dict[table] =  date_to_dt(my_dict[table], my_dict["base"])
        if len(my_dict[table]) == 0:
            my_dict.pop(table)
    
    return my_dict


def id_presence(df_dict:dict):
    ''' 
    Returns a record of checking whether each `case_id` present at each table.
    '''

    record = pd.DataFrame(index=df_dict["base"]["case_id"])
    for table in df_dict.keys():
        record[table] = record.index.isin(df_dict[table]["case_id"].unique())
    
    return record


def view_sample_obvs(table:pl.DataFrame, seed:int, size=5):
    ''' 
    View sample observations
    '''

    np.random.seed(seed)
    chosen_id = np.random.choice(table["case_id"].unique(), size=5)
    sample = table.filter(pl.col("case_id").is_in(chosen_id)).to_pandas()

    return sample


def join_data(df_dict:dict):
    ''' 
    Join the tables as a single dataframe

    Parameter:
        df_dict: A dictionary of Polars DataFrames, with each key-value pair representing a table.

    Return:
        A single concatenated Polars dataframe
    '''
   
    df = df_dict["base"]
    for name, df_i in df_dict.items():
        if name == "base":
            continue
        df = df.join(df_i, on="case_id", how="left", suffix= "_" + name)

    return df 


def date_to_dt(table:pl.DataFrame, base_table:pl.DataFrame):
    ''' 
    Replace the date columns with the day difference between the column and date_decision.
    '''

    df = table.clone().join(base_table[["case_id", "date_decision"]], on="case_id", how="left")
    col_list = df.columns
    col_list.remove("date_decision")

    for col in col_list:
        if df[col].dtype == pl.Date:
            df = df.with_columns(-(pl.col(col) - pl.col("date_decision")).dt.total_days().cast(pl.Int32))
    
    return df.drop("date_decision")


def missing_by_group(table:pl.DataFrame, by:str):
    ''' 
    Count the poportion of missing values in each column, grouped by a given categorical column.
    '''
    
    missing = table.select(pl.all().is_null())
    missing = missing.drop([col for col in missing.columns if missing[col].n_unique() == 1])
    missing = missing.with_columns(table[by])
    missing_count = missing.group_by(by).sum()
    missing_count = missing_count.join(missing.group_by(by).len(), on=by, how="left")

    for col in missing_count.columns[1:-1]:
        missing_count = missing_count.with_columns(pl.col(col) / pl.col("len"))

    missing_count = missing_count.sort(by)
    missing = missing_count.to_pandas().round(3)
    missing.set_index(by, inplace=True)
    missing = missing.drop("len", axis=1)
    
    return missing



def missingness_corr(table:pl.DataFrame):
    ''' 
    Check the correlation between the presence of missing value and the target value.
    '''
    
    df = table.clone()
    missing_prop, rho = [], []

    for col in table.columns[5:]:
        isnull = df[col].is_null()
        missing_prop.append(df[col].is_null().sum() / len(df))
        rho.append(np.corrcoef(df["target"], isnull)[1,0])
    
    output = pd.DataFrame({"missing_prop":missing_prop, "rho":rho}, index=table.columns[5:])
    output.sort_values(by="missing_prop", inplace=True, ascending=False)

    return output


def find_allequal_col(df:pl.DataFrame):
    ''' 
    Find columns if all observations have the same value
    '''
    
    df_copy = df.clone()
    col_list = []

    for col in df_copy.columns:
        data = df_copy[col].drop_nulls()
        if data.n_unique() == 1:
            col_list.append(col)

    return col_list


def corr_wt_target(df:pl.DataFrame):
    ''' 
    Compute the correlations between the target and numerical features (missing values omitted).
    '''

    df_copy = df.clone()
    rho = {}
    
    for col in df_copy.columns[5:]:
        if df_copy[col].dtype != pl.String:
            data = df_copy[["target", col]].drop_nulls()
            y = data["target"]
            x = data[col] if data[col].dtype != pl.Boolean else data[col].cast(pl.Int8)
            rho[col] = [np.corrcoef(x, y)[0,1]]
    
    output = pd.DataFrame(rho).T.sort_values(0)
    output.columns = ["rho"]

    return output


def summarize(table:pl.DataFrame, feat_def:pl.DataFrame):
    ''' 
    Analyze the table with summary statistics.
    '''

    # missing values
    cols = table.columns
    df = table.select(pl.all().is_null()).sum().to_pandas().T / len(table)
    df.reset_index(names="Variable", inplace=True)

    # column descriptions
    df = pd.merge(df, feat_def[feat_def["Variable"].isin(cols)], how="left", on="Variable")
    df.columns = ["Column", "Missing_Prop", "Description"]

    # Are all the non-null observations identical?
    dup_col = find_allequal_col(table)
    df["All_Obvs_Equal"] = df["Column"].isin(dup_col)

    # value count of categorical features
    cat_feat = [col for col in table.columns if table[col].dtype in (pl.Categorical, pl.Boolean)]
    if len(cat_feat) > 0:
        val_count = table[cat_feat].select(pl.all().n_unique()).to_pandas().T
        val_count.columns = ["N_Unique"]
        val_count.reset_index(names="Column", inplace=True)
        df = pd.merge(df, val_count, on="Column", how="left")

    # summary statistics of numerical features
    stats = table.to_pandas().describe().T
    stats.reset_index(names="Column", inplace=True)
    df = pd.merge(df, stats, on="Column", how="left")
    df = df.sort_values("Missing_Prop")

    return df.round(4).reset_index()
    

def null_plot(ax, table:pl.DataFrame, n_col:int=30):
    ''' 
    Visualize the proportion of  missing values
    '''

    df = table.select(pl.all().is_null()).sum().to_pandas().T / len(table)
    df.columns = ["Missing_Prop"]
    df.reset_index(names="Column", inplace=True)
    df.sort_values(by="Missing_Prop", ascending=False, ignore_index=True, inplace=True)
    if n_col < len(df):
        df = df.iloc[:n_col]

    sns.set_theme()
    ax.bar(df["Column"], df["Missing_Prop"]*100)
    ax.set_title("Proportion of Missing Value (Sorted)")
    ax.set_ylabel("Proportion(%)")
    ax.set_xlabel("Column")
    ax.tick_params(axis='x', labelrotation=90)

    return ax


def aggregate(table:pl.DataFrame, count_obvs:bool=False):
    ''' 
    Perform aggregation on a table with depth = 1.
    '''

    df = table.clone()

    # Classify the columns
    col_date, col_num, col_cat = [], [], []
    for col in df.columns[1:]:
        if col in ("num_group1", "num_group2"):
            df = df.drop(col)
        elif col[-1] == "D":
            col_date.append(col)
        elif df[col].dtype.is_numeric() or df[col].dtype == pl.Boolean:
            col_num.append(col)
        else:
            col_cat.append(col)
    
    # Aggregation
    # out = df.select("case_id",
    #              *[pl.col(col).mean().over('case_id').alias(col+"_mean") for col in col_num],
    #              *[pl.col(col).mean().over('case_id').alias(col+"_std") for col in col_num],
    #              *[pl.col(col).max().over('case_id').alias(col+"_max") for col in col_date],
    #              *[pl.col(col).min().over('case_id').alias(col+"_min") for col in col_date],
    #              *[pl.col(col).mode().over('case_id').alias(col+"_mode") for col in col_cat],
    #              *[pl.col(col).first().over('case_id').alias(col+"_first") for col in col_cat],
    #              *[pl.col(col).last().over('case_id').alias(col+"_last") for col in col_cat])
            
    # out = out.group_by("case_id").first().sort("case_id")
    
    expr_mean = [pl.col(col).mean().alias(col+"_mean") for col in col_num]
    expr_std = [pl.col(col).std().alias(col+"_std") for col in col_num]

    expr_max = [pl.col(col).max().alias(col+"_max") for col in col_date]
    expr_min = [pl.col(col).min().alias(col+"_min") for col in col_date]

    expr_first = [pl.col(col).first().alias(col+"_first") for col in col_cat]
    expr_last = [pl.col(col).last().alias(col+"_last") for col in col_cat]

    aggregater = expr_mean + expr_std + expr_max + expr_min + expr_first+ expr_last

    out = df.group_by("case_id").agg(aggregater)

    if count_obvs:
        case_count = df["case_id"].value_counts()
        out = out.join(case_count, on="case_id", how="left")
        out = out.rename({"count": "obvs_count"})
    
    return out


def observation_by_group(table:pl.DataFrame):
    ''' 
    Count the number of unique observations per case_id.
    '''
    
    df_obs = table.clone()
    for col in df_obs.columns[1:]:
        df_obs = df_obs.with_columns((~pl.col(col).is_null()).cast(pl.Int8))
    out = df_obs.group_by('case_id').sum().sort(by='case_id')

    return out


def fill_missing_value(table:pl.DataFrame, is_merge:bool=False):
    ''' 
    Use representation scheme to fill in missing values.
    is_merge == True means the missing values are caused by mergeing tables
    '''
    
    df = table.clone()

    for col in df.columns:
        if df[col].dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Int8))
    
    is_num = np.isin(df.dtypes, [pl.Int8, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.UInt32])
    is_cat = ~np.isin(df.dtypes, [pl.Int8, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.UInt32])
    is_cat[0] = True  # include case_id

    cols = np.array(df.columns)
    df_num, df_cat = df[cols[is_num]], df[cols[is_cat]]
    if is_merge == False:
        df_num = df_num.fill_null(-1)
        df_cat = df_cat.fill_null("NaN")
    else:
        df_num = df_num.fill_null(-2)
        df_cat = df_cat.fill_null("NaN_byMerge")

    out = df_num.join(df_cat, on="case_id", how="left")

    return out
