import numpy as np
import pandas as pd

def ratings_to_ord(df,col,inplace = False):
    '''
    This Function takes a dataframe and a column of that dataframe and returns and converts it to ordinal 
    df:
    col:
    inplace: 
    '''
    df[col] = df[col].fillna('Na')
    qual_ = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na": 0}
    if inplace == False:
        return df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])
    elif inplace == True:
        df[col] = df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])

def outliers(df,outlier_column ,num_sd = 4,method = 'outlier_df', operator = 'or'):
    '''
    This function takes a dataframe and returns a dictionary that identifies the outliers of each column
    inputs:
    method: (length | Outlier_df)
    operator: (any | min_2 | all)
    '''
    outlier_dict = {}
    d = []
    full_outliers = []
    for col in outlier_column:
        if (df[[col]].dtypes[0] == np.int64()) or (df[[col]].dtypes[0] == np.float64()):
            lst_ = []
            outlier_dict[col] = lst_
            mean = df[col].mean()
            sd   = df[col].std()
            outlier_bound_high = mean + sd*num_sd
            outlier_bound_low  = mean - sd*num_sd
            outliers_idx = df.index[df[col].apply(lambda x: (x < outlier_bound_low) or (x > outlier_bound_high))].tolist()        
            for i in outliers_idx:
                full_outliers.append(i)
            if method == 'length':
                outlier_dict[col] = [len(outliers_idx)]
            elif method == 'outlier_df':
                for i in outliers_idx:
                    d.append(df.iloc[i])
    if method == 'length':
        return pd.DataFrame.from_dict(outlier_dict,orient='index',columns=['Outlier_Count'])
    elif (method == 'outlier_df'):
        df_ = pd.DataFrame(d)
        if operator == 'any':
            return df_.drop_duplicates()
        elif operator == 'min_2':
            return df_[df_.duplicated()]
        elif operator == 'all':
            x = (pd.Series(full_outliers).value_counts(sort = False) == len(outlier_column.columns))
            x = pd.DataFrame(x,columns = ['and_'])
            x = x[x.and_ == True]
            return df_.merge(x,right_index = True, left_index = True,how = 'inner').drop('and_', axis = 1).drop_duplicates()
