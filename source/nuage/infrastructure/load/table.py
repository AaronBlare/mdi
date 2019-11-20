import pandas as pd
import os.path

def load_table_dict_xlsx(fn):
    if os.path.isfile(fn):
        df = pd.read_excel(fn)
        tmp_dict = df.to_dict()
        table_dict = {}
        for key in tmp_dict:
            curr_dict = tmp_dict[key]
            table_dict[key] = list(curr_dict.values())
        return table_dict
    else:
        raise IOError(f'No such file: {fn}')