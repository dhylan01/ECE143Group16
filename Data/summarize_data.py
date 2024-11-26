from os import listdir
from os.path import isfile, join
import pandas as pd


def get_sorted_files_from_data_dir():

    onlyfiles = [f for f in listdir() if (isfile(join('', f)) and  join('', f).count('.py') <= 0)]
    onlyfiles.sort()
    return onlyfiles
def merge_gun_data():
    print(get_sorted_files_from_data_dir())
    df = pd.concat(map(pd.read_csv,get_sorted_files_from_data_dir()),ignore_index=True)
    df.to_csv('MergedGunData.csv',index=False)


if __name__ == '__main__':
    merge_gun_data()