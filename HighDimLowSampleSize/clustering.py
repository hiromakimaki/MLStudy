"""
Summary:
    (TODO: Write this.)

Usage:
    (TODO:  Write this.)

Requirements:
    numpy, pandas, matplotlib

Reference:
    chapter 4 of "高次元の統計学"(https://www.kyoritsu-pub.co.jp/bookdetail/9784320112636)

DataSource:
    Paper:
        "Classification of human lung carcinomas by mRNA expression profiling reveals distinct adenocarcinoma subclasses"
            https://www.pnas.org/content/98/24/13790
    Download link:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC61120/bin/pnas_191502998_DatasetA_12600gene.xls
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC61120/bin/pnas_191502998_DatasetA_3312genesetdescription_sd50.xls
    Memo:
        - squamous cell lung carcinomas (n = 21)
            "SQ-..."
        - pulmonary carcinoids (n = 20)
            "COID-..."
        - normal lung (n = 17)
            "NL-..."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    df_gene = pd.read_excel('pnas_191502998_DatasetA_12600gene.xls')
    df_desc = pd.read_excel('pnas_191502998_DatasetA_3312genesetdescription_sd50.xls', header=None)
    key_columns = ['probe set', 'gene']
    df_desc.columns = key_columns
    df = pd.merge(df_desc, df_gene, how='inner', on=key_columns)
    df = df.filter(regex='^(SQ|COID|NL)-')
    print('The shape of data frame: ', df.shape) # df.shape is (3312, 58)
    return df


def main():
    df = load_data()


if __name__ == '__main__':
    main()