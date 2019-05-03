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

from scipy.sparse.linalg import eigsh
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


def get_top_k_eigenvalues(X, k):
    _, n = X.shape
    P = np.eye(n) - np.ones((n,n)) / n
    XP = np.dot(X, P)
    mat_dual_cov = np.dot(XP.T, XP) / (n - 1)
    eigvals, eigvecs = eigsh(mat_dual_cov, k=k)
    # Sorting by eigenvalue-descending order
    sorted_eigvals = - (np.sort(- eigvals))
    sorted_eigvecs = np.zeros(eigvecs.shape)
    before_eigval = None
    sort_completed_index = 0
    for eigv in sorted_eigvals:
        if before_eigval == eigv:
            continue
        indices = np.where(eigvals == eigv)[0]
        sorted_eigvecs[:, sort_completed_index:(sort_completed_index + len(indices))] = eigvecs[:, indices]
        sort_completed_index = sort_completed_index + len(indices)
        before_eigval = eigv
    # Noise reduction method
    denominators = n - 1 - np.arange(1, k+1)
    noise_reduced_sorted_eigvals = sorted_eigvals - (np.trace(mat_dual_cov) - sorted_eigvals.cumsum()) / denominators
    return noise_reduced_sorted_eigvals, sorted_eigvecs


def get_top_k_pca_score(df, k):
    _, n = df.shape
    _, eigvecs = get_top_k_eigenvalues(df.values, k)
    return np.sqrt(n) * eigvecs


def main():
    k = 2
    df = load_data()
    pca_score = get_top_k_pca_score(df, k)
    plt.scatter(pca_score[:, 0], pca_score[:, 1])
    plt.show()


if __name__ == '__main__':
    main()