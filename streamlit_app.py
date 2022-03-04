import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar

import pandas as pd
import numpy as np
import h5py
from scipy import sparse

import matplotlib.pyplot as plt
import seaborn as sns

from footer import *

footer()

def load_sparse_matrices(ids, fh):
    """
    load a list of csr matrix in HDF5 (based on h5py syntax)

    Parameters
    ----------
    ids: List(int)
        list of indices
    fh: str
        handle to source HDF5 group 
    """
    data = []
    for idx in ids:
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(fh[attribute][idx])
        # construct sparse matrix
        data.append(sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3]))
    return data


def unbinarize_DNA(dna_bin, img=False, quick=False, custom_dict=None):
    if len(dna_bin.shape) == 1:
        dna_bin= dna_bin.reshape(-1,1)
    if custom_dict is None:
        img_dict = {1: {1: {1: {1: 'N', 0: 'H'}, 0: {1: 'D', 0: 'W'}},
                        0: {1: {1: 'V', 0: 'M'}, 0: {1: 'R', 0: 'A'}}},
                    0: {1: {1: {1: 'B', 0: 'Y'}, 0: {1: 'K', 0: 'T'}},
                        0: {1: {1: 'S', 0: 'C'}, 0: {1: 'G', 0: 'N'}}}}
        img_dict_simple = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        int_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'R', 5: 'Y',
                    6: 'S', 7: 'W', 8: 'K', 9: 'M', 10: 'B', 11: 'D',
                    12: 'H', 13: 'V', 14: 'N'}
    else:
        img_dict = img_dict_simple = int_dict = custom_dict
        
    if img:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        dna_bin_stripped = dna_bin[:, :4]
        if quick:
            for idx, nt_img in enumerate(dna_bin_stripped):
                dna_str[idx] = img_dict_simple[nt_img.argmax()]
        else:
            for idx, nt_img in enumerate(dna_bin_stripped):
                value = img_dict[nt_img[0]][nt_img[1]][nt_img[2]][nt_img[3]]
                dna_str[idx] = value
    else:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        for idx, nt_img in enumerate(dna_bin):
            dna_str[idx] = int_dict[nt_img[0]]

    return dna_str.tostring().decode('utf-8')

def get_prot_info(seq, out, cutoff = 0.1, single=False):
    tiss = []
    len_prots = []
    prot_bounds = []
    prots = []
    prot_count = np.sum(out > cutoff)
    if single:
        prot_count = min(prot_count, 1)
    for i in range(prot_count):
        tis = np.argsort(out)[-(i+1)]
        tiss.append(tis)
        prots.append(construct_prot(seq[tis:]))
        len_prots.append(len(prots[-1]))
        prot_bounds.append([tis, tis + len_prots[-1]*3])
        
    return np.array(tiss), np.array(len_prots), np.array(prot_bounds), np.array(prots)

def construct_prot(seq):
    stop_cds = ['TAG', 'TGA', 'TAA']
    sh_cds = np.array([seq[n:n+3] for n in range(0,len(seq)-2, 3)])
    stop_site_pos = np.where(pd.Series(sh_cds).isin(stop_cds))[0]
    if len(stop_site_pos) > 0:
        stop_site = stop_site_pos[0]
        cdn_seq = sh_cds[:stop_site]
    else:
        cdn_seq = sh_cds
    
    string = ''
    for cdn in cdn_seq:
        string += cdn_prot_dict[cdn]
        
    return string

cdn_prot_dict = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                 
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

st.write("# TIS Transformer prediction browser")
add_selectbox = st.sidebar.selectbox(
    "Species",
    ("Homo sapiens",)
)
add_selectbox = st.sidebar.selectbox(
    "Assembly",
    ("GRCh38",)
)
add_selectbox = st.sidebar.selectbox(
    "Version",
    ("v104",)
)


#preds = np.load('pred_sparse.npy', allow_pickle=True)

f = h5py.File('preds_TIS.h5py', mode='r')
pred_ids = np.array(f['ids']).astype(str)
ids = pd.read_hdf('preds_TIS.h5py', 'search')

gene_ids = list(ids['gene_id'].unique()) + list(ids['gene_name'].unique())
#gene_ids = list(ids['gene_name'].unique())
#gene_ids = list(ids['gene_id'].unique())
tr_ids = list(ids['transcript_id'])

gene_id = st_tags(
    label='Select gene',
    text='gene ID/name',
    value=None,
    suggestions=gene_ids,
    maxtags = 1,
    key='ge_select')

if gene_id is not None and len(gene_id)>0:
    tr_ids = list(ids.loc[np.logical_or(gene_id[0].upper()==ids['gene_id'],
                                        gene_id[0].upper()==ids['gene_name'].str.upper()),'transcript_id'])
    
    tr_id = st.selectbox('Select transcript', tr_ids)

else:
    tr_id_dict = st_tags(
        label='Select transcript',
        text='transcript ID/name',
        value=None,
        suggestions=tr_ids,
        maxtags = 1,
        key='tr_select')
    try:
        tr_id = tr_id_dict[0]
    except:
        tr_id = None

if tr_id:
    
    st.write("### Model output")
    mask = tr_id == pred_ids
    idx = np.where(mask)[0][0]
    if np.sum(mask) == 1:
        x = f['seqs'][mask][0]
        out = load_sparse_matrices([idx], f)[0].toarray()[0]
        tiss, len_prots, prot_bounds, _ = get_prot_info(unbinarize_DNA(x), out, cutoff=0.1)
        fig, ax = plt.subplots(1,1, figsize=(8*1.2,2*1.2))
        ax.scatter(np.arange(len(out)), out)
        colors = ['g','brown','b','y','o']
        [ymin, ymax] =  ax.get_ylim()
        ymax = ymax + 0.1
        labels = ['bounds CDS', 'bounds CDS', 'bounds CDS', 'bounds CDS']
        for color, bounds, label in zip(colors, prot_bounds, labels):
            ax.vlines(bounds, ymin, ymax, color=color, label=label+ f': {int((bounds[1]-bounds[0])/3)} aa', linestyle='--', alpha=0.7)
        ax.set_xlim(0, len(out))
        ax.grid(alpha=0.35)
        sns.despine(left=True, bottom=True, right=True)
        ax.set_xlabel('transcript position', fontsize=13)
        ax.set_ylabel('model output', fontsize=13)
        ax.set_title(f'{tr_id}', fontsize=14)
        fig.legend(loc=1)

        st.pyplot(fig)
        
    st.write("### Sites of interest")
    
    
    st.dataframe(pd.read_hdf('preds_TIS.h5py', 'top', where=np.where(np.array(f['top/tr_ids']) == tr_id.encode())[0]))
    
        
#st.download_button('Download outputs', out)
        
