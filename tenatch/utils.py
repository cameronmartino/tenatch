import numpy as np
import pandas as pd
from deicode.preprocessing import rclr

def reshape_tensor(table,mapping,timecol,IDcol):
    """ 
    Restructure dataframe into tensor 
    by metadata IDs and timepoints
    """
    # table by timepoint
    mapping_time = {k:df for k,df in mapping.groupby(timecol)}
    # create tensor
    tensor_stack = []
    # wort in numerical order
    for timepoint in sorted(mapping_time.keys()):
        # get table timepoint
        table_tmp,meta_tmp = match(table,mapping_time[timepoint])
        # fix ID cols
        meta_tmp.set_index(IDcol,inplace=True,drop=True)
        # sort so all ID match
        table_tmp = table_tmp.T.sort_index().T
        # check to make sure id's are unique to each time
        if len(meta_tmp.index.get_duplicates()):
            idrep = [str(idrep) for idrep in meta_tmp.index.get_duplicates()]
            idrep = ', '.join(idrep)
            raise ValueError('At timepoint '+str(timepoint)+
                             'The ids '+idrep+' are repeated.'
                            ' Please provide unique IDs to each time.')
        # index match
        table_tmp.index = meta_tmp.index
        table_tmp.sort_index(inplace=True)
        meta_tmp.sort_index(inplace=True)
        # update mapping time
        mapping_time[timepoint] = meta_tmp
        tensor_stack.append(table_tmp)
    # return both tensor and time_metadata dict, table for ref.
    return np.dstack(tensor_stack).T,mapping_time,table_tmp

def tensor_rclr(T):
    # flatten, transform, and reshape 
    T_rclr = np.concatenate([T[i,:,:].T 
                             for i in range(T.shape[0])],axis=0)
    T_rclr = rclr().fit_transform(T_rclr)
    T_rclr = np.dstack([T_rclr[(i-1)*T.shape[-1]:(i)*T.shape[-1]] 
                        for i in range(1,T.shape[0]+1)])
    T_rclr[np.isnan(T_rclr)] = 0 
    return T_rclr


def table_to_tensor(table,mapping,IDcol,timecol,
                    filter_timepoints=True,filter_samples=False,
                    sample_replace=False,min_sample_count = 0,
                    min_feature_count = 0):
    """
    filter_timepoints,bool- if true then filter out timepoints with missing samples
    filter_samples,bool- if true then filter out samples with missing timepoints
    sample_replace,bool- sample replacement (replace missing samples with t-1 sample)
    min_sample_count,min_feature_count,int- read count filter
    """
    # cannot have both
    if sum([filter_timepoints,filter_samples,sample_replace])>1:
        raise ValueError('Must choose to replace samples by (t-1),'
                         ' filter by missing samples or timepoints'
                         ' not multiple.')
    # filter cutoffs
    table = table.T[table.sum() > min_feature_count]
    table = table.T[table.sum() > min_sample_count]

    # filter setup 
    # table by timepoint
    mapping_time = {k:df for k,df in mapping.groupby(timecol)}
    # remove timepoint with missing samples
    drop = {k_:[v_ 
            for v_ in list(set(mapping[IDcol])-set(df_[IDcol]))] 
            for k_,df_ in mapping_time.items()}

    # sample-removal
    if filter_samples==True:
        mapping = mapping[~mapping[IDcol].isin([v_ 
                                                for k,v in drop.items() 
                                                for v_ in v])]
    # timepoint-removal
    elif filter_timepoints==True:
        mapping = mapping[~mapping[timecol].isin([k for k,v in drop.items() 
                                                  if len(v)!=0])]
    # sample-replacement
    elif sample_replace == True:
        TODO =0 
    else:
        raise ValueError('Must choose to replace samples by (t-1),'
                         ' filter by missing samples or timepoints.'
                         ' All of them can _not_ be False')  

    # remove zero sum features across flattened tensor b4 rclr
    T, mapping_time,table_tmp = reshape_tensor(table,mapping,timecol,IDcol)
    T_filter = np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0)
    sum_zero = [table_tmp.columns[i] for i, x in enumerate(list(T_filter.sum(axis=0))) if x == 0]
    table = table.drop(sum_zero,axis=1)
    T, mapping_time,table_tmp = reshape_tensor(table,mapping,timecol,IDcol)

    #test for zeros
    if any(~(np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0).sum(axis=1) > 0)):
        raise ValueError('Some samples sum to zero,'
                         ' consider increasing the sample'
                         ' read count cutoff')
    elif any(~(np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0).sum(axis=0) > 0)):
        raise ValueError('Some features sum to zero,'
                         ' consider increasing the feature'
                         ' read count cutoff')
    # if passed zero check
    return T,mapping_time,table_tmp.columns,table_tmp.index

def match(table, metadata):
    """ Match on dense pandas tables,
        taken from gneiss (now dep.)
        https://github.com/biocore/
        gneiss/blob/master/gneiss/util.py
    """
    subtableids = set(table.index)
    submetadataids = set(metadata.index)
    if len(subtableids) != len(table.index):
        raise ValueError("`table` has duplicate sample ids.")
    if len(submetadataids) != len(metadata.index):
        raise ValueError("`metadata` has duplicate sample ids.")

    idx = subtableids & submetadataids
    if len(idx) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))
    subtable = table.loc[idx]
    submetadata = metadata.loc[idx]
    return subtable, submetadata
