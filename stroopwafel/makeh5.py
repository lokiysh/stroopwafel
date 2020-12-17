import numpy as np
import h5py as h5
from astropy.table import Table
import os, sys
import pandas as pd

def hdf5(folder_path, num_systems, num_per_core, batch_nr=None, batch=None, postproc=False, weights=None):
    """
        Define the name and groups of the h5 file and call the functions to append the data from the BSE files

        Args:
            folder_path    --> [string]        Folder in which data is stored
            batch          --> [string]        Batch number of stored data

        Returns:
            sp             --> [object]        Pandas dataframe with information about the system parameters
            dco            --> [object]        Pandas datafram with information about the double compact objects
    """
    
    # define the files you want to save to hdf5 
    BSE_files = ['System_Parameters', 'Common_Envelopes', 'Double_Compact_Objects', 'RLOF', 'Supernovae']
    bse_name = ['SystemParameters', 'CommonEnvelopes', 'DoubleCompactObjects', 'RLOF', 'Supernovae']
    bse_names = {}
    h5name = 'COMPAS_output.h5'
    folder = folder_path + str(batch)
    File = h5.File(folder_path + h5name, 'a') 
    sp, dco = [], []
    for index, group in enumerate(bse_name):
        if postproc == False:
            # check if the files exist and then read them
            df_name = group
            file_name = folder + '/BSE_' + BSE_files[index] + '.csv'
            if not os.path.isfile(file_name):
                continue
            bse_names[df_name] = pd.read_csv(file_name, low_memory=False)
            
            # at the first batch, we must first create the groups and datasets of the hdf5 file
            if batch == 'batch_0':
                File.create_group(group)
                systems = createHdf5(bse_names[df_name], File[group], batch)
            else:
                # sometimes when running GenAIS an extra batch will be created at the end which we don't want
                if int(batch_nr) >= int(num_systems/num_per_core):
                    continue
                systems = appendToHdf5(bse_names[df_name], File[group], batch)
            
            # solely to read in systems when looking for interesting systems in the interface file
            if group == 'SystemParameters':
                sp = systems
            elif group == 'DoubleCompactObjects':
                dco = systems

        else:
            allseeds = File['SystemParameters']['SEED'][()]
            try:
                addWeights(File[group], allseeds, weights)
            except:
                continue
    
    # always close your file!
    File.close()
    return sp, dco

def createHdf5(systems, group, batch):
    """
        Create the datasets of each group for the hdf5 file and append the data of the first batch 

        Args:
            systems       --> [object]          Pandas dataframe that contains data from the BSE files 
            group         --> [string]          Name of the current group (one of the BSE files)

        Returns:
            systems       --> [object]          Adapted dataframe with BSE information
    """

    systems = systems.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    types = systems.columns.str.strip()
    dtypes = []
     
    # save the data types of each column 
    for iType, dataType in enumerate(types):
        if dataType[0] == 'I':
            dtypes.append(np.int64)
        elif dataType[0] == 'F':
            dtypes.append(np.float64)
        elif dataType[0] == 'B':
            dtypes.append(bool)
        elif dataType[0] == 'S':
            dtypes.append(h5.special_dtype(vlen=str))
            # I think this only works with upgraded version of h5py
            #dtypes.append(h5.string_dtype(encoding='utf-8'))
        else:
            raise ValueError("Unrecognised datatype dataType=%s - for column %s in file%s "\
                            %(dataType, iType, group))
     
    # save the units and convert the dataframe to the proper headers and data
    units = np.array(systems.iloc[0])
    systems.columns = systems.iloc[1]
    systems = systems[2:]
    headers = np.array(systems.columns)

    # add the data and units to the hdf5 data sets
    for header, dtype, unit in zip(headers, dtypes, units):
        # BSE files use 0 and 1 for boolean. To convert to real boolean for hdf5 file (True or False) we must first convert to integer (otherwise it will always return True)
        if dtype == bool:
            systems[header] = systems[header].astype(np.int64)
        data = np.array(systems[header], dtype=dtype)
        dset = group.create_dataset(header, dtype=dtype, data=data, maxshape=(None,))
        dset.attrs['units'] = unit
        if dtype == bool:
            dtype = np.int64
        systems[header] = systems[header].astype(dtype, errors="ignore")

    data = np.array([batch] * len(systems['SEED']), dtype=h5.special_dtype(vlen=str))
    dset = group.create_dataset('batch', dtype=h5.special_dtype(vlen=str), data=data, maxshape=(None,))
    return systems

def appendToHdf5(systems, group, batch, replace=False):
    """
        Append the data of the other batches to the data sets

        Args:
            systems       --> [object]          Pandas dataframe that contains data from the BSE files 
            group         --> [string]          Name of the current group (one of the BSE files)

        Returns:
            systems       --> [object]          Adapted dataframe with BSE information
    """
    
    systems = systems.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    types = systems.columns.str.strip()
    dtypes = []

    for iType, dataType in enumerate(types):
        if dataType[0] == 'I':
            dtypes.append(np.int64)
        elif dataType[0] == 'F':
            dtypes.append(np.float64)
        elif dataType[0] == 'B':
            dtypes.append(bool)
        elif dataType[0] == 'S':
            dtypes.append(h5.special_dtype(vlen=str))
        else:
            raise ValueError("Unrecognised datatype dataType=%s - for column %s in file%s "\
                            %(dataType, iType, group))

    systems.columns = systems.iloc[1]
    systems = systems[2:]
    headers = np.array(systems.columns)
    
    # because we run multiple batches in parallel, extra batches can be produced at the end of the
    # exploratory phase that have to be deleted (otherwise we will have too many systems in the end)
    if batch in set(group['batch'][()]):
        index = np.where(group['batch'][()] == batch)[0][0]
        replace = True

    for header, dtype in zip(headers, dtypes):
        if dtype == bool:
            systems[header] = systems[header].astype(np.int64)
        data = np.array(systems[header], dtype=dtype)
        # we have to resize the data set in order to append the new data
        if replace == True:
            group[header].resize((index + systems[header].shape[0]), axis=0)
        else:    
            group[header].resize((group[header].shape[0] + systems[header].shape[0]), axis=0)
        group[header][-systems.shape[0]:] = data
        if dtype == bool:
            dtype = np.int64
        systems[header] = systems[header].astype(dtype, errors="ignore")

    data = np.array([batch] * len(systems['SEED']), dtype=h5.special_dtype(vlen=str))
    if replace == True:
        group['batch'].resize((index + data.shape[0]), axis=0)
    else:
        group['batch'].resize((group['batch'].shape[0] + data.shape[0]), axis=0)
    group['batch'][-data.shape[0]:] = data
    
    return systems

def addWeights(group, allseeds, weights):
    """
        Append the STROOPWAFEL mixture weights to the hdf5 groups

        Args:
            group         --> [string]          Name of the current group (one of the BSE files)
            weights       --> [list of floats]  STROOPWAFEL mixture weight of each systems
    """

    seeds = group['SEED'][()]
    seedmask = np.in1d(allseeds, seeds)
    masked_weights = np.array(weights)[seedmask]
    group.create_dataset('mixture_weight', data=masked_weights)
    
    return
