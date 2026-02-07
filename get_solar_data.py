import pandas as pd
import numpy as np
import sys
import time
import warnings

def subroutine_main_automated_qc(comprehensive_location):
    '''
    Load the comprehensive format files
    Perform the reasonableness test
    Perform the comparison test
    Save the comprehensive format file once you are done. 
    '''

    # Load the comprehensive file
    # The metadata section (rows 0-43) has inconsistent column counts
    # Read with error_bad_lines to handle this gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_full = pd.read_csv(comprehensive_location, header=None, dtype=str, 
                              on_bad_lines='skip', engine='c')
    df_header = df_full.iloc[:44].copy()
    df = df_full.iloc[44:].copy()  # Make a copy to avoid SettingWithCopyWarning

    # The EUO PIR columns got mixed up. 
    # Change the order of them
    if ('EUO' in comprehensive_location) & ('2024' in comprehensive_location):
        print(df_header.iloc[0,33:37])
        
        if ('PIR' in df_header.iloc[0, 33]) & ('PIR' in df_header.iloc[0, 34]) & ('PIR' in df_header.iloc[0, 35]) & ('PIR' in df_header.iloc[0, 36]):

            df_header.iloc[0:4, 33] = ['PIR_NET_I', '7008', 'PIR(30923F3)_NET_I', 'PIR_Net_I']
            df_header.iloc[43, 33] = 'PIR_NET_I'
            
            df_header.iloc[0:4, 34] = ['Flag_PIR_NET_I', '-', '-', '-']
            df_header.iloc[43, 34] = 'Flag_PIR_NET_I'
            
            df_header.iloc[0:4, 35] = ['PIR_DW_I', '7009', 'PIR(30923F3)_DW_I', 'PIR_DW_I']
            df_header.iloc[43, 35] = 'PIR_DW_I'
            
            df_header.iloc[0:4, 36] = ['Flag_PIR_DW_I', '-', '-', '-']
            df_header.iloc[43, 36] = 'Flag_PIR_DW_I'

        else:
            sys.exit('line 122, SRML_AutomatedQC')

    # Check for column mismatch and pad header if needed BEFORE assigning columns
    if df.shape[1] > df_header.shape[1]:
        print(f"Adjusting header width: {df_header.shape[1]} -> {df.shape[1]}")
        # Add missing columns to header (filled with empty strings)
        for i in range(df_header.shape[1], df.shape[1]):
            df_header[i] = ''

    # Assign df column headers (useful in using column names)
    # These values will be searched for useful information
    # Build column names safely by padding all Series to the same length
    col_0 = df_header.iloc[0,:].astype(str).fillna('')
    col_2 = df_header.iloc[2,:].astype(str).fillna('')
    col_8 = df_header.iloc[8,:].astype(str).fillna('')
    
    # Ensure all series have the same length
    max_len = df.shape[1]
    col_0 = col_0.reindex(range(max_len), fill_value='')
    col_2 = col_2.reindex(range(max_len), fill_value='')
    col_8 = col_8.reindex(range(max_len), fill_value='')
    
    df.columns = col_0 + '_' + col_2 + '_' + col_8

    # Get the SZA of the data           
    sza = np.array(df.iloc[:,3], dtype = float)
    # print(sza[0:5])
    
    # Compute the Cos(SZA)
    # Don't let the cos go negative
    cos_sza = np.cos(sza.clip(min=0.00001) * np.pi / 180)
    
    # Get the ETR irradiance
    etr = np.array(df.iloc[:,5], dtype = float)

    # Compute the ETRN irradiance
    # DrHI = DNI * Cos(SZA) --> DNI = DrHI / Cos(SZA)
    etrn = np.array(etr / cos_sza)


    # Change the flags from '1' to '11'
    # Change the flags from '2' to 12'
    index_column = 7
    for i_column in df.columns[7:-1:2]: 
        mask_1 = df.iloc[:, index_column + 1] == '1'
        mask_2 = df.iloc[:, index_column + 1] == '2'

        df.iloc[mask_1, index_column + 1] = '11'
        df.iloc[mask_2, index_column + 1] = '12'
        
        index_column = index_column + 2

    # Put the header and data back together in one array
    df = np.concatenate((np.array(df_header), np.array(df)), axis=0)

    
    return df, df_header