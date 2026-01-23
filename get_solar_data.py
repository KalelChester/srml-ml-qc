import pandas as pd
import numpy as np
import sys
import time

def subroutine_main_automated_qc(comprehensive_location):
    '''
    Load the comprehensive format files
    Perform the reasonableness test
    Perform the comparison test
    Save the comprehensive format file once you are done. 
    '''

    # Load the comprehensive file
    df = pd.read_csv(comprehensive_location, header = None, dtype=str)

    # The EUO PIR columns got mixed up. 
    # Change the order of them
    if ('EUO' in comprehensive_location) & ('2024' in comprehensive_location):
        print(df.iloc[0,33:37])
        
        if ('PIR' in df.iloc[0, 33]) & ('PIR' in df.iloc[0, 34]) & ('PIR' in df.iloc[0, 35]) & ('PIR' in df.iloc[0, 36]):

            df.iloc[0:4, 33] = ['PIR_NET_I', '7008', 'PIR(30923F3)_NET_I', 'PIR_Net_I']
            df.iloc[43, 33] = 'PIR_NET_I'
            
            df.iloc[0:4, 34] = ['Flag_PIR_NET_I', '-', '-', '-']
            df.iloc[43, 34] = 'Flag_PIR_NET_I'
            
            df.iloc[0:4, 35] = ['PIR_DW_I', '7009', 'PIR(30923F3)_DW_I', 'PIR_DW_I']
            df.iloc[43, 35] = 'PIR_DW_I'
            
            df.iloc[0:4, 36] = ['Flag_PIR_DW_I', '-', '-', '-']
            df.iloc[43, 36] = 'Flag_PIR_DW_I'

        else:
            sys.exit('line 122, SRML_AutomatedQC')

    # Get df header row info
    # You will put this back on top of the df before export
    df_header = df.iloc[:44]

    # Get df data
    df = df[44:]

    # Assign df column headers (useful in using column names)
    # These values will be searched for useful information 
    df.columns = df_header.iloc[0,:] + '_' + df_header.iloc[2,:] + '_' + df_header.iloc[8,:]

    # Make sure that the CF file is the correct length. Must be divisible by 1440
    # The df is only the data, the header rows have been removed.
    if (len(df))/1440 % 1 !=0:
        print('The comprehensive format file is not the correct length')
        print('JOSH You should probably delete the CF here and start over. Something went wrong.')
        time.sleep(5)
        sys.exit('line 132')

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