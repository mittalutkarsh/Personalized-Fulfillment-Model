# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:21:13 2018

@author: Umittal
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np
import pandas as pd
import time
import timeit
#import simutils as ut
import random
import pdb
from datetime import datetime as dt
import matplotlib.dates as mdates
import pyodbc
from numba import jit
import sqlite3
from numba import vectorize
import os
inputfile_path = os.getcwd() # taking the dynamic path current directory path
inputfile_path = inputfile_path + "\\"
Network_model_start_time_function = time.time()

###########Scenario Modeling Starts####################################################################
#### Variable for dynamic dc capacity
Variable_DC_capacity_option =1
Variable_capacity_baclog = 0
rem_dc_cap_baclog = 0
Base_Year = 2017


#### New Promise Time Calculation Future Years Starts ##################################################

New_Promise_Time_Option = 0 # 1 -> to Switch to new promise time based on delivery for future years, 0 -> Old Promise time
Fiscal_Year = 0


#### New Promise Time Calculation Future Years Ends ####################################################






###########Scenario Modeling End####################################################################




###########DEFINE INPUT PATH####################################################################
#inputfile_path = "C:/Users/Umittal/Desktop/Network_model/"
#inputfile_path = "C:/Users/Umittal/Desktop/Network_model/Scenario modeling/Scenario_C_Rerun_DC_5/" # runnig for DC 6
#inputfile_path = "C:/Users/Ki847yg/Desktop/Network Model Inputs"
###############################################################################################


####SCENARIO CONSTANTS##################################################################

start = dt(2021, 1, 15) # YYYY-MM-DD
#end = dt(2021, 1, 20)
end = dt(2022, 2, 21)  # YYYY-MM-DD

#Q1 = 1/29 - 4/29
########################################s###############################################33


#print(ut.temp_test)

#import parcel cost from Excel file
#df_Parcel_Cost_final=pd.read_excel("C:/Users/Umittal/Desktop/Network_model/CPP_final.xlsx",sheet_name = "CPP_Final")







#  Below is be necessary to connect with SQL server
#server = 'net-model.database.windows.net'
#database = 'network-model'
#username = 'sqladmin'
#password = 'eXazC9ukJBQNP#j'
#driver= '{ODBC Driver 13 for SQL Server}'
#cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)

#df2 = pd.read_sql('select  * from [network-model].dbo.Venus_Data_Model_run',cnxn )
#cnxn.close()

# parse dates to making sure that Column2(Order day) comes as a date
df2 = pd.read_csv(inputfile_path + "Venus_data11_SFS.csv", dtype={'Year':int, 'Order_Hr': int, 'Zip' : int, 'Network Alignment': int, 'DC':int, 'Segment' : int , 'Ship Option' : int, 'Brand' : int, 'Ordered Units' : float, 'Modeled DC': float, 'Modeled Ship Option' : int, 'Modeled Order Units' : float, 'Time to Promise' : float, 'Remaining Units' : float, 'Time fully Processed' : float, 'Units_Processed1' : float, 'Units_Processed_Final' : float, 'Cost' : float,	'Total_Cost' : float,	'UPP' : float,	'Geo' : int ,	'Customer_Type' : float,	'Free_Speed_Eligible' : int,	'DC_name' : int,	'Promise_time' : int  } , parse_dates=[2] ) 
#df2 = pd.read_csv(inputfile_path + "Venus_data11.csv")
                  
df2['Modeled Order Units'].sum() 


df_dummy = pd.read_excel(inputfile_path + "df_initial_Baclog_Ath_retail.xlsx", dtype={'Year':int, 'Order_Hr': int, 'Zip' : int, 'Network Alignment': int, 'DC':int, 'Segment' : int , 'Ship Option' : int, 'Brand' : int, 'Ordered Units' : float, 'Modeled DC': float, 'Modeled Ship Option' : int, 'Modeled Order Units' : float, 'Time to Promise' : float, 'Remaining Units' : float, 'Time fully Processed' : float, 'Units_Processed1' : float, 'Units_Processed_Final' : float, 'Cost' : float,	'Total_Cost' : float,	'UPP' : float,	'Geo' : int ,	'Customer_Type' : float,	'Free_Speed_Eligible' : int,	'DC_name' : int,	'Promise_time' : int  } , parse_dates=[2] ) 

# Remove number column that comes with SQL server
df2 = df2.drop(['Unnamed: 0'], axis=1)
df2= pd.concat([df2,df_dummy],axis=0)

# only for one dc ==1

#df2['Order_day'] = pd.to_datetime(df2['Order_day'],format='%Y%M%d')




##### CODE FOR DATAFRAME MEMORY OPTIMIZATION ###########################################################################


# Optimizing the Memory in the dataframe
for dtype in ['float','int','object']:
    selected_dtype = df2.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
    
int_types = ["uint8", "int8", "int16", "int32", "int64"]
for it in int_types:
    print(np.iinfo(it))
    
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

gl_int = df2.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')

print(mem_usage(gl_int))
print(mem_usage(converted_int))

compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)

gl_float = df2.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')
print(mem_usage(gl_float))
print(mem_usage(converted_float))

compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
compare_floats.apply(pd.Series.value_counts)


df2[converted_int.columns] = converted_int
df2[converted_float.columns] = converted_float

# memory Optimized

# function to reduce memory size

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object :  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist



##### CODE FOR DATAFRAME MEMORY OPTIMIZATION END ###########################################################################







# FUTURE REMOVAL POSSIBLE
#date = df2.Order_day
#print(mem_usage(date))
#date.head()
#-----------------------------------------


##### IMPORT MODEL INPUTS FROM EXCEL ###########################################################################

# 
print(df2.head())
print('input_files')    
df_Parcel_Cost_final=pd.read_excel(inputfile_path + "CPP_final.xlsx",sheet_name = "CPP_DC")
df_Parcel_Cost_final_SFS=pd.read_excel(inputfile_path + "CPP_final.xlsx",sheet_name = "CPP_SFS")
df_Parcel_Cost_final['mdates'] = df_Parcel_Cost_final['Fis_Cal_Date'].map(mdates.date2num)
df_Parcel_Cost_final.mdates = df_Parcel_Cost_final.mdates.astype(int)


# for Geo CPP

df_Parcel_Cost_final_Geo =pd.read_excel(inputfile_path + "CPP_final.xlsx",sheet_name = "CPP_DC_GEO")
df_Parcel_Cost_final_Geo['mdates'] = df_Parcel_Cost_final_Geo['Fis_Cal_Date'].map(mdates.date2num)
df_Parcel_Cost_final_Geo.mdates = df_Parcel_Cost_final_Geo.mdates.astype(int)



#import UPP by Brand
df_UPP_final=pd.read_excel(inputfile_path + "UPP.xlsx",sheet_name = "DC_UPP")
df_UPP_final_SFS = pd.read_excel(inputfile_path + "UPP.xlsx",sheet_name = "SFS_UPP")
df_UPP_final['mdates_from'] = df_UPP_final['Fis_Cal_Date_From'].map(mdates.date2num)
df_UPP_final['mdates_to'] = df_UPP_final['Fiscal_Date_to'].map(mdates.date2num)
df_UPP_final.mdates_from = df_UPP_final.mdates_from.astype(int)
df_UPP_final.mdates_to = df_UPP_final.mdates_to.astype(int)

#Import Capacity by Network Alignment and Promise time
#dc_cap=pd.read_excel(inputfile_path + "Capacity_Jackie2_WFC.xlsx",sheet_name = "Simulation")

#Fiscal calendar for lookup
df_fis_cal = pd.read_excel(inputfile_path + "Fis calendar_Network_model.xlsx" , sheet_name = "Sheet1")


Cancelation_Rate=pd.read_excel(inputfile_path + "Cancelation_Rate_by_Brand.xlsx",sheet_name = "Sheet1")

Ship_Option = pd.read_excel(inputfile_path + "Ship_Option.xlsx",sheet_name = "Sheet1")

############IMPORT FROM EXCEL END #######################################################################

#use for checking processing time of scenario
start_time = time.time()





# Creating the Capacity data by date and events

day_hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
df_day_hours = pd.DataFrame(index=np.arange(0,24),columns=['Key_HR', 'Day_Hours'])
df_day_hours ['Day_Hours'] = day_hours
df_day_hours ['Key_HR'] = 1
df_Fis_Cal_hrs = df_fis_cal [['id_Dy',	'Fiscal_Year',	'Fiscal_Month',	'Fiscal_Week',	'Fiscal_Day','Fis_Mon_Day',	'Cal_Date',	'Fis_Cal_date', 'Key_HR']].merge (df_day_hours[['Day_Hours', 'Key_HR']], on = ['Key_HR'] , how = 'outer')

############  MODIFY BELOW TO ADD DC TO MODEL #######################################################################
df_DC = pd.DataFrame(index=np.arange(0,8),columns=['Key_HR', 'DC_name', 'DC_id'])
DC_name = ['OFC','TFC','WFC','EAO','ODC','WEO','OCC','SFS']
#DC_name = ['EAO','SFS']
#DC_name = ['WEO','SFS']
DC_id = [1,2,3,4,5,6,7,8]
## Below Code checks for all dc in  input data has entry


# Code for Missing Dc


## 
############  MODIFY BELOW TO ADD DC TO MODEL #######################################################################


df_DC ['Key_HR'] = 1
df_DC ['DC_name'] = DC_name
df_DC ['DC_id'] = DC_id


dc=4 # you have to change the value down in the for loop as well
df2 = df2[df2['DC'].isin(DC_id)] # might have to comment later




#  Adding mdates (date key)
# Below creates each combination fiscal dates and Dc for all dates in scenario df_fis_cal_hrs_dc
df_Fis_Cal_hrs_DC = pd.merge(df_Fis_Cal_hrs, df_DC , on = ['Key_HR'] , how = 'outer')
# Write in mdate key and make it as int datatype
df_Fis_Cal_hrs_DC['mdates'] = df_Fis_Cal_hrs_DC['Cal_Date'].map(mdates.date2num)
df_Fis_Cal_hrs_DC.mdates = df_Fis_Cal_hrs_DC.mdates.astype(int)
df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC.loc[(df_Fis_Cal_hrs_DC['Fiscal_Year']>= Base_Year),:]
df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC.loc[(df_Fis_Cal_hrs_DC['DC_id']!= 8),:]

#df_Fis_Cal_hrs_DC ['Capacity'] = 0



############  SETUP DATAFRAME FOR DC CAPACTY #######################################################################
#dc_cap_no_event = dc_cap[dc_cap['Event']=='Normal']

#First update "Normal Capacity" for all dates
#df_Fis_Cal_hrs_DC = pd.merge(df_Fis_Cal_hrs_DC, dc_cap_no_event[['Day_Hours','Hrly', 'DC_name']] , on = ['Day_Hours', 'DC_name'] , how = 'left')


# Overwrite with Normal_One_Day only for that particalur day
#dc_cap_Normal_One_Day = dc_cap[dc_cap['Event']=='Normal_One_Day']
#dc_cap_Normal_One_Day['mdates'] = dc_cap_Normal_One_Day['Fis_Cal_date'].map(mdates.date2num)
#dc_cap_Normal_One_Day.mdates = dc_cap_Normal_One_Day.mdates.astype(int)
#dc_cap_Normal_One_Day = dc_cap_Normal_One_Day.sort_values(by=['DC','mdates'])
#df_Fis_Cal_hrs_DC = pd.merge(df_Fis_Cal_hrs_DC, dc_cap_Normal_One_Day[['Day_Hours','Hrly', 'DC_name','mdates']] , on = ['Day_Hours', 'DC_name','mdates'] , how = 'left')
#df_Fis_Cal_hrs_DC.loc[df_Fis_Cal_hrs_DC[~df_Fis_Cal_hrs_DC['Hrly_y'].isnull()].index,'Hrly_x'] = df_Fis_Cal_hrs_DC[~df_Fis_Cal_hrs_DC['Hrly_y'].isnull()]['Hrly_y']
#df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC.drop(['Hrly_y'], axis=1)
#df_Fis_Cal_hrs_DC.rename(columns={'Hrly_x': 'Hrly'}, inplace=True)



# Then Overwrite with "Normal Future" Capacity
#dc_cap_Normal_Future = dc_cap[dc_cap['Event']=='Normal_Future']
#dc_cap_Normal_Future['mdates'] = dc_cap_Normal_Future['Fis_Cal_date'].map(mdates.date2num)
#dc_cap_Normal_Future.mdates = dc_cap_Normal_Future.mdates.astype(int)
#dc_cap_Normal_Future = dc_cap_Normal_Future.sort_values(by=['DC','mdates'])
#dc_cap_Normal_Future = dc_cap_Normal_Future.loc[(dc_cap_Normal_Future['Day_Hours']==7),:] # Capacity at 7 Am- Normally the hrly capacity is same for all 9 hrs during the day
#dc_cap_Normal_Future_date_DC_unique = dc_cap_Normal_Future[['DC', 'mdates', 'Hrly' ]].drop_duplicates()
#dc_cap_Normal_Future_date_DC_unique = dc_cap_Normal_Future_date_DC_unique.sort_values(by=['DC','mdates'])
#dc_cap_Normal_Future= dc_cap_Normal_Future.drop(['DC_name','Fis_Cal_date','Event'], axis=1)


#reduce_mem_usage(dc_cap_Normal_Future_date_DC_unique)
#print(mem_usage(dc_cap_Normal_Future_date_DC_unique))


#conn = sqlite3.connect("df_Fis_Cal_hrs_DC.db")
#df_Fis_Cal_hrs_DC.to_sql("df_Fis_Cal_hrs_DC", conn, if_exists="replace")

#pd.read_sql_query("select * from df_Fis_Cal_hrs_DC where mdates = 736656 and DC_id = 1  ;", conn).head(15)

#cur = conn.cursor()

#@jit
#def normal_future(df_Fis_Cal_hrs_DC, dc_cap_Normal_Future_date_DC_unique, dc_cap_Normal_Future):
#len(dc_cap_Normal_Future_date_DC_unique)

#DC_ID_loop_Normal_future = dc_cap_Normal_Future_date_DC_unique.iat[0,1]
#@jit
#def normal_future_cap_update( dc_cap_Normal_Future_date_DC_unique):
#    for row in dc_cap_Normal_Future_date_DC_unique.itertuples():
#        print ("c1 :",row.DC,"c2 :",row.mdates)
#        values = (row.Hrly, row.mdates, row.DC)
#        cur.execute("update df_Fis_Cal_hrs_DC set Hrly =? where mdates >= ? and DC_id = ? and Day_Hours in (7,8,9,10,11,12,13,14,15)  ", values)
#        conn.commit()
#    return  pd.read_sql_query("select * from df_Fis_Cal_hrs_DC ;", conn)
#
#df_Fis_Cal_hrs_DC = normal_future_cap_update( dc_cap_Normal_Future_date_DC_unique)

#for loop_Normal_future in range(0, len(dc_cap_Normal_Future_date_DC_unique)):
#    DC_ID_loop_Normal_future = dc_cap_Normal_Future_date_DC_unique.iat[loop_Normal_future,1]
#    mdates_loop_normal_future = dc_cap_Normal_Future_date_DC_unique.iloc[loop_Normal_future,3]
#    dc_cap_Normal_Future_loop = dc_cap_Normal_Future [(dc_cap_Normal_Future ['mdates'] == mdates_loop_normal_future) & (dc_cap_Normal_Future ['DC'] == DC_ID_loop_Normal_future) ]
#    df_Fis_Cal_hrs_DC = pd.merge(df_Fis_Cal_hrs_DC[['id_Dy',	'Fiscal_Year',	'Fiscal_Month',	'Fiscal_Week',	'Fiscal_Day',	'Fis_Mon_Day',	'Key_HR',	'Day_Hours',	'DC_id',	'mdates', 'Hrly']], dc_cap_Normal_Future_loop[['Day_Hours','Hrly', 'DC']] , left_on=['Day_Hours','DC_id'] ,right_on = ['Day_Hours', 'DC'] , how = 'left')
#    df_Fis_Cal_hrs_DC.loc[(df_Fis_Cal_hrs_DC['mdates']>= mdates_loop_normal_future ) & (df_Fis_Cal_hrs_DC['DC_id']== DC_ID_loop_Normal_future ), 'Hrly_x'] = df_Fis_Cal_hrs_DC.loc[(df_Fis_Cal_hrs_DC['mdates']>= mdates_loop_normal_future ) & (df_Fis_Cal_hrs_DC['DC_id']== DC_ID_loop_Normal_future ), 'Hrly_y']
#    df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC.drop(['DC','Hrly_y'], axis=1)
#    df_Fis_Cal_hrs_DC.rename(columns={'Hrly_x': 'Hrly'}, inplace=True)
#    #df_Fis_Cal_hrs_DC.rename(columns={'DC_name_x': 'DC_name'}, inplace=True)
#    #df_Fis_Cal_hrs_DC.to_excel(inputfile_path + str(loop_Normal_future) + "out.xlsx")
#    print(str(loop_Normal_future) + "-"+  str(DC_ID_loop_Normal_future) + "-"+ str(mdates_loop_normal_future) )
#    #df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC.drop(['DC_name','Fis_Cal_date','Cal_Date'], axis=1)
#    reduce_mem_usage(df_Fis_Cal_hrs_DC)
#    print(mem_usage(df_Fis_Cal_hrs_DC)) 
#    print (df_Fis_Cal_hrs_DC.count().max())
#    #return df_Fis_Cal_hrs_DC
    
    


#df_Fis_Cal_hrs_DC2 = normal_future(df_Fis_Cal_hrs_DC, dc_cap_Normal_Future_date_DC_unique, dc_cap_Normal_Future)
#df_Fis_Cal_hrs_DC = df_Fis_Cal_hrs_DC2
# Finally Overwrite "Special" Capacity for specific dates
#dc_cap_event = dc_cap[(dc_cap['Event']=='Special') & (dc_cap['Fiscal_Year']== Base_Year) ]
#df_temp = pd.merge(df_Fis_Cal_hrs_DC[df_Fis_Cal_hrs_DC['Fiscal_Year']== Base_Year], dc_cap_event[['Day_Hours','Hrly', 'DC_name', 'Fiscal_Day', 'Fiscal_Week','Fis_Cal_date']], how='left', left_on=['DC_name','Day_Hours', 'Fiscal_Day','Fiscal_Week', 'Cal_Date' ], right_on=['DC_name','Day_Hours', 'Fiscal_Day','Fiscal_Week','Fis_Cal_date' ])
# Hourly X Contains Normal and Normal Future
# Hourly Y contains Special
# Update Null Hourly X with Hourly Y -> essentially updating hourly x with special capacity
#df_temp.loc[df_temp[~df_temp['Hrly_y'].isnull()].index,'Hrly_x'] = df_temp[~df_temp['Hrly_y'].isnull()]['Hrly_y']
#df_temp = df_temp.drop(['Fis_Cal_date_y'], axis=1)

#dc_cap_event_future = dc_cap[(dc_cap['Event']=='Special') & (dc_cap['Fiscal_Year']> Base_Year) ]
#df_temp2 = pd.merge(df_Fis_Cal_hrs_DC[df_Fis_Cal_hrs_DC['Fiscal_Year']> Base_Year], dc_cap_event_future[['Day_Hours','Hrly', 'DC_name', 'Fiscal_Day', 'Fiscal_Week']], how='left', left_on=['DC_name','Day_Hours', 'Fiscal_Day','Fiscal_Week' ], right_on=['DC_name','Day_Hours', 'Fiscal_Day','Fiscal_Week' ])
#df_temp2.loc[df_temp2[~df_temp2['Hrly_y'].isnull()].index,'Hrly_x'] = df_temp2[~df_temp2['Hrly_y'].isnull()]['Hrly_y']
#df_temp= pd.concat([df_temp,df_temp2],axis=0)



############ ??????
#df_temp['Cal_Date'] =  [dt.date() for dt in df_temp["Cal_Date"]]
#df_temp = df_temp[start:end]

print('Capacity_Modeling')  
df_temp=pd.read_excel(inputfile_path + "df_cap_SQL_server.xlsx",sheet_name = "Sheet1")
df_temp = df_temp[(df_temp['Cal_Date'] >= start) & (df_temp['Cal_Date'] <= end )]  

df_temp['mdates'] = df_temp['Cal_Date'].map(mdates.date2num)
df_temp.mdates = df_temp.mdates.astype(int)

Min_mdate = df_temp.loc[(df_temp['Cal_Date'] == start),'mdates'].max()
Max_mdate = df_temp.loc[(df_temp['Cal_Date'] == end),'mdates'].max()




df_temp['Key'] = df_temp['mdates']  * 24 + df_temp['Day_Hours']
df_temp = df_temp.sort_values(by=['DC_id', 'Key'])
Min_id_cap= int(min(df_temp['Key']))
Max_id_cap= int(max(df_temp['Key']))

print('Capacity_Modeling_Inc_Hrs')  
Capacity_inc_start = time.time()

no_dc_Cap = len(df_temp['DC_id'].unique())
@jit
def df_temp_inc_hrs(a,b, Min_id_cap, Max_id_cap, no_dc_Cap2):
    no_dc_Cap = no_dc_Cap2
    
    
    for dc_cap in range (1,no_dc_Cap+1):
        no_of_hrs = 0
    
        for no_of_hrs_inc in range(Min_id_cap,Max_id_cap +1 ):
            if a == no_of_hrs_inc and b == dc_cap :
                c = no_of_hrs
        
            no_of_hrs = no_of_hrs + 1
    return c
            
#df_temp = df_temp_inc_hrs(df_temp, Min_id_cap, Max_id_cap)

vfunc = np.vectorize(df_temp_inc_hrs)

print('before Optimization memory-' + mem_usage(df_temp))     
reduce_mem_usage(df_temp[['DC_id','Day_Hours', 'Fis_Mon_Day', 'Fiscal_Day', 'Fiscal_Month', 'Fiscal_Week', 'Fiscal_Year', 'Hrly_x','mdates', 'Key']])
print('after Optimization memory-' + mem_usage(df_temp))           
df_temp['Day_Hrs_Inc'] = vfunc(df_temp['Key'], df_temp['DC_id'], Min_id_cap, Max_id_cap, no_dc_Cap)
df_temp = df_temp[df_temp['DC_id'].isin(DC_id)]
capacity_rate_inc_end = time.time()
        
df_temp.head(100)

print("Elapsed (after compilation) = %s" % (capacity_rate_inc_end - Capacity_inc_start))


print('Capacity_Modeling_Inc_Hrs_ends')  
   
############  END SETUP DATAFRAME FOR DC CAPACTY #######################################################################
######## df_temp contains DC capacity by day and hour for use in scenario

# Handing Nan errors
def isNaN(num):
    return num != num

print('baclog')  

############   SETUP DATAFRAME FOR Backlog #######################################################################
df_backlog = pd.DataFrame(index=np.arange(0,(Max_id_cap +1  - Min_id_cap)),columns=['Hour',  'Baclog', 'Key_DC' ])
pd.options.display.max_columns = None

no_of_hrs = 0

for no_of_hrs_inc in range(Min_id_cap,Max_id_cap +1 ):
    
    df_backlog.iat[no_of_hrs,0] = no_of_hrs
    no_of_hrs = no_of_hrs + 1

df_backlog['Key_DC'] = 1

df_backlog = pd.merge(df_backlog, df_DC , left_on = ['Key_DC'] , right_on = ['Key_HR']  , how = 'outer')
df_backlog['Baclog'] = df_backlog['Baclog'].astype(float)
df_backlog['Baclog'] = 0.0
df_backlog['Min_Time_To_Promise'] = 0.0
df_backlog['Max_Time_To_Promise'] = 0.0
df_backlog['Avg_Time_To_Promise'] = 0.0
df_backlog['Rem_dc_Cap_Start'] = 0.0
df_backlog['Rem_dc_Cap_End'] = 0.0
df_backlog['Total_DC_Cap_Used'] = 0.0
df_backlog['U_Process_Brand_1'] = 0.0
df_backlog['U_Process_Brand_2'] = 0.0
df_backlog['U_Process_Brand_3'] = 0.0
df_backlog['U_Process_Brand_4'] = 0.0
df_backlog['U_Process_Brand_5'] = 0.0
df_backlog['U_Process_Brand_6'] = 0.0
df_backlog['U_Process_Brand_dummy_99'] = 0.0
df_backlog['U_Process_Ship_Option_1'] = 0.0
df_backlog['U_Process_Ship_Option_2'] = 0.0
df_backlog['U_Process_Ship_Option_3'] = 0.0
df_backlog['U_Process_Ship_Option_4'] = 0.0
df_backlog['Total_DC_Demand_At_Start'] = 0.0
df_backlog['Total_DC_Demand_At_End'] = 0.0
df_backlog['Baclog_Units_Brand_1'] = 0.0
df_backlog['Baclog_Units_Brand_2'] = 0.0
df_backlog['Baclog_Units_Brand_3'] = 0.0
df_backlog['Baclog_Units_Brand_4'] = 0.0
df_backlog['Baclog_Units_Brand_5'] = 0.0
df_backlog['Baclog_Units_Brand_6'] = 0.0
df_backlog['Baclog_Units_Brand_dummy_99'] = 0.0
df_backlog['Baclog_Units_Ship_Option_1'] = 0.0
df_backlog['Baclog_Units_Ship_Option_2'] = 0.0
df_backlog['Baclog_Units_Ship_Option_3'] = 0.0
df_backlog['Baclog_Units_Ship_Option_4'] = 0.0
df_backlog['Starting_Baclog_7AM'] = 0.0
df_backlog['Starting_Baclog_if_Var_Cap_Used_7AM'] = 0.0 # this is basicaly to know if the new capacity is applied
df_backlog['Outstanding_demand_7AM'] = 0.0
df_backlog['Variable_capacity_used']= 'N'

# Have to define the variables and assign some values- Python is giving name error

Min_TTP_Baclog = 0
Max_TTP_Baclog = 0
Avg_Time_To_Promise_baclog=0
rem_dc_cap_start_baclog = 0
rem_dc_cap_end_baclog = 0
Total_DC_Cap_Used_baclog = 0
U_Process_Brand_1_baclog= 0
U_Process_Brand_2_baclog= 0
U_Process_Brand_3_baclog= 0
U_Process_Brand_4_baclog= 0
U_Process_Brand_5_baclog= 0
U_Process_Brand_6_baclog= 0
U_Process_Brand_dummy_99_baclog= 0
U_Process_Ship_option_1_baclog= 0
U_Process_Ship_option_2_baclog= 0
U_Process_Ship_option_3_baclog= 0
U_Process_Ship_option_4_baclog= 0
Total_DC_Demand_At_Start_baclog = 0
Total_DC_Demand_At_End_baclog = 0
Total_Dc_Demand_Brand_1 =0
Total_Dc_Demand_Brand_2 =0
Total_Dc_Demand_Brand_3 =0
Total_Dc_Demand_Brand_4 =0
Total_Dc_Demand_Brand_5 =0
Total_Dc_Demand_Brand_6 =0
Total_Dc_Demand_dummy_99 =0

Total_Dc_Demand_Ship_Option_1 =0
Total_Dc_Demand_Ship_Option_2 =0
Total_Dc_Demand_Ship_Option_3 =0
Total_Dc_Demand_Ship_Option_4 =0
Starting_Baclog = 0
Starting_Baclog_if_Var_Cap_Used_7AM = 0
Outstanding_demand_7AM = 0
Variable_capacity_used = 'N'








#df_backlog['Hour']=np.random.randint(0,30, size=len(df_backlog))


#df_backlog['DC_id']=np.random.randint(1,2, size=len(df_backlog))


#df_temp_x.to_excel(inputfile_path + "out.xlsx")

#dc_cap = pd.DataFrame(data, index=['Hrly', 'Remaining'], columns=['1', '2', '3'])



#dfx is fake data at approximate scale of actual data for sizing
#dfx = pd.DataFrame(index=np.arange(0,48),columns=['Year','Hour', 'Zip', 'Network Alignment','DC', 'Segment','Ship Option','Brand','Ordered Units', 'Modeled DC', 'Modeled Ship Option', 'Modeled Order Units', 'Time to Promise', 'Remaining Units', 'Time fully Processed','UPP', 'Units_Processed1' , 'Units_Processed_Final', 'No_of_Packages', 'No_of_Packages_final', 'Cost','Total_Cost' ])
#pd.options.display.max_columns = None
#df2.drop('Time_Status',1)
#x = np.array(range(1,366))
#y = np.repeat(x, 24*50)
#a= np.array(range(1,24))
#df2['Day']=y
#a= np.array(range(1,25))
#b= np.tile(a,365*50)
#df2['Hour']=b
#df2['Time_Status']='BOH'
Year = 2018

#dfx['Year']= Year

#dfx['Hour']=np.random.randint(0,30, size=len(dfx))
#dfx['Zip']=np.random.randint(94407,94408, size=len(dfx))
#dfx['Ordered Units']=np.random.randint(10,999, size=len(dfx))
#dfx['DC']=np.random.randint(1,2, size=len(dfx))
#UPP_Year1 = [df_UPP_final()]


# Comment df2['Brand']=np.random.randint(1,6, size=len(df2))
#df2['Network Alignment']=np.random.randint(1,4, size=len(df2))


#Net_Alignment = ['Base slow', 'Fast - BRAT center', 'Fast - BRAT coastal']

#dfx['Network Alignment']=np.random.choice(Net_Alignment, size=len(dfx))

#Brand = ['ATOL','BROL','GOL','ONOL','GOBRFS']

#dfx['Brand']=np.random.choice(Brand, size=len(dfx))

#dfx['Time to Promise']=np.random.randint(1,240, size=len(dfx))





#dfx['Segment']=np.random.randint(1,6, size=len(dfx))
#dfx['Modeled Order Units']=dfx['Ordered Units']

#df2 = dfx [['Year','Hour', 'Zip', 'Network Alignment','DC', 'Segment','Ship Option','Brand','Ordered Units', 'Modeled DC', 'Modeled Ship Option', 'Modeled Order Units', 'Time to Promise', 'Remaining Units', 'Time fully Processed', 'Units_Processed1' , 'Units_Processed_Final', 'No_of_Packages', 'No_of_Packages_final', 'Cost','Total_Cost']].merge (df_UPP_final[['Year','Brand','UPP']],  on = ['Year','Brand'], how = 'left')

#df2 = df3 [['Year','Hour', 'Zip', 'Network Alignment','DC', 'Segment','Ship Option','Brand','Ordered Units', 'Modeled DC', 'Modeled Ship Option', 'Modeled Order Units', 'Time to Promise', 'Remaining Units', 'Time fully Processed', 'Units_Processed1' , 'Units_Processed_Final', 'No_of_Packages', 'No_of_Packages_final', 'Cost','Total_Cost','UPP',]].merge (df_Parcel_Cost_final[['Year','Network Alignment','Promise_time', 'Brand']],  on = ['Year','Network Alignment', 'Brand'], how = 'left')

df2['Units_Processed1'] = np.dtype('f8')
df2['Units_Processed1'] = 0.00
df2['Units_Processed_Final'] = np.dtype('f8')
df2['Units_Processed_Final'] = 0.00
df2['No_of_Packages'] = np.dtype('f8')
df2['No_of_Packages'] = 0.00
df2['No_of_Packages_final'] = np.dtype('f8')
df2['No_of_Packages_final']= 0.00
df2['Cost'] = np.dtype('f8')
df2['Cost'] = 0.00
df2['Total_Cost'] = np.dtype('f8')
df2['Total_Cost']= 0.00

#Geo = 'Not_Geo'
#df2['Geo'] = Geo

#TIme_to_promise = [24,48,72,96,120,144,168,192,216,240]
#df2['Time to Promise']=np.random.choice(TIme_to_promise, size=len(df2))

#Load in actual df2 from SQL data


#df2 = df2[(df2['mdates'] >= Min_mdate) & (df2['mdates'] <= Max_mdate )]  
#df3x =  df2.copy()
df2['mdates'] = df2['Order_day'].map(mdates.date2num)
df2.mdates = df2.mdates.astype(int)

df2 = df2[(df2['mdates'] >= Min_mdate) & (df2['mdates'] <= Max_mdate )]  
df2.mdates.value_counts()


df2['UPP'] = 0.00

df2 = df2[df2['DC'].isin(DC_id)]

print('capacity_merge_df2')  


df2 = pd.merge(df2, df_temp[['Cal_Date', 'Fiscal_Year', 'Fiscal_Month', 'Fiscal_Week', 'Fiscal_Day', 'mdates','Day_Hours',  'Day_Hrs_Inc', 'DC_id']].drop_duplicates() , right_on = ['mdates', 'DC_id', 'Day_Hours'] , left_on = ['mdates', 'DC','Order_Hr']  , how = 'inner')
#df2 = pd.merge(df2, Ship_Option , left_on = ['Modeled Ship Option','Fiscal_Day'], right_on = ['Shipment_Type_id','Fiscal_Day']  , how = 'inner')
df2['Hour'] = df2['Day_Hrs_Inc']
df2['New_Promise_Time'] = 0
#df2.loc[(df2['Order_Hr']< 15), 'New_Promise_Time'] = df2.loc[(df2['Order_Hr']< 15), 'TTP_Order_Entry_Before_3PM']
#df2.loc[(df2['Order_Hr']>= 15), 'New_Promise_Time'] = df2.loc[(df2['Order_Hr']>= 15), 'TTP_Order_Entry_After_3PM']
#df2.loc[(df2['Delivery_Option']== "Y"), 'New_Promise_Time'] = 15 - df2.loc[(df2['Delivery_Option']== "Y"), 'Order_Hr']  +  df2.loc[(df2['Delivery_Option']== "Y"), 'New_Promise_Time']
#df2 = df2.drop(['Cal_Date','Day_Hours','Day_Hrs_Inc', 'DC_id', 'sno', 'Shipment_Type', 'Delivery_Option', 'TTP_Order_Entry_Before_3PM', 'TTP_Order_Entry_After_3PM','DC_name', 'Day', 'Shipment_Type_id'], axis=1)
df2= df2.loc[df2['Modeled Order Units'] >0,:]
#-----
#df2['Key'] = df2['mdates']  * 24 + df2['Order_Hr']
#df2 = df2.sort_values(by=['DC', 'Key'])

#Min_id_main_data= int(min(df2['Key']))
#Max_id_main_data= int(max(df2['Key']))




#for dc_cap in range (1,no_dc_Cap+1):
#    no_of_hrs = 0

#    for no_of_hrs_inc in range(Min_id_main_data,Max_id_main_data +1 ):
    
#        df2.loc[(df2['Key'] == no_of_hrs_inc) & (df2['DC'] == dc_cap)  ,'Hour'] = no_of_hrs
#        no_of_hrs = no_of_hrs + 1
        
#df2.to_excel("merged3.xlsx")
Max_hour_main_Data = int(max(df2['Hour']))


# Converting the dataframes string to Upper case
#df2[['Geo']].apply(lambda value: value.str.upper())
#df2['Geo'] = df2['Geo'].str.upper()
#df2['Network Alignment'] = df2['Network Alignment'].str.upper()
#df2['Brand'] = df2['Brand'].str.upper()
#df_UPP_final['Brand'] = df_UPP_final['Brand'].str.upper()
#Cancelation_Rate['Brand'] = Cancelation_Rate['Brand'].str.upper()
#df_Parcel_Cost_final['Network Alignment'] = df_Parcel_Cost_final['Network Alignment'].str.upper()
#df_Parcel_Cost_final['Geo'] = df_Parcel_Cost_final['Geo'].str.upper()


# Cancelation rate insertion
print('cancelation_rate')

df2['Cancelation_Rate']=0
df2['UPP']=0

conn = sqlite3.connect("df2.db")
df2.to_sql("df2", conn, if_exists="replace")




cur = conn.cursor()
@jit
def cancelation_rate_update(Cancelation_Rate):
    for row in Cancelation_Rate.itertuples():
        values = (row.Cancelation_Rate, row.Brand_id)
        cur.execute("update df2 set Cancelation_Rate  = ?  where Brand = ?", values)
        conn.commit()
    return  pd.read_sql_query("select * from df2 ;", conn)

df2 = cancelation_rate_update(Cancelation_Rate)

df2 = df2.drop(['index'], axis=1)
print('cancelation_rate_ends')
#pd.read_sql_query("select * from df2 limit 2;", conn)

#pd.read_sql_query("select * from Cancelation_Rate limit 2;", conn2)


#df2= pd.merge(df2, Cancelation_Rate , left_on=['Brand'], right_on=['Brand_id']  , how = 'left')
#df2 = df2.drop(['Brand_id', 'Brand_y'], axis=1)
#df2.rename(columns={'Brand_x': 'Brand'}, inplace=True)

# UPP code 

#conn = sqlite3.connect("df2.db")
#df2.to_sql("df2_UPP", conn, if_exists="replace")
#cur = conn.cursor()

#pd.read_sql_query("select * from df2_UPP limit 2;", conn)


Brand_list_for_UPP = df_UPP_final[['Brand_id','Brand', 'mdates_from', 'mdates_to','UPP']].drop_duplicates()
#len_Brand_list_UPP = len(Brand_list_for_UPP)
Brand_list_for_UPP = Brand_list_for_UPP.sort_values(by=['Brand_id','mdates_from'])
Brand_list_for_UPP = Brand_list_for_UPP.drop(['Brand'], axis=1)
Brand_list_for_UPP.rename(columns={'Brand_id': 'Brand'}, inplace=True)
#df_UPP_final = df_UPP_final.drop(['Brand'], axis=1)
#df_UPP_final.rename(columns={'Brand_id': 'Brand'}, inplace=True)

                        
#Order_mdate_UPP = df2.loc[df2['Hour']==mhr, 'mdates'].unique()
#Brand_loop_UPP = 0

start_time_function = time.time()
def UPP_Update_dataframe (Brand_list_for_UPP):
    
    for row in Brand_list_for_UPP.itertuples():
        values = (row.UPP, row.Brand, row.mdates_from, row.mdates_to)
        cur.execute("update df2 set UPP  = ?  where Brand = ? and mdates >= ? and mdates <= ?", values)
        conn.commit()
    return  pd.read_sql_query("select * from df2 ;", conn)




df2= UPP_Update_dataframe(Brand_list_for_UPP)
end_time_function = time.time()

print ('Time was %.3f'%(end_time_function-start_time_function ) )
df2 = df2.drop(['index'], axis=1)
print('UPP_update_ends')
cur.close()


df2['CPP'] = np.dtype('f8')
df2['CPP'] = 0.00
df2['Time fully Processed'] = np.dtype('f8')
df2['Time fully Processed'] = 0




df_counting_time = pd.DataFrame(index=np.arange(0,len(df2['Hour'].drop_duplicates())),columns=['Hour','Start_time', 'End_time'])

df_counting_time['Hour'] = df2['Hour'].unique()
df_counting_time ['Start_time'] = start_time
df_counting_time ['End_time'] = start_time

# final memory optimization
#df2['YearWeek'] = df2['Order_day'].dt.strftime('%Y%U')
df2['YearWeek'] = df2['Fiscal_Year'] * 100 + df2['Fiscal_Week']
df2.YearWeek = pd.to_numeric(df2.YearWeek, errors='coerce').fillna(0).astype(np.int64)
df_Unique_week_id = df2[['YearWeek']].drop_duplicates()
df_Unique_week_id = df_Unique_week_id.sort_values(by=['YearWeek'])

df_Unique_week_id = df_Unique_week_id.reset_index()
df_Unique_week_id['Yearweek_id'] =  df_Unique_week_id.index + 1
df_Unique_week_id = df_Unique_week_id.drop(['index'], axis=1)
df2 = pd.merge(df2,df_Unique_week_id, on = ['YearWeek'])
print('Before Optimization memory-' + mem_usage(df2))

df2 = df2.drop(['Order_day'], axis=1)
reduce_mem_usage(df2)
reduce_mem_usage(df_counting_time)
print('after Optimization memory-' + mem_usage(df2))


# final memory optimization ends


# Network Alignment  Network_Alignment_id

df_Parcel_Cost_final = df_Parcel_Cost_final.drop(['Network Alignment'], axis=1)
df_Parcel_Cost_final.rename(columns={'Network_Alignment_id': 'Network Alignment'}, inplace=True)
df_Parcel_Cost_final = df_Parcel_Cost_final.drop(['Geo'], axis=1)
df_Parcel_Cost_final.rename(columns={'Geo_id': 'Geo'}, inplace=True)




df2 = df2.sort_values(by=['Hour'])
Min_Year_week= int(min(df2['Yearweek_id']))
Max_Year_week= int(max(df2['Yearweek_id']))+1
df2['Shipped_Units'] = df2['Modeled Order Units'] * (1- df2['Cancelation_Rate'])
df2['Capacity_Used'] = 0
df2['Theoritical_Max_Capacity_Used'] = 0 # 0 means not used
#Simulation LLoop - each loop is 1 simulated year

# No OCC and SFS in the capacity. 
df_temp =df_temp.loc[(df_temp['DC_id']!=8),:]
#df_temp =df_temp.loc[(df_temp['DC_id']!=7),:]



df_SFS = df2.loc[(df2['DC']==8),:]
df_fullyear=df2.loc[(df2['DC']!=8),:]
df_processed =df_fullyear.loc[(df_fullyear['YearWeek'] == 0),:] # Creating the blank dataframe
df_carryover =df_fullyear.loc[(df_fullyear['YearWeek'] == 0),:] # Creating the blank dataframe

#df2= df3.loc[(df3['YearWeek'] == Min_Year_week),:]

for yearweek_id in range(Min_Year_week,Max_Year_week):
    df2= pd.concat([df_fullyear.loc[(df_fullyear['Yearweek_id'] == yearweek_id),:],df_carryover],axis=0)
    df2 = df2.reset_index(drop=True)
    
    Min_df2hour= df_fullyear.loc[(df_fullyear['Yearweek_id'] == yearweek_id),'Hour'].min()
    Max_df2hour= df_fullyear.loc[(df_fullyear['Yearweek_id'] == yearweek_id),'Hour'].max()
    yearweek = df_fullyear.loc[(df_fullyear['Yearweek_id'] == yearweek_id),'YearWeek'].max()
    if isNaN(Min_df2hour):
        Min_df2hour = 0
    if isNaN(Max_df2hour):
        Max_df2hour = -1
    if isNaN(yearweek):
        yearweek = 0
    

    
    #Copy data from previous year to this year
    
    #--w/ units growing at growth rate for brand
    
    
    #--w/ new ship options: Set ship option designation based on 'Customer Segment', 'Zip', etc
    
    
    #--w/ other data modifications 
    
    # Simulation Loop - each i is 1 hr
    for mhr in range(Min_df2hour, Max_df2hour+1):
        print('Test')
        df_counting_time.loc[(df_counting_time ['Hour'] == mhr), 'Start_time'] = time.time()
        
         
        #New Modeled hour - Refresh remaining DC Capacity
        #pdb.set_trace()  
        if mhr % 24 ==7:
            print(mhr / 24)
            print(yearweek)
            Variable_capacity_baclog = 1 #Trigger this option 1 at the start of the day
        # Assigning DC cap to the base data to remaining capacity for that hour
        
        #dc_cap.loc[dc_cap[dc_cap['Hour']==mhr].index[0],'Remaining']=(dc_cap[dc_cap['Hour'] == mhr]['Hrly'].iloc[0])
        #Copy over 'Modeled Order Units' from base data to 'Remaining Ordered Units' for that hour
        # UPP update code dynamic by dates
        
       # Brand_list_for_UPP = df2.loc[df2['Hour']==mhr, 'Brand'].unique()
       # len_Brand_list_UPP = len(Brand_list_for_UPP)
       # Order_mdate_UPP = df2.loc[df2['Hour']==mhr, 'mdates'].unique()
       # Brand_loop_UPP = 0
       # for Brand_loop_UPP in range(0 ,len_Brand_list_UPP):
       #     Date_Upp_Changed = df_UPP_final.loc[(df_UPP_final['Brand'] == Brand_list_for_UPP[Brand_loop_UPP]) & (df_UPP_final['mdates'] <= Order_mdate_UPP[0]), 'Fis_Cal_Date' ].max()
       #     UPP = df_UPP_final.loc[(df_UPP_final['Brand'] == Brand_list_for_UPP[Brand_loop_UPP]) & (df_UPP_final['Fis_Cal_Date'] == Date_Upp_Changed), 'UPP' ].max()
       #     df2.loc[(df2['Hour']==mhr) & (df2['Brand']==Brand_list_for_UPP[Brand_loop_UPP]) , 'UPP'] = UPP
        
        #This is entry into DC queue     
        df2['Units_Processed1'] = 0.00
        df2['No_of_Packages'] = 0.00          
        df2.loc[df2['Hour']==mhr, 'Remaining Units']=df2['Modeled Order Units']
        
        
        #Calculate 'Time to Promise' for all demand records in current modeled hour
        #pdb.set_trace()
        #TIme_to_promise = [24,48,72,96,120,144,168,192,216,240]
        # fix the issue Fiscal_Year = df2[(df2['Hour'] == mhr)] ['Fiscal_Year'].iloc[0]
        #Fiscal_Year = 2018
        df2.loc[df2['Hour']==mhr, 'Time to Promise']=df2['Promise_time']
        
        # Logic for the new promise time
        #if (Fiscal_Year == Base_Year and New_Promise_Time_Option == 1  ):
        #    df2.loc[df2['Hour']==mhr, 'Time to Promise']=df2['New_Promise_Time']
        
        
        #df2['Time to Promise']=np.random.choice(TIme_to_promise, size=len(df2))
        #df2.loc[df2['Hour']==mhr, 'Time to Promise']=10*24
        
        
        #Model processing of units
        #Loop through each DC
        no_dc = len(df_temp['DC_id'].unique())
        for dc in range(1,no_dc+1):
            if mhr % 24 ==7:
                #print(mhr / 24)
                #print(yearweek)
                Variable_capacity_baclog = 1 #Trigger this option 1 at the start of the da
            #Loop through outstanding DC demand While there is DC Capacity reaming (1 is index for second row...the remaining DC capacity  Note: this should also account for if DC is closed at mhr since DC capacity would be 0 if not open at mhr
            # AND outstanding demand to be processed    
            Min_TTP_Baclog = 0
            Max_TTP_Baclog = 0
            Avg_Time_To_Promise_baclog=0
            rem_dc_cap_start_baclog = 0
            rem_dc_cap_end_baclog = 0
            Total_DC_Cap_Used_baclog = 0
            U_Process_Brand_1_baclog= 0
            U_Process_Brand_2_baclog= 0
            U_Process_Brand_3_baclog= 0
            U_Process_Brand_4_baclog= 0
            U_Process_Brand_5_baclog= 0
            U_Process_Brand_6_baclog= 0
            U_Process_Brand_6_baclog= 0
            U_Process_Brand_dummy_99_baclog= 0
            
            U_Process_Ship_option_1_baclog= 0
            U_Process_Ship_option_2_baclog= 0
            U_Process_Ship_option_3_baclog= 0
            U_Process_Ship_option_4_baclog= 0
            Total_DC_Demand_At_Start_baclog = 0
            Total_DC_Demand_At_End_baclog = 0
            Total_Dc_Demand_Brand_1 =0
            Total_Dc_Demand_Brand_2 =0
            Total_Dc_Demand_Brand_3 =0
            Total_Dc_Demand_Brand_4 =0
            Total_Dc_Demand_Brand_5 =0
            Total_Dc_Demand_Brand_6 =0
            Total_Dc_Demand_dummy_99 =0
            
            Total_Dc_Demand_Ship_Option_1 =0
            Total_Dc_Demand_Ship_Option_2 =0
            Total_Dc_Demand_Ship_Option_3 =0
            Total_Dc_Demand_Ship_Option_4 =0
            Starting_Baclog = 0
            Starting_Baclog_if_Var_Cap_Used_7AM = 0
            Outstanding_demand_7AM = 0
            Variable_capacity_used = 'N'                       
            
            #initialize total_outstanding demand to 1 so while loop starts
            total_dc_outstanding_demand=df2.loc[(df2['DC']==dc), 'Remaining Units'].sum() 
            Total_DC_Demand_At_Start_baclog = total_dc_outstanding_demand
            #Backlog_con_stat = (df2['DC']==dc) & (df2['Remaining Units']>0 )
            
            #rem_dc_cap=dc_cap.iat[1,dc]
            #rem_dc_cap= dc_cap[(dc_cap['Hour'] == mhr) & (dc_cap['DC']==dc)]['Hrly'].iloc[0]
            rem_dc_cap= df_temp[(df_temp['Day_Hrs_Inc'] == mhr) & (df_temp['DC_id']==dc)]['Utilized_Capacity'].iloc[0]
            Backlog_Mdate_Min = df_temp[(df_temp['Day_Hrs_Inc'] == mhr) & (df_temp['DC_id']==dc)]['mdates'].iloc[0]
            Event_Network_Model= df_temp[(df_temp['Day_Hrs_Inc'] == mhr) & (df_temp['DC_id']==dc)]['Event_Network_Model'].iloc[0] # If there is any event for the day or It's a Peak day/Off Peak day/ Thanksgiving/Christmas
            
            if ( Variable_capacity_baclog == 1) :
                rem_dc_cap_baclog = df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Capacity'].sum()
            
            Backlog_Mdate_Max  = Backlog_Mdate_Min +5
            # Before the variable capacity is applied
            Total_five_day_Cap = df_temp.loc[(df_temp['mdates']>=Backlog_Mdate_Min) & (df_temp['mdates']<Backlog_Mdate_Max) & (df_temp['DC_id']==dc) , 'Utilized_Capacity'].sum()
            Avg_five_day_Cap = Total_five_day_Cap/5
            Baclog =  total_dc_outstanding_demand/Avg_five_day_Cap
            
            
            ### Start of Variable capacity if baclog of more than 2 days
            
            if rem_dc_cap_baclog == 0:
                Baclog = 0
                
            else:
                Baclog = total_dc_outstanding_demand/Avg_five_day_Cap
            #if ( Variable_capacity_baclog == 1) : 
            
            if ( Variable_capacity_baclog == 1) :
                Starting_Baclog =  Baclog 
                Outstanding_demand_7AM = total_dc_outstanding_demand
                
            if (Baclog > 1.25 and Variable_DC_capacity_option == 1 and Variable_capacity_baclog == 1 and Event_Network_Model == 'OFF_PEAK'):
                
                df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Capacity'] = df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Theortical_Max_Peak_Capacity'] 
                rem_dc_cap_baclog2 = df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Capacity'].sum()
                rem_dc_cap2 = df_temp[(df_temp['Day_Hrs_Inc'] == mhr) & (df_temp['DC_id']==dc)]['Utilized_Capacity'].iloc[0]
                rem_dc_cap = rem_dc_cap2
                Starting_Baclog_if_Var_Cap_Used_7AM =  total_dc_outstanding_demand/rem_dc_cap_baclog2 
                Variable_capacity_used = 'Y'
                
            if (Baclog > 4.5 and Variable_DC_capacity_option == 1 and Variable_capacity_baclog == 1 and Event_Network_Model == 'PEAK'):
                
                df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Capacity'] = df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Theortical_Max_Peak_Capacity'] 
                rem_dc_cap_baclog2 = df_temp.loc[(df_temp['mdates'] == Backlog_Mdate_Min) & (df_temp['DC_id']==dc),'Utilized_Capacity'].sum()
                rem_dc_cap2 = df_temp[(df_temp['Day_Hrs_Inc'] == mhr) & (df_temp['DC_id']==dc)]['Utilized_Capacity'].iloc[0]
                rem_dc_cap = rem_dc_cap2
                Starting_Baclog_if_Var_Cap_Used_7AM =  total_dc_outstanding_demand/rem_dc_cap_baclog2 
                Variable_capacity_used = 'Y'
            
            #Backlog_Mdate_Max  = Backlog_Mdate_Min +5
            
            # Baclog in the Baclog file is reported based of the Avg_Five_Day_Cap after variable capacity is applied
            Total_five_day_Cap = df_temp.loc[(df_temp['mdates']>=Backlog_Mdate_Min) & (df_temp['mdates']<Backlog_Mdate_Max) & (df_temp['DC_id']==dc) , 'Utilized_Capacity'].sum()
            Avg_five_day_Cap = Total_five_day_Cap/5
            
            Variable_capacity_baclog = 0 # Variable capacity is assigned at the start of the day , not during any hour
            ### End of Variable capacity if baclog of more than 2 days    
            #pdb.set_trace()  
           
            #rem_dc_cap = int(rem_dc_cap)
            #start at smallest time to promise
            ttp=df2.loc[ (df2['Remaining Units']>0) & (df2['DC']== dc), 'Time to Promise'].min()
            #Baclog Statistics 
            
            rem_dc_cap_start_baclog = rem_dc_cap
            #Max_TTP_Baclog=df2.loc[ (df2['Remaining Units']>0) & (df2['DC']== dc), 'Time to Promise'].max()
            
            
            thr = ttp
            lowest_window_demand=df2.loc[(df2['DC']==dc) & (df2['Time to Promise']==thr) , 'Remaining Units'].sum() 
                
            while ((rem_dc_cap > 0.1) & (total_dc_outstanding_demand>0)):
                #Process demand in order of lowest 'Time to Promise' to largest 
                
     
                #for thr in range( df2['Time to Promise'].min() , df2['Time to Promise'].max()+1 ):
                    #If remaining units > DC Capacity, then calculate remaining capacity as % of all demand in thr, apply against demand, update remaining capacity of DC
                  
    
                #pdb.set_trace()
                if rem_dc_cap > 0:
                    total_outstanding_demand=df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc), 'Remaining Units'].sum() 
                    
                    
                    df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Remaining Units']>0), 'Time fully Processed']=mhr
                    
                    if total_outstanding_demand>0:
                        perc_process= rem_dc_cap /total_outstanding_demand
                        if perc_process>1.0:
                            perc_process=1.0
                    
                        df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Units_Processed1'] = df2['Remaining Units']*(perc_process)
                        
                        
                        
                    
                    
                                  
                    #Geo_iterative = pd.DataFrame(index=np.arange(0,1),columns=['Geo'])
                        
                    
                    
                    #filtered_df = df2[df2['Remaining Units'].notnull()]
                    
                    #Geo_iterative
                    
                    #dc_CPP = pd.DataFrame(index=np.arange(0,1),columns=['Time to Promise','Geo','Network Alignment'])
                    #pd.options.display.max_columns = None
    
    
                    #dc_CPP['Time to Promise'] = int(thr)
                    #dc_CPP['Geo'] = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Geo'] 
                    #dc_CPP['Geo'] = 'Not_Geo'
                   #Network_Alignment_iterative = pd.DataFrame(index=np.arange(0,1),columns=['NA'])
                      
                    #kk= pd.merge_asof(left=dc_CPP,right=df_Parcel_Cost_final,on=['Time to Promise'], by = ['Geo', 'Network Alignment'], direction='backward')
                    #pdb.set_trace()  
                        
                        
                    
                        
                        # UPP update
                        
                        
                        
                    
                    
                        df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc)& (df2['Remaining Units']>0), 'Remaining Units']=df2['Remaining Units']*(1-perc_process)
                    
                        rem_dc_cap = rem_dc_cap - (total_outstanding_demand*perc_process)
                        rem_dc_cap_end_baclog = rem_dc_cap
                    
                        
                        #pdb.set_trace()
                        df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'No_of_Packages'] = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Units_Processed1']/ df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'UPP']
                        # Capturing the cancelation rate by Brand
                        Brand_list_for_cancelation = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Brand'].unique()
                        len_Brand_list_cancel = len(Brand_list_for_cancelation)
                        #BrandforCancel_rate = df2[(df2['Time to Promise']==thr) & (df2['DC']==dc)][ 'Brand'].iloc[0]
                        #Cancelation_rate_By_Brand = Cancelation_Rate[(Cancelation_Rate['Brand']== BrandforCancel_rate)]['Cancelation_Rate'].iloc[0] 
                        #Brand_loop = 0
                        #for Brand_loop in range(0 ,len_Brand_list_cancel):
                        #   BrandforCancel_rate = Brand_list_for_cancelation[Brand_loop]
                        #   Cancelation_rate_By_Brand = Cancelation_Rate[(Cancelation_Rate['Brand']== BrandforCancel_rate)]['Cancelation_Rate'].iloc[0]
                        Cancelation_Con = (df2['Time to Promise']==thr) & (df2['DC']==dc   )
                        df2.loc[Cancelation_Con, 'No_of_Packages'] = df2.loc[Cancelation_Con,'No_of_Packages'] * (1-df2.loc[ Cancelation_Con , 'Cancelation_Rate'] )
                        df2.loc[Cancelation_Con, 'No_of_Packages_final'] = df2.loc[Cancelation_Con, 'No_of_Packages'] + df2.loc[Cancelation_Con, 'No_of_Packages_final']
                        df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Units_Processed_Final'] = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Units_Processed1'] + df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Units_Processed_Final']    
                         # Parcel cost Update
                        df_Parcel_Cost_final = df_Parcel_Cost_final.sort_values(by=['Fis_Cal_Date','Time to Promise'])
                        df_Parcel_Cost_final_Geo = df_Parcel_Cost_final_Geo.sort_values(by=['Fis_Cal_Date','Brand_id'])
                        Network_Alignment_iterative = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ), 'Network Alignment'].unique()
                        len_Network_Alignment_iterative = len(Network_Alignment_iterative)
                        Brand_iterative_Geo = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ) & (df2['Geo']==2   ), 'Brand'].unique()
                        len_Brand_iterative_Geo = len(Brand_iterative_Geo)
                        Geo_iterative = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc   ) , 'Geo'].unique()
                        len_Geo_iterative = len(Geo_iterative)
                        Order_mdate_CPP = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Hour']<=mhr), 'mdates'].max()
                        
                        for Geo_iterative_loop in range(0 ,len_Geo_iterative):
                            if (Geo_iterative[Geo_iterative_loop] == 1): # for Non Geo, Use the CPP by promise time
                            
                                for Network_Alignment_iterative_loop in range(0 ,len_Network_Alignment_iterative):
                                    UPP_Changed_date = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['mdates']<= Order_mdate_CPP) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final['DC_id']== dc), 'mdates'].max()
                                    
                                    if (int(thr) >=0):
                                        Promise_time_iterative_max = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['Time to Promise'] <=int(thr)) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final['mdates']== UPP_Changed_date) & (df_Parcel_Cost_final['DC_id']== dc), 'Time to Promise'].max()
                                    if (int(thr) < 0):
                                        Promise_time_iterative_max = 0
        
                                    Parcel_cost_iterative = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['Time to Promise']==Promise_time_iterative_max) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final['mdates']== UPP_Changed_date) & (df_Parcel_Cost_final['DC_id']== dc) , 'CPP'].max()
                                    #df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] = Parcel_cost_iterative * df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'No_of_Packages']
                                    #df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]) , 'Total_Cost'] = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] + df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Total_Cost']    
                                    Loop_cost = Parcel_cost_iterative * df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'No_of_Packages']
                                    df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] = Loop_cost
                                    df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]) , 'Total_Cost'] = Loop_cost + df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Total_Cost']    
                            
                            if (Geo_iterative[Geo_iterative_loop] == 2): # for Geo, Use the CPP By Brand and DC
                            
                                for Brand_iterative_Geo_loop in range(0 ,len_Brand_iterative_Geo):
                                    UPP_Changed_date = df_Parcel_Cost_final_Geo.loc[(df_Parcel_Cost_final_Geo['mdates']<= Order_mdate_CPP) & (df_Parcel_Cost_final_Geo['Brand_id']==Brand_iterative_Geo[Brand_iterative_Geo_loop]) & (df_Parcel_Cost_final_Geo['Geo_id']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final_Geo['DC_id']== dc), 'mdates'].max()
                                    
                                    Parcel_cost_iterative_Geo = df_Parcel_Cost_final_Geo.loc[(df_Parcel_Cost_final_Geo['Brand_id']==Brand_iterative_Geo[Brand_iterative_Geo_loop]) & (df_Parcel_Cost_final_Geo['Geo_id']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final_Geo['mdates']== UPP_Changed_date) & (df_Parcel_Cost_final_Geo['DC_id']== dc) , 'CPP'].max()
                                    #df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] = Parcel_cost_iterative * df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'No_of_Packages']
                                    #df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]) , 'Total_Cost'] = df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] + df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Total_Cost']    
                                    Loop_cost = Parcel_cost_iterative_Geo * df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Brand']==Brand_iterative_Geo[Brand_iterative_Geo_loop]), 'No_of_Packages']
                                    df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Brand']==Brand_iterative_Geo[Brand_iterative_Geo_loop]), 'Cost'] = Loop_cost
                                    df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Brand']==Brand_iterative_Geo[Brand_iterative_Geo_loop]) , 'Total_Cost'] = Loop_cost + df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Brand']==Brand_iterative_Geo[Brand_iterative_Geo_loop]), 'Total_Cost']    
    
                    
                        #for Geo_iterative_loop in range(0 ,len_Geo_iterative):
                         #   for Network_Alignment_iterative_loop in range(0 ,len_Network_Alignment_iterative):
                          #      UPP_Changed_date = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['mdates']<= Order_mdate_CPP) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]), 'mdates'].max()
                                
                          #      Promise_time_iterative_max = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['Time to Promise'] <=int(thr)) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final['mdates']== UPP_Changed_date), 'Time to Promise'].max()
                          #      Parcel_cost_iterative = df_Parcel_Cost_final.loc[(df_Parcel_Cost_final['Time to Promise']==Promise_time_iterative_max) & (df_Parcel_Cost_final['Network Alignment']== Network_Alignment_iterative[Network_Alignment_iterative_loop]) & (df_Parcel_Cost_final['Geo']== Geo_iterative[Geo_iterative_loop]) & (df_Parcel_Cost_final['mdates']== UPP_Changed_date) , 'CPP'].max()
                          #      df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'Cost'] = Parcel_cost_iterative * df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ) & (df2['Geo']==Geo_iterative[Geo_iterative_loop]) & (df2['Network Alignment']==Network_Alignment_iterative[Network_Alignment_iterative_loop]), 'No_of_Packages']
                                
                    
                    #day = int(thr/24)
                    
    
                    #rem_dc_cap=dc_cap.loc[dc_cap['DC']==dc,'Remaining'].max()
                    #rem_dc_cap = int(rem_dc_cap)
                    #else DC_cap = DC_cap - remaining units and set remaining units =0, set 'Time Processed'=mhr
                    
                    #update loop variables
                    total_dc_outstanding_demand=df2.loc[(df2['DC']==dc), 'Remaining Units'].sum() 
                    
                    
                    
                    
                    
                    #pdb.set_trace()
                    #df2.loc[(df2['Time to Promise']==thr) & (df2['DC']==dc ), 'Backlog'  ] = total_dc_outstanding_demand/Avg_five_day_Cap
                    thr+=1
                    ttp_list_min = df2.loc[ (df2['Remaining Units']>0) & (df2['DC']== dc) & (df2['Time to Promise']>= thr), 'Time to Promise'].min()
                    
                    thr = ttp_list_min
                    
                #End Lowest to Highest 'Time to Promise' loop
            #end While Loop
            
            Total_DC_Demand_At_End_baclog =df2.loc[(df2['DC']==dc), 'Remaining Units'].sum() 
            Total_Dc_Demand_Brand_1 = df2.loc[(df2['DC']==dc) & (df2['Brand']==1), 'Remaining Units'].sum()
            Total_Dc_Demand_Brand_2 = df2.loc[(df2['DC']==dc) & (df2['Brand']==2), 'Remaining Units'].sum()
            Total_Dc_Demand_Brand_3 = df2.loc[(df2['DC']==dc) & (df2['Brand']==3), 'Remaining Units'].sum()
            Total_Dc_Demand_Brand_4 = df2.loc[(df2['DC']==dc) & (df2['Brand']==4), 'Remaining Units'].sum() 
            Total_Dc_Demand_Brand_5 = df2.loc[(df2['DC']==dc) & (df2['Brand']==5), 'Remaining Units'].sum()
            Total_Dc_Demand_Brand_6 = df2.loc[(df2['DC']==dc) & (df2['Brand']==6), 'Remaining Units'].sum()
            Total_Dc_Demand_dummy_99 = df2.loc[(df2['DC']==dc) & (df2['Brand']==99), 'Remaining Units'].sum()
            Total_Dc_Demand_Ship_Option_1 = df2.loc[(df2['DC']==dc) & (df2['Modeled Ship Option']==1), 'Remaining Units'].sum()
            Total_Dc_Demand_Ship_Option_2 = df2.loc[(df2['DC']==dc) & (df2['Modeled Ship Option']==2), 'Remaining Units'].sum()
            Total_Dc_Demand_Ship_Option_3 = df2.loc[(df2['DC']==dc) & (df2['Modeled Ship Option']==3), 'Remaining Units'].sum()
            Total_Dc_Demand_Ship_Option_4 = df2.loc[(df2['DC']==dc) & (df2['Modeled Ship Option']==3), 'Remaining Units'].sum()
            
    
            Max_TTP_Baclog = df2.loc[ (df2['Units_Processed1']>0) & (df2['DC']== dc) , 'Time to Promise'].max()
            if isNaN(Max_TTP_Baclog):
                Max_TTP_Baclog = 0
                
            if isNaN(Total_Dc_Demand_dummy_99):
                Total_Dc_Demand_dummy_99 = 0
                
            
            Min_TTP_Baclog = df2.loc[ (df2['Units_Processed1']>0) & (df2['DC']== dc)  , 'Time to Promise'].min()
            if isNaN(Min_TTP_Baclog):
                Min_TTP_Baclog = 0
            
            Total_DC_Cap_Used_baclog = df2.loc[(df2['DC']==dc   ), 'Units_Processed1'].sum()
            if Total_DC_Cap_Used_baclog == 0:
                Total_DC_Cap_Used_baclog = 0.01
            
            Avg_Time_To_Promise_baclog = sum (df2.loc[ (df2['Units_Processed1']>0) & (df2['DC']== dc)  , 'Time to Promise'] * df2.loc[ (df2['Units_Processed1']>0) & (df2['DC']== dc)  , 'Units_Processed1'])/Total_DC_Cap_Used_baclog
            
            if isNaN(Avg_Time_To_Promise_baclog):
                Avg_Time_To_Promise_baclog = 0
            
            
            if rem_dc_cap_start_baclog == 0:
                rem_dc_cap_end_baclog = 0
                
            
            U_Process_Brand_1_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==1) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_1_baclog):
                U_Process_Brand_1_baclog = 0
                
            U_Process_Brand_2_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==2) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_2_baclog):
                U_Process_Brand_2_baclog = 0
            
            U_Process_Brand_3_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==3) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_3_baclog):
                U_Process_Brand_3_baclog = 0
                
            U_Process_Brand_4_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==4) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_4_baclog):
                U_Process_Brand_4_baclog = 0
            
            U_Process_Brand_5_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==5) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_5_baclog):
                U_Process_Brand_5_baclog = 0
                
            U_Process_Brand_6_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==6) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_6_baclog):
                U_Process_Brand_6_baclog = 0
                
            U_Process_Brand_dummy_99_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Brand']==99) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Brand_dummy_99_baclog):
                U_Process_Brand_dummy_99_baclog = 0
                
            U_Process_Ship_option_1_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Modeled Ship Option']==1) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Ship_option_1_baclog):
                U_Process_Ship_option_1_baclog = 0
            
            
            U_Process_Ship_option_2_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Modeled Ship Option']==2) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Ship_option_2_baclog):
                U_Process_Ship_option_2_baclog = 0
            
            U_Process_Ship_option_3_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Modeled Ship Option']==3) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            if isNaN(U_Process_Ship_option_3_baclog):
                U_Process_Ship_option_3_baclog = 0
            
            
            U_Process_Ship_option_4_baclog = df2.loc[ (df2['DC']==dc   ) &  (df2['Modeled Ship Option']==4) & (df2['Hour']<=mhr), 'Units_Processed1'].sum()
            
            if isNaN(U_Process_Ship_option_4_baclog):
                U_Process_Ship_option_4_baclog = 0
    
            backlog_con = (df_backlog['Hour']==mhr) & (df_backlog['DC_id']==dc )
            df_backlog.loc[backlog_con,'Baclog'] = (total_dc_outstanding_demand/Avg_five_day_Cap)
            df_backlog.loc[backlog_con,'Min_Time_To_Promise'] = Min_TTP_Baclog
            df_backlog.loc[backlog_con,'Max_Time_To_Promise'] = Max_TTP_Baclog
            df_backlog.loc[backlog_con,'Avg_Time_To_Promise'] = Avg_Time_To_Promise_baclog
            df_backlog.loc[backlog_con,'Rem_dc_Cap_Start'] = rem_dc_cap_start_baclog
            df_backlog.loc[backlog_con,'Rem_dc_Cap_End'] = rem_dc_cap_end_baclog
            df_backlog.loc[backlog_con,'Total_DC_Cap_Used'] = Total_DC_Cap_Used_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_1'] = U_Process_Brand_1_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_2'] = U_Process_Brand_2_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_3'] = U_Process_Brand_3_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_4'] = U_Process_Brand_4_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_5'] = U_Process_Brand_5_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_6'] = U_Process_Brand_6_baclog
            df_backlog.loc[backlog_con,'U_Process_Brand_dummy_99'] = U_Process_Brand_dummy_99_baclog
            
            df_backlog.loc[backlog_con,'U_Process_Ship_Option_1'] = U_Process_Ship_option_1_baclog
            df_backlog.loc[backlog_con,'U_Process_Ship_Option_2'] = U_Process_Ship_option_2_baclog
            df_backlog.loc[backlog_con,'U_Process_Ship_Option_3'] = U_Process_Ship_option_3_baclog
            df_backlog.loc[backlog_con,'U_Process_Ship_Option_4'] = U_Process_Ship_option_4_baclog
            df_backlog.loc[backlog_con,'Total_DC_Demand_At_Start'] = Total_DC_Demand_At_Start_baclog
            df_backlog.loc[backlog_con,'Total_DC_Demand_At_End'] = Total_DC_Demand_At_End_baclog
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_1'] = Total_Dc_Demand_Brand_1
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_2'] = Total_Dc_Demand_Brand_2
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_3'] = Total_Dc_Demand_Brand_3
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_4'] = Total_Dc_Demand_Brand_4
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_5'] = Total_Dc_Demand_Brand_5
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_6'] = Total_Dc_Demand_Brand_6
            df_backlog.loc[backlog_con,'Baclog_Units_Brand_dummy_99'] = Total_Dc_Demand_dummy_99
            
            df_backlog.loc[backlog_con,'Baclog_Units_Ship_Option_1'] = Total_Dc_Demand_Ship_Option_1
            df_backlog.loc[backlog_con,'Baclog_Units_Ship_Option_2'] = Total_Dc_Demand_Ship_Option_2
            df_backlog.loc[backlog_con,'Baclog_Units_Ship_Option_3'] = Total_Dc_Demand_Ship_Option_3
            df_backlog.loc[backlog_con,'Baclog_Units_Ship_Option_4'] = Total_Dc_Demand_Ship_Option_4
            df_backlog.loc[backlog_con,'Starting_Baclog_7AM'] = Starting_Baclog
            df_backlog.loc[backlog_con,'Starting_Baclog_if_Var_Cap_Used_7AM'] = Starting_Baclog_if_Var_Cap_Used_7AM # this will only be updated if variable capacity is used
            df_backlog.loc[backlog_con,'Outstanding_demand_7AM'] = Outstanding_demand_7AM
            df_backlog.loc[backlog_con,'Variable_capacity_used'] = Variable_capacity_used
            # Input from the capacity file for that DC
        #end DC Loop
        
        #Decrement 'Time to Promise' for everything  (ie simulate passage of 1 hr to unprocessed demand)
        df2.loc[df2['Remaining Units']>0, 'Time to Promise']=df2['Time to Promise']-1
        df_counting_time.loc[(df_counting_time ['Hour'] == mhr), 'End_time'] = time.time()
        #pdb.set_trace()
    df_processed = df_processed.append(df2.loc[(df2['Remaining Units']==0),:], ignore_index=True)   
    df_carryover = df2.loc[(df2['Remaining Units']>0),:]
    #df5['Modeled Order Units'] = df5['Remaining Units']
    #df5['Promise_time'] = df5['Time to Promise']
    #df4.drop(df4[df4['Remaining Units'] > 0].index, inplace=True)
    
        
        
        
        
    #Calculations/capture of statistics on backlog
    
    #end mhr  loop

#end Year Loop

#SFS Modeling Starts ################################################################# 
#df_UPP_final_SFS['mdates'] = df_UPP_final_SFS['Fis_Cal_Date'].map(mdates.date2num)  

#df_UPP_final_SFS['mdates_from'] = df_UPP_final_SFS['Fis_Cal_Date_From'].map(mdates.date2num)
#df_UPP_final_SFS['mdates_to'] = df_UPP_final_SFS['Fiscal_Date_to'].map(mdates.date2num)
#
#  
#Brand_list_for_UPP_SFS = df_UPP_final_SFS[['Brand_id','Brand', 'mdates']].drop_duplicates()
#len_Brand_list_UPP_SFS = len(Brand_list_for_UPP_SFS)
#Brand_list_for_UPP_SFS = Brand_list_for_UPP_SFS.sort_values(by=['Brand_id','mdates'])
#Brand_list_for_UPP_SFS = Brand_list_for_UPP_SFS.drop(['Brand'], axis=1)
#Brand_list_for_UPP_SFS.rename(columns={'Brand_id': 'Brand'}, inplace=True)
#df_UPP_final_SFS = df_UPP_final_SFS.drop(['Brand'], axis=1)
#df_UPP_final_SFS.rename(columns={'Brand_id': 'Brand'}, inplace=True)
#
#                        
##Order_mdate_UPP = df2.loc[df2['Hour']==mhr, 'mdates'].unique()
#Brand_loop_UPP_SFS = 0
#for Brand_loop_UPP_SFS in range(0 ,len_Brand_list_UPP_SFS):
#    Brand_UPP_SFS = Brand_list_for_UPP_SFS.iloc[Brand_loop_UPP_SFS,0]
#    mdates_UPP_SFS = Brand_list_for_UPP_SFS.iloc[Brand_loop_UPP_SFS,1]
#    df_UPP_final_iterative_SFS = df_UPP_final_SFS[(df_UPP_final_SFS['Brand'] == Brand_UPP_SFS) & (df_UPP_final_SFS['mdates'] == mdates_UPP_SFS)]
#    df_SFS = pd.merge(df_SFS, df_UPP_final_iterative_SFS[['Brand', 'UPP']], on=['Brand']  , how = 'left')
#    df_SFS.loc[(df_SFS['mdates']>= mdates_UPP_SFS ) & (df_SFS['Brand']== Brand_UPP_SFS ), 'UPP_x'] = df_SFS.loc[(df_SFS['mdates']>= mdates_UPP_SFS ) & (df_SFS['Brand']== Brand_UPP_SFS ), 'UPP_y']
#    df_SFS = df_SFS.drop(['UPP_y'], axis=1)
#    df_SFS.rename(columns={'UPP_x': 'UPP'}, inplace=True)
    
conn = sqlite3.connect("df_SFS.db")
df_SFS.to_sql("df_SFS", conn, if_exists="replace")
cur = conn.cursor()
df_UPP_final_SFS['mdates_from'] = df_UPP_final_SFS['Fis_Cal_Date_From'].map(mdates.date2num)
df_UPP_final_SFS['mdates_to'] = df_UPP_final_SFS['Fiscal_Date_to'].map(mdates.date2num)

Brand_list_for_UPP_SFS = df_UPP_final_SFS[['Brand_id','Brand', 'mdates_from', 'mdates_to','UPP']].drop_duplicates()
#len_Brand_list_UPP = len(Brand_list_for_UPP)
Brand_list_for_UPP_SFS  = Brand_list_for_UPP_SFS.sort_values(by=['Brand_id','mdates_from'])
Brand_list_for_UPP_SFS  = Brand_list_for_UPP_SFS.drop(['Brand'], axis=1)
Brand_list_for_UPP_SFS .rename(columns={'Brand_id': 'Brand'}, inplace=True)


start_time_function = time.time()
@jit
def UPP_Update_dataframe_SFS (Brand_list_for_UPP_SFS ):
    
    for row in Brand_list_for_UPP_SFS.itertuples():
        values = (row.UPP, row.Brand, row.mdates_from, row.mdates_to)
        cur.execute("update df_SFS set UPP  = ?  where Brand = ? and mdates >= ? and mdates <= ?", values)
        conn.commit()
    return  pd.read_sql_query("select * from df_SFS;", conn)




df_SFS = UPP_Update_dataframe_SFS (Brand_list_for_UPP_SFS )
end_time_function = time.time()

print ('Time was %.3f'%(end_time_function-start_time_function ) )
df_SFS  = df_SFS .drop(['index'], axis=1)
print('SFS_UPP_update_ends')
cur.close()
      
    
df_Parcel_Cost_final_SFS['mdates'] = df_Parcel_Cost_final_SFS['Fis_Cal_Date'].map(mdates.date2num)

# SFS CPP for the promise time >= 168
df_Parcel_Cost_final_SFS_168 =df_Parcel_Cost_final_SFS[df_Parcel_Cost_final_SFS['Time to Promise']==168]

df_Parcel_Cost_final_SFS_168 = df_Parcel_Cost_final_SFS_168.sort_values(by=['mdates'])

for loop_Parcel_cost_SFS_168 in range(0, len(df_Parcel_Cost_final_SFS_168)):
    #loop_Parcel_cost_SFS_168 = 1
    mdate_SFS = df_Parcel_Cost_final_SFS_168.iloc[loop_Parcel_cost_SFS_168,8]
    CPP_SFS= df_Parcel_Cost_final_SFS_168.loc[(df_Parcel_Cost_final_SFS_168['mdates']==mdate_SFS), :].iloc[0,6]
    df_SFS.loc[(df_SFS['mdates']>=mdate_SFS) & (df_SFS['Promise_time']>=168),'CPP'] = CPP_SFS * 1
    
# All Promise time CPP 
df_Parcel_Cost_final_SFS_all_promise_time =df_Parcel_Cost_final_SFS[df_Parcel_Cost_final_SFS['Time to Promise']<168]
df_Parcel_Cost_final_SFS_all_promise_time_mdates =df_Parcel_Cost_final_SFS_all_promise_time[['mdates']].drop_duplicates()
df_Parcel_Cost_final_SFS_all_promise_time_mdates = df_Parcel_Cost_final_SFS_all_promise_time_mdates.sort_values(by=['mdates'])

for loop_Parcel_cost_SFS_all_promise_Time in range(0, len(df_Parcel_Cost_final_SFS_all_promise_time_mdates)):
    #loop_Parcel_cost_SFS_all_promise_Time = 1
    mdates_iterative = df_Parcel_Cost_final_SFS_all_promise_time_mdates.iloc[loop_Parcel_cost_SFS_all_promise_Time,0]
    df_Parcel_Cost_final_SFS_iterative = df_Parcel_Cost_final_SFS_all_promise_time[(df_Parcel_Cost_final_SFS_all_promise_time['mdates'] == mdates_iterative)]
    df_SFS = pd.merge(df_SFS, df_Parcel_Cost_final_SFS_iterative[['CPP','Time to Promise']] , left_on = ['Promise_time'] , right_on = ['Time to Promise'], how = 'left')
    df_SFS.loc[(df_SFS['Promise_time'] <168) & (df_SFS['mdates'] >= mdates_iterative) , 'CPP_x'] = df_SFS.loc[(df_SFS['Promise_time'] <168) & (df_SFS['mdates'] >= mdates_iterative), 'CPP_y']    
    df_SFS = df_SFS.drop(['CPP_y', 'Time to Promise_y'], axis=1)
    df_SFS.rename(columns={'CPP_x': 'CPP'}, inplace=True)  
    df_SFS.rename(columns={'Time to Promise_x': 'Time to Promise'}, inplace=True)  

#UPP_SFS = df_UPP_final.loc[(df_UPP_final['Brand']==88),:].iloc[0,1]


#df_SFS['UPP'] = 1.44567* 1



df_SFS['Total_Cost'] = df_SFS['Modeled Order Units'] * df_SFS['CPP']/df_SFS['UPP']
df_SFS['Modeled Order Units'].sum()
#df_SFS.pivot_table(index='Fiscal_Year',columns='Total_Cost',aggfunc=sum)


#SFS Modeling Ends #################################################################  

df_processed = df_processed.append(df_SFS, ignore_index=True)   

#length_df_baclog = len(df_backlog)
#def baclog_false_0 (a, length_df_baclog):
#    for row in range (length_df_baclog+1):
#        if a == 'FALSE':
#            a = 0
#        else:
#            a
#
#vfunc2 = np.vectorize(baclog_false_0)

#df_backlog['U_Process_Brand_2'] = df_backlog['U_Process_Brand_2'] *1








df_processed.to_csv("sim_test_original_processedQ1.csv")
df_counting_time.to_csv("Processing_time.csv")
df_temp.to_csv(inputfile_path + "Capacity_Used_Model_Ouput.csv")    
    
#Calculations/capture of statistics on ship cost/Processing

#use for checking processing time of scenario
end_time=time.time()
duration = end_time-start_time


#use for checking processing time of scenario
print(duration/60)

df_simtest = pd.read_csv(inputfile_path + "sim_test_original_processedQ1.csv")



df_simtest['Modeled Order Units'].sum() 
df_carryover['Modeled Order Units'].sum() 

df_mdates= df_temp[['Cal_Date', 'Fiscal_Year', 'Fiscal_Month',  'Fiscal_Week', 'Fiscal_Day', 'mdates'  ]].drop_duplicates()
df_mdates.to_csv("mdates.csv")

df_hrkey= df_temp[['Day_Hrs_Inc', 'mdates', 'Fiscal_Year', 'Fiscal_Month',  'Fiscal_Week', 'Fiscal_Day', 'Event_Network_Model'  ]].drop_duplicates()
df_hrkey.to_csv("Hrkey.csv")

df_backlog['Hour'] = df_backlog['Hour'].astype(int)
df_backlog = pd.merge(df_backlog, df_hrkey[['Day_Hrs_Inc','mdates', 'Fiscal_Year', 'Fiscal_Month','Fiscal_Week', 'Fiscal_Day', 'Event_Network_Model']] , left_on = ['Hour'] , right_on = ['Day_Hrs_Inc'], how = 'left')
df_backlog.to_csv(inputfile_path + "BacklogQ1.csv")


Network_model_end_time_function = time.time()

print ('Time was %.3f'%(Network_model_end_time_function - Network_model_start_time_function  ) )
#df2[Day]
#df2['Day']= np.random.randint(1,365)
#list(df2.columns.values)
#df2[Hour]
#df2['Day']


