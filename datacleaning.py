import numpy as np
import pandas as pd
from houseprice_functions import ratings_to_ord

HousePrices = pd.read_csv("data/train.csv")
ratings_to_ord(df = HousePrices,col = 'HeatingQC',inplace = True)

#read_csvs
HousePrices = pd.read_csv("data/train.csv")

#electrical | converted to ordinal | imputing NA to 0
elec_ = {'SBrkr':5, 'FuseF':3, 'FuseA':4, 'FuseP':2, 'Mix':1, 'Na':0}
HousePrices.Electrical = HousePrices.Electrical.fillna('Na')
HousePrices.Electrical = HousePrices.Electrical.apply(lambda x: list(elec_.values())[list(elec_.keys()).index(x)])

#Central Air | converted to bool | NO missingness
HousePrices.CentralAir = HousePrices.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)

#Heating | Bool Gas or not | No missingness
heating_ = {"GasA":1,"GasW":1,"Floor":0,"Grav":0,"OthW":0,"Wall":0}
HousePrices.Heating = HousePrices.Heating.apply(lambda x: list(heating_.values())[list(heating_.keys()).index(x)])

#HeatingQC | Ordinal Categorical | No missingness
ratings_to_ord(df = HousePrices,col = 'HeatingQC',inplace = True)

#### Garage ####
#Garage Quality and condition comb | ordingal catergorical | Na Mapped to 0 assuming NA are no garage
ratings_to_ord(df = HousePrices,col = 'GarageQual',inplace = True)
ratings_to_ord(df = HousePrices,col = 'GarageCond',inplace = True)
HousePrices['garage_score'] = HousePrices.GarageCond + HousePrices.GarageQual

#Garage Finish - UNUSED
gfin = {"Fin":1,"RFn":1,"Unf":0,"Na":0}
HousePrices.GarageFinish = HousePrices.GarageFinish.fillna('Na')
HousePrices.GarageFinish = HousePrices.GarageFinish.apply(lambda x: list(gfin.values())[list(gfin.keys()).index(x)])

#Garage Type | Dummified - dropping 'Attchd'| NA converted to no garage
HousePrices.GarageType = HousePrices.GarageType.fillna('No_garage')
garage_type_dummy = pd.get_dummies(HousePrices.GarageType).drop('Attchd',axis = 1)

#### Basement ####
# BsmtUnfSF Finished/ unfished basementv | percent between 0 and 1 | if Na, zero percent
HousePrices['finishedbsmt'] = 1 - HousePrices['BsmtUnfSF']/HousePrices['TotalBsmtSF']
HousePrices['finishedbsmt'] = HousePrices['finishedbsmt'].fillna(0) #to avoid divide by zero error

##### 
#Fence | Dummy - dropped no fence | imputed NA to mean no fence
fence_dict = {'MnPrv':'b_fence','MnWw':'b_fence','Na':'n_fence','GdWo':'g_fence','GdPrv':'g_fence'}
HousePrices.Fence = HousePrices.Fence.fillna('Na')
HousePrices.Fence = HousePrices.Fence.apply(lambda x: list(fence_dict.values())[list(fence_dict.keys()).index(x)])
fence_dummy = pd.get_dummies(HousePrices.Fence).drop('n_fence',axis = 1)

cleaned_columns = HousePrices[['CentralAir','HeatingQC','garage_score','Heating','Electrical',\
                               'GarageArea','TotalBsmtSF','finishedbsmt']]
cleaned_columns = cleaned_columns.merge(garage_type_dummy,how = "outer",left_index=True,right_index=True)
cleaned_columns = cleaned_columns.merge(fence_dummy,how = "outer",left_index=True,right_index=True)


hp_clean = cleaned_columns.to_csv('data/cleaned_hp',index = False)



### Notes ###
# TotalBsmtSF - nothing done to it
# GarageCars / GarageArea - Garage Area only
# g_fence is good fence b_fence is bad fence

#garageT_ = {'Attchd':, 'Detchd':1, 'BuiltIn':2, 'CarPort':1, 'Na':0, 'Basment':2, '2Types':2}
#HousePrices.GarageType = HousePrices.GarageType.fillna('Na')
#HousePrices.GarageType = HousePrices.GarageType.apply(lambda x: list(garageT_.values())[list(garageT_.keys()).index(x)])