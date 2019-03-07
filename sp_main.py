import numpy as np
import pandas as pd
import glob
import json
from spanalysis import Spclass 

def read_json(JSON_FILE_PATH):
	with open(JSON_FILE_PATH) as f:
		config = json.load(f)
		return config

def main():
	#80度以上のすべてのデータを読み込む
	requirement = read_json("config.json")
	start_day = requirement["start_day"]
	stop_day = requirement["stop_day"]
	sp_type = requirement["sp_type"]
	max_latitude = requirement["max_latitude"]
	min_latitude = requirement["min_latitude"]
	max_longitude = requirement["max_longitude"]
	min_longitude= requirement["min_longitude"]
	water_absorption = requirement["water_absorption"]
	day_list = list(range(start_day, stop_day + 1))
	print("パスの読み込み中です. 2,3分ほどお待ちください. ")
	file_list1 = glob.glob("/Volumes/HD-PNFU3_1/200712/200*/data/*[S,N]8*.lbl")
	file_list2 =glob.glob("/Volumes/HD-PNFU3_[1,2,3]/20080[1,2,3,4,5,6,7,8,9]/2008*/data/*[S,N]8*.lbl")
	file_list3 =glob.glob("/Volumes/HD-PNFU3_[1,2,3]/20081[0,1,2]/2008*/data/*[S,N]8*.lbl")
	file_list4 = glob.glob("/Volumes/HD-PNFU3_3/20090[1,2,3,4,5,6]/200*/data/*[S,N]8*.lbl")
	file_list = file_list1 + file_list2 + file_list3 + file_list4
	file_list = file_list3
	#import pdb;pdb.set_trace()
	total_list = []
	i = 0
	for lbl_name in file_list:
		data_day = int(lbl_name[-43:-35])
		if data_day in day_list:
			sp = Spclass(lbl_name , sp_type = sp_type, max_latitude = max_latitude , min_latitude = min_latitude ,
			 max_longitude = max_longitude, min_longitude = min_longitude , water_absorption = water_absorption)
			total_list = total_list + sp.spectual_list
			print(lbl_name)
			print(data_day)
			print("{}/{}".format(i, len(file_list)))
		i+=1
	sp_df = pd.DataFrame(total_list)
	sp_df.columns = (["FILE_NAME", "number", "day","TI","START_UT","STOP_UT","ALTITUDE","latitude", 
		"longitude","TM_DATASET_NAME", "TM_UPPER_LEFT_LATITUDE","TM_UPPER_LEFT_LONGITUDE","TM_UPPER_RIGHT_LATITUDE",
		"TM_UPPER_RIGHT_LONGITUDE", "TM_LOWER_LEFT_LATITUDE", "TM_LOWER_LEFT_LONGITUDE","TM_LOWER_RIGHT_LATITUDE" ,
		"TM_LOWER_RIGHT_LONGITUDE","EMISSION_ANGLE","SPECECRAFT_AZIMUTH", "INCIDENCE_ANGLE", "SCLAR_AZIMUTH_ANGLE", "water_1500", "water_1280"]+list(range(296))) 
	sp_df.to_csv("/Users/hk/GoogleDrive/test_{}.csv".format(sp_type), index = False)
	
	
	import pdb;pdb.set_trace()

if __name__ == "__main__":
	main()