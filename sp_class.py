import pandas as pd
import numpy as np
import argparse
import sys
import os
import re
import matplotlib.pyplot as plt
import cv2
import math


class Spclass:
	def __init__(self, label_data, sp_type = "ref1" ,max_latitude = 90, min_latitude = 80, max_longitude = 360, min_longitude = 0, water_absorption = False):
		self.label_name = label_data
		self.max_latitude = max_latitude
		self.min_latitude = min_latitude
		self.max_longitude = max_longitude
		self.min_longitude = min_longitude
		self.water_absorption = water_absorption
		self.water_ab_amount = 0.05
		self.__getdata__(sp_type= sp_type)
		

	def __getdata__(self, sp_type):
		SCALING_FACTOR_WAV = 0.100000#値は整数値で入っているので、係数をかけなければならない。
		SCALING_FACTOR_RAD = 0.010000
		SCALING_FACTOR_RAW = 1
		SCALING_FACTOR_ref = 0.000100	
		self.absorption_OH=[]#リストの定義
		self.spectual_list =[]#リストの
		ans_wavelist = []#欲しかったスペクトルデータが入っているlist
		count = 0.0#欲しいデータが何個あるか
		self.sum_wave = np.zeros(296,float)
		argvs = sys.argv
		self.time_list =[]
		filepath = self.label_name
		basedir = os.path.dirname(filepath )#args.sp_l2c_labelはファイル名
		params, pointers = self.get_params(filepath)#パラメーター、ポインターをファイルから取得する。
		self.size = int(params["NORMAL_SP_POINT_NUM"])
		try:
			self.TM_lati_list = []
			self.TM_lati_list.append(params["TM_UPPER_LEFT_LATITUDE"])
			self.TM_lati_list.append(params["TM_UPPER_LEFT_LONGITUDE"])
			self.TM_lati_list.append(params["TM_UPPER_RIGHT_LATITUDE"])
			self.TM_lati_list.append(params["TM_UPPER_RIGHT_LONGITUDE"])
			self.TM_lati_list.append(params["TM_LOWER_LEFT_LATITUDE"])
			self.TM_lati_list.append(params["TM_LOWER_LEFT_LONGITUDE"])
			self.TM_lati_list.append(params["TM_LOWER_RIGHT_LATITUDE"])
			self.TM_lati_list.append(params["TM_LOWER_RIGHT_LONGITUDE"])
			TM_DATASET_NAME = params["TM_DATA_SET_NAME"]
			self.start_time = params["START_TIME"]
			self.stop_time =  params["STOP_TIME"]
			self.altitude = params["SPACECRAFT_ALTITUDE"]
			wav = self.get_spectrum_data1(basedir, pointers['^SP_SPECTRUM_WAV'], '>u2', 296)#波長の長さ
			raw_dn =  self.get_spectrum_data2(basedir, pointers['^SP_SPECTRUM_RAW'], '>u2', 296, int(params['NORMAL_SP_POINT_NUM']))
			ref1 = self.get_spectrum_data2(basedir, pointers['^SP_SPECTRUM_REF1'], '>u2', 296, int(params['NORMAL_SP_POINT_NUM']))
			rad = self.get_spectrum_data2(basedir, pointers['^SP_SPECTRUM_RAD'], '>u2', 296, int(params['NORMAL_SP_POINT_NUM']))
			#import pdb;pdb.set_trace()	
			self.time = self.get_time(basedir, pointers["^ANCILLARY_AND_SUPPLEMENT_DATA"], pointers["^SP_SPECTRUM_WAV"][1] - 1,int(params["NORMAL_SP_POINT_NUM"]))
				#radには83個のデータが要素となるリスト型となっている。	
		except:
			print("ファイルが存在しません")	
		self.get_LATITUDE(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))
		self.get_LONGITUDE(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))			
		self.get_EMISSION_ANGLE(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))	
		self.get_SPECECRAFT_AZIMUTH(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))	
		self.get_INCIDENCE_ANGLE(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))	
		self.get_SCLAR_AZIMUTH_ANGLE(basedir, pointers['^ANCILLARY_AND_SUPPLEMENT_DATA'], pointers['^SP_SPECTRUM_WAV'][1] - 1,int(params['NORMAL_SP_POINT_NUM']))	
		#try:
		#import pdb;pdb.set_trace()
		if sp_type == "ref":
			self.check_data(SCALING_FACTOR_ref*np.array(ref1), filepath ,filepath[-43:-35], filepath[-29:],params["NORMAL_SP_POINT_NUM"], TM_DATASET_NAME, self.TM_lati_list)				
		elif sp_type == "rad":
			self.check_data(SCALING_FACTOR_RAD*np.array(rad), filepath, filepath[-43:-35], filepath[-29:],params["NORMAL_SP_POINT_NUM"], TM_DATASET_NAME, self.TM_lati_list)
		elif sp_type == "DN":
			self.check_data(SCALING_FACTOR_RAW*np.array(raw_dn), filepath, filepath[-43:-35], filepath[-29:],params["NORMAL_SP_POINT_NUM"], TM_DATASET_NAME, self.TM_lati_list)				
		else:
			print("指定されたモードは存在しません")
			sys.exit(1)
		count+=1
		#except:
		#	print("no file")
			#print(i)
		#print("指定範囲内のデータは"+str(int(count))+"個存在します。")

	def check_data(self, data, filepath, day, f_name,max_number, TM_DATASET_NAME, TM_lati_list):
		ans_array = np.zeros((self.size, 23+295), dtype = "<U40")
		ans_array[:,0] = f_name
		ans_array[:,1] = f_name[9:14]
		ans_array[:,2] = int(day)
		ans_array[:,3] = self.time
		ans_array[:,4] = self.start_time 
		ans_array[:,5] = self.stop_time
		ans_array[:,6] = self.altitude
		ans_array[:,7] = self.get_latitude_list
		ans_array[:,8] = self.get_longitude_list
		ans_array[:,9] = TM_DATASET_NAME
		ans_array[:,10:18] = TM_lati_list[:]
		ans_array[:,18] = self.get_EMISSION_ANGLE_list
		ans_array[:,19] = self.get_SPECECRAFT_AZIMUTH_list
		ans_array[:,20] = self.get_INCIDENCE_ANGLE_list
		ans_array[:,21] = self.get_SCLAR_AZIMUTH_ANGLE_list
		ans_array[:,22:] = data[:,:]#750nmにおける強度を格納
		good_array = ans_array[(data[:,:184].min(axis = 1) > 0)&(data[:,40]< 10)]
		for i in range(good_array.shape[0]):
			if (np.abs(float(good_array[i,7])) >self.min_latitude and np.abs(float(good_array[i,7])) <self.max_latitude
				and np.abs(float(good_array[i,8])) >self.min_longitude and np.abs(float(good_array[i,8])) <self.max_longitude):
				#print("match!")
				ave_1200 = good_array[i, 29+115:34+115].astype(float).mean()
				ave_1280 = good_array[i, 39+115:44+115].astype(float).mean()
				ave_1360 = good_array[i, 49+115:54+115].astype(float).mean()
				ave_1436 = good_array[i, 58+115:63+115].astype(float).mean()
				ave_1500 = good_array[i, 66+115:71+115].astype(float).mean()
				ave_1564 = good_array[i, 74+115:79+115].astype(float).mean()#1580
				ave_1436_1564  = (ave_1436 + ave_1564)/2
				ave_1200_1360 = (ave_1200 + ave_1360)/2
				if ave_1500 < 1- self.water_ab_amount * ave_1436_1564:
					kari = np.insert(good_array[i], 22, ave_1500/ave_1436_1564)
					final_array = np.insert(kari, 23, ave_1280/ave_1200_1360)
					print("water!")
					self.spectual_list.append(final_array)
				elif self.water_absorption == False:
					kari = np.insert(good_array[i], 22, 0)
					final_array = np.insert(kari, 23, 0)
					self.spectual_list.append(final_array)
		


	def get_params(self,label):
		pat = re.compile('\((.*),(.*) <BYTES>\)')
		params = {}
		pointers = {}
		with open(label, 'r') as f:
			for line in f:
				if '=' in line:
					data = [x.strip() for x in line.strip().split('=')]
					val = data[1].replace('"', '')
					params[data[0]] = val
				if data[0][0] == '^':
					m = pat.search(val)
					pointers[data[0]] = [m.group(1), int(m.group(2))]
		return params, pointers
	
	def get_time(self, basedir, pointer, cnt,raw):#補助データを読み込むための関数
		with open(basedir+"/" + pointer[0], "rb") as f:
			time_list =[]
			for i in range(raw):
				f.seek(pointer[1] - 1+166*i)
				data = np.fromfile(f,">f8")
				time_list.append(data[0])
			return time_list

	
	def get_spectrum_data1(self, basedir, pointer, type, cnt):#wavの値をファイルからよみこむ。
	    with open(basedir +"/"+pointer[0], 'r') as f:
	        f.seek(pointer[1] - 1)
	        data = np.fromfile(f, dtype=type, count=cnt)
	        return data

	
	
	def get_spectrum_data2(self, basedir, pointer, type, cnt,raw):#wav以外の値をファイルから読み込む。
	    with open(basedir +"/"+ pointer[0], 'r') as f:
	        f.seek(pointer[1] - 1)
	        data = np.fromfile(f, dtype=type, count=25160)
	        S_data = []
	        for i in range(raw):
	            S_data.append(data[296*(i):296*(i+1)])
	        return S_data

	def sp_plot1(self, x, y, i):#wavの値をグラフにプロットする
		vis = y[0:80]
		nir1 = y[90:184]
		#nir2 = y[184:230]
		plt.plot(x[0:80], vis, color='blue')
		plt.plot(x[90:184], nir1, color='green')
		#plt.plot(x[184:230], nir2, color='red')
		plt.plot(x[149],nir1[59],color = 'red',marker = 'o')
		plt.plot(x[148],nir1[58],color = 'red',marker = 'o')
		plt.savefig("/Users/hk/GoogleDrive/jaxa/report/2018/6-15/png/200807_{:d}.png".format(i))
		print(i)
		plt.clf()
		#plt.show()
		#plt.plot(x[184:], nir2, color='red')

	def threshold(self, data, num):#603nm,705nm,801nmでnum以上のスペクトルのみをリスト化する関数
		if data[15] > num and data[32] > num and data[48] > num:
			try:
				self.threshold_list.append(data)
				print('リストに追加しました。')
			except:
				self.threshold_list =[]
				print('リストを作成します。')
	
	def make_histgram(self, data):#ヒストグラムを作成するための関数
		value = (data[15]  + data[32] + data[48])*0.01/3
		#import pdb;pdb.set_trace()
		try:
			self.hist_list.append(value)
		except:
			print("hist_listを作成します。")
			self.hist_list =[]



	def plot_img(self, latitude, longitude, direction ,mode):#緯度、経度、SorN、plotする色を指定。
		self.N_img = cv2.imread("/Users/hk/GoogleDrive/jaxa/N_maru.jpg")
		self.S_img = cv2.imread("/Users/hk/GoogleDrive/jaxa/S_maru.jpg")
		print(latitude, longitude)
		pi = 3.141592
		latitude = np.abs(latitude)#絶対値を使う。
		r = (90 - latitude)*40
		x = int(500 + r * math.cos(longitude*pi/180)) 
		y = int(500 + r * math.sin(longitude*pi/180)) 
		print(x,y)
		if mode == 1:#1450nm付近に吸収をもつスペクトルが存在する場合.
			if direction == "S":
				self.S_img[x:x+3, y:y+3,[1,2]] = 0
			elif direction == "N":
				self.N_img[x:x+3, y:y+3,[1,2]] = 0

		
	def get_LATITUDE(self, basedir, pointer, cnt,raw):#緯度を読み込むための関数
		with open(basedir+"/" + pointer[0], 'r') as f:
			self.get_latitude_list =[]
			for i in range(raw):
				f.seek(pointer[1] -1+166*i + 80)
				data = np.fromfile(f,'>f8')
				self.get_latitude_list.append(data[0])
			return self.get_latitude_list

	def get_LONGITUDE(self, basedir, pointer, cnt,raw):#経度を読み込むための関数
	    with open(basedir+"/" + pointer[0], 'r') as f:
	        self.get_longitude_list =[]
	        for i in range(raw):
	            f.seek(pointer[1] -1 +166*i + 88)#77 89
	            data = np.fromfile(f,'>f8')
	            self.get_longitude_list.append(data[0])
	        return self.get_longitude_list

	def get_EMISSION_ANGLE(self, basedir, pointer, cnt,raw):#センサ観測幾何学条件(天頂角)
	    with open(basedir+"/" + pointer[0], 'r') as f:
	        self.get_EMISSION_ANGLE_list =[]
	        for i in range(raw):
	            f.seek(pointer[1] -1 +166*i + 96)#77 89
	            data = np.fromfile(f,'>f4')
	            self.get_EMISSION_ANGLE_list.append(data[0])
	        return self.get_EMISSION_ANGLE_list

	def get_SPECECRAFT_AZIMUTH(self, basedir, pointer, cnt,raw):#センサ観測幾何学条件(方位角)
	    with open(basedir+"/" + pointer[0], 'r') as f:
	        self.get_SPECECRAFT_AZIMUTH_list =[]
	        for i in range(raw):
	            f.seek(pointer[1] -1 +166*i + 100)#77 89
	            data = np.fromfile(f,'>f4')
	            self.get_SPECECRAFT_AZIMUTH_list.append(data[0])
	        return self.get_SPECECRAFT_AZIMUTH_list

	def get_INCIDENCE_ANGLE(self, basedir, pointer, cnt,raw):#太陽照射幾何学条件(天頂角)
	    with open(basedir+"/" + pointer[0], 'r') as f:
	        self.get_INCIDENCE_ANGLE_list =[]
	        for i in range(raw):
	            f.seek(pointer[1] -1 +166*i + 104)#77 89
	            data = np.fromfile(f,'>f4')
	            self.get_INCIDENCE_ANGLE_list.append(data[0])
	        return self.get_INCIDENCE_ANGLE_list

	def get_SCLAR_AZIMUTH_ANGLE(self, basedir, pointer, cnt,raw):#太陽照射幾何学条件（方位角）
	    with open(basedir+"/" + pointer[0], 'r') as f:
	        self.get_SCLAR_AZIMUTH_ANGLE_list =[]
	        for i in range(raw):
	            f.seek(pointer[1] -1 +166*i + 108)#77 89
	            data = np.fromfile(f,'>f4')
	            self.get_SCLAR_AZIMUTH_ANGLE_list.append(data[0])
	        return self.get_SCLAR_AZIMUTH_ANGLE_list