import json
import pandas as pd
import itertools
from collections import defaultdict, Counter
import time
import random
import math 
import dill
from sklearn.externals import joblib
from datetime import datetime as dt

"""
scikit-learn
"""

from sklearn.svm import SVC
from sklearn.linear_model import Lars
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

import numpy as np
import sys, os, shutil
from sklearn import svm
from sklearn.decomposition import PCA
#import progressbar
# see https://pypi.python.org/pypi/multiprocessing_on_dill
import multiprocessing_on_dill as multiprocessing
from multiprocessing_on_dill import Pool
from functools import wraps
import re

from sklearn.model_selection import train_test_split

def run_and_time(method):

	@wraps(method)
	def wrap_func(self,*method_args,**kwargs):

		t_start = time.time()  # in seconds
		res = method(self,*method_args,**kwargs)
		t_end = time.time()
		print(u"elapsed time (min:sec):", dt.fromtimestamp(t_end-t_start).strftime('%M:%S'))

		return res

	return wrap_func

def setup_dirs(dnames, todo="check"):

		for dname in dnames:

			if os.path.isdir("./" + dname):  # if dir exists
				if todo == "reset":  # and we need delete it and create anew 
					print("directory {} already exists. resetting...".format(dname), end="")
					shutil.rmtree(dname)
					os.mkdir(dname)
					print("ok")			
				elif todo == "check":  # only had to check if exists, do nothing
					pass
			else:  # if directory doesn't exist
				if todo == "check":
					sys.exit(">> error >> directory {} is MISSING!".format(dname))
				elif todo == "reset":
					print("creating new directory {}...".format(dname))
					os.mkdir(dname)
					print("ok")

class SportsClassifier(object):

	def __init__(self):
		
		# sports currently supported; nonsport is everything we are NOT interested in at all
		self.sports = u"nonsport afl soccer nrl rugby_union basketball netball tennis horse_racing".split()
		
		print("""
			|
			| SPORTS CLASSIFIER
			| 
			\n::: {} :::\n""".format(" ".join(self.sports[1:])))

		# specifically, team and individual sports (useful later on)
		self.SPORTS_W_TEAMS = u"afl soccer nrl rugby_union basketball netball".split()
		self.SPORTS_WO_TEAMS = u"tennis horse_racing".split()
		# encode sports like {"tennis": 3, ..}
		self.sports_encoded = {v:k for k,v in enumerate(self.sports)}
		# decode sports, e.g. {2: "soccer", ..}
		self.sports_decoded = {v: sport_name for sport_name, v in self.sports_encoded.items()}

		print("[initialisation]")

		self.PKS_DIR = "pks/"   			# primary key lists are sitting here
		self.RAW_DATA_DIR = "data_raw/"		# raw data (note: it's a gzipped CSV) 
		self.AUX_DATA_DIR = "data_aux/"		# auzilliary data to be stored here

		setup_dirs([self.PKS_DIR, self.RAW_DATA_DIR, self.AUX_DATA_DIR])		
		
		self.pks_dic = {}
		# all pk files MUST be called pks_[SPORT].txt!
		self.pks_fl_dic = {sport: self.PKS_DIR + "pks_" + sport + ".txt" for sport in self.sports} 
		self.raw_data_file = self.RAW_DATA_DIR + "db10.csv.gz"
		self.data_file = self.AUX_DATA_DIR + "data.csv"  # raw data with added sport types for each pk
		
		self.MODEL_DIR = "model/"
		self.TEMP_DATA_DIR = "data_tmp/"

		setup_dirs([self.TEMP_DATA_DIR, self.MODEL_DIR], "reset")

		self.rare_words_file = self.TEMP_DATA_DIR  + "train_rare_words.txt"
		self.predictions_test_file = self.TEMP_DATA_DIR  + "predictions_on_test.csv"
		self.model_fnames = self.MODEL_DIR + "model_features.txt"
		
		
		self.data_df = pd.DataFrame()
		
		self.train = pd.DataFrame()  # training data w features
		self.test = pd.DataFrame()  # testing data w features

		self.venue_names = defaultdict(lambda: defaultdict(list))
		self.team_names = defaultdict(lambda: defaultdict(list))
		self.team_name_words = defaultdict(list)
		self.comp_names = defaultdict(list)
		self.comp_name_words = defaultdict(list)

		# dictionary to store words from the processed training data frame
		self.train_dict = defaultdict(int)
		# model feature names - the same as the features extracted from the training set 
		self.model_feature_names = []
		self.words_once = []
		self.train_word_list = []

		self.AUS_THEATRE_COMPANIES = self.AUS_OPERA_COMPANIES = self.NAMES_M = self.NAMES_F = \
		self.STPW = self.AUS_SUBURBS = self.COUNTRIES = self.PERFORMERS = frozenset()

	def __list2file(self, filename, lst):

		with open(filename, "w") as f:
			for w in lst:
				if isinstance(w, bytes):
					print("WARNING: __list2file is trying to save BYTES!")
				f.write("{}\n".format(w))

	def __read2list(self,filename, msg):
			"""
			read lines from a text file "filename" and put them into a list; 
			show message "msg" 
			"""
			with open(self.AUX_DATA_DIR + filename, "r") as f:
				lst = [line.strip() for line in f if line.strip()]
			for w in lst:
				if isinstance(w, bytes):
					print("WARNING: __read2list just read BYTES!")
			print(msg + "...{}...ok".format(len(lst)))
			return frozenset(lst)

	def read_data(self):

		print("[reading data]")

		self.AUS_THEATRE_COMPANIES = self.__read2list("theatre_companies_australia.txt", "australian theatre companies")
		self.AUS_OPERA_COMPANIES = self.__read2list("opera_companies_australia.txt", "australian opera companies")
		self.NAMES_M = self.__read2list("names_m_5k.txt", "male names")
		self.NAMES_F = self.__read2list("names_f_5k.txt", "female names")
		self.STPW = self.__read2list("english_stopwords.txt", "english stopwords")
		# australian suburb/city list (by Australia Post, Sep 2016)
		self.AUS_SUBURBS = self.__read2list("aus_suburbs.txt", "australian suburbs")
		# load australian sports team nicknames
		self.AUS_TEAM_NICKS = self.__read2list("aus_team_nicknames.txt", "australian team nicknames")
		#print(self.AUS_TEAM_NICKS)
		# load artist/musician list (from www.rollingstone.com)
		self.PERFORMERS = self.__read2list("performer_list.txt", "artists")
		self.COUNTRIES = self.__read2list("countries.txt", "world countries")
		# dictionary to keep all sports primary keys (pks) 
		self.pks_dic = {sport: pd.read_csv(self.pks_fl_dic[sport], sep="\n", dtype=np.int32).drop_duplicates().ix[:,0].tolist() for sport in self.pks_fl_dic}
		
		# check if the pks happen to belong to some sport AND non-sport at the same time; if this is the case, remove that pk from non-sports
		print("checking for pks in multiple sports...", end="")
		for sport in self.pks_dic:
			for sport2 in self.pks_dic:
				if sport != sport2:
					cmn = set(self.pks_dic[sport]) & set(self.pks_dic[sport2])  # whick pks are in common
					if cmn:
						sys.exit("\nERROR: pks on two lists! in both {} and {}: {}".format(sport, sport2, ",".join([str(w) for w in cmn])))
		print("ok")
		
		#
		# create the data frame we will work with; it contains only the events we have pks for
		# note that we add a new column called "sport" containing sports codes speficied in self.sports_encoded
		#

		self.data_df = pd.concat([pd.read_csv(self.raw_data_file, parse_dates=["performance_time"], 
										 	date_parser=lambda x: dt.strptime(str(x), "%Y-%m-%d"), encoding="utf-8", index_col="pk_event_dim").drop_duplicates(),
										pd.DataFrame([(pk, self.sports_encoded[sport]) for sport in self.pks_dic for pk in self.pks_dic[sport]], 
											columns=["pk_event_dim", "sport"]).set_index("pk_event_dim")],
												axis=1, join="inner")
		#self.data_df["event"] = self.data_df["event"].astype(str)
		#print("concatenated")
		
		# add new columns (to be used as features)
		self.data_df = self.data_df.assign(month = pd.Series(self.data_df.performance_time.apply(lambda x: x.month)))
		#print("made montn")
		self.data_df = self.data_df.assign(weekday = pd.Series(self.data_df.performance_time.apply(lambda x: x.weekday())))
		#print("made weekday")
		#self.data_df = self.data_df.assign(hour = pd.Series(self.data_df.performance_time.apply(lambda x: x.hour)))  # Monday is 0
		#print("made hour")
		# remove columns that are now useless
		#self.data_df.drop(["performance_time"], axis=1, inplace=True)

		# save data to a CSV file
		print("saving data to {}...".format(self.data_file), end="")
		self.data_df.to_csv(self.data_file, columns=u"event venue month weekday sport".split(), compression="gzip")
		print("ok")

		"""
		create a dict of team names like {'soccer': {'england_championship': ['brighton & hove albion', 'newcastle united', 'reading',..],
											'australia_a_league': ['sydney fc', 'brisbane roar',..],..},
								  'rugby_union': {...}, ..}
		"""	

		for team_sport in self.SPORTS_W_TEAMS:
			# read the available team names to lists
				with open(self.AUX_DATA_DIR + "list_" + "tnam_" + team_sport + ".txt","r") as f_list:
					for fl in f_list:
						if fl.strip():
							with open(self.AUX_DATA_DIR + fl.strip(), "r") as f:
								self.team_names[team_sport][re.search("(?<=" + "tnam_" + ")\w+(?=.txt)",fl).group(0)] = \
																			[line.strip() for line in f if line.strip()]
		"""
		create a dictionary of team name words: {"soccer": ["sydney", "fc", "united",..], "basketball": ["bullets",..]}

		"""
		for team_sport in self.SPORTS_W_TEAMS:
			for league in self.team_names[team_sport]:
				self.team_name_words[team_sport] = list({w.strip() for team in self.team_names[team_sport][league] for w in team.split() if w not in self.STPW})

		# print(self.team_name_words)

		"""
		create venue names just like the team names above
		"""	

		for team_sport in self.SPORTS_W_TEAMS:
			# read the available team names to lists
				with open(self.AUX_DATA_DIR + "list_" + "vnam_" + team_sport + ".txt","r") as f_list:
					for fl in f_list:
						if fl.strip():
							with open(self.AUX_DATA_DIR + fl.strip(), "r") as f:
								self.venue_names[team_sport][re.search("(?<=" + "vnam_" + ")\w+(?=.txt)",fl).group(0)] = \
																			[line.strip() for line in f if line.strip()]

		"""
		create a dictionary of competition names like {"soccer": ["a-league", "asian cup",..], "nrl": [..]}
		"""																		

		for sport in self.sports:
			if sport != "nonsport":
				with open(self.AUX_DATA_DIR + "cnam_" + sport + ".txt","r") as f:
					self.comp_names[sport] = [line.strip().lower() for line in f if line.strip()]

		for sport in self.sports:
			self.comp_name_words[sport] = list({w.strip() for comp in self.comp_names[sport]  for w in comp.split() if (w.strip() not in self.STPW) and len(w.strip()) > 2})

	
	def create_train_test(self):
		"""
		split into the training and teating sets; the 

		"""
		print("[creating the training and testing sets]")

		self.train_nofeatures, self.test_nofeatures, self.y_train, self.y_test = train_test_split(self.data_df.loc[:, u"event venue month weekday".split()], self.data_df.sport, test_size=0.3, 
																stratify = self.data_df.sport, random_state=113)
		
	def __prelabel_from_list(self, st, lst,lab,min_words):

			c = set([v for w in lst for v in w.split()]) & set(st.split())

			if c:
				for s in c:
					for l in lst:
						if l.startswith(s):  # there is a chance..

							if (l in st) and ((len(l.split()) > min_words - 1) or ("-" in l)):
								st = st.replace(l,lab)
							else:
								pass
						else:
							pass
			else:
				pass

			return st

	def __remove_duplicates_from_string(self, st):
		
		ulist = []
		[ulist.append(w) for w in st.split() if w not in ulist]  # note that comprehension or not, ulist grows

		return " ".join(ulist)

	def normalize_string(self, st):
		
		st = " ".join([t for t in [w.strip(",:;") for w in st.split()] if len(t) > 1])
		st = self.__remove_duplicates_from_string(st)
		
		for sport in self.team_names:
			for comp in self.team_names[sport]:
				st = self.__prelabel_from_list(st, self.team_names[sport][comp], u"_" + sport.upper() + u"_TEAM_", 2)
		#print("after pre-labelled team names:",st)
		# for n in self.AUS_TEAM_NICKS:
			
		# 	nick = n.split("-")[0].strip()
		# 	sport = n.split("-")[1].strip()
		# 	st = self.__prelabel_from_list(st, [nick], u"_" + sport.upper() + u"_TEAM_", 1)
		#print("after pre-labelled team nicks:",st)

		for sport in self.comp_names:
			st = self.__prelabel_from_list(st, self.comp_names[sport], u"_" + sport.upper() + u"_COMPETITION_", 2)
		#print("after pre-labelled comp names:",st)
		for sport in self.venue_names:
			for comp in self.venue_names[sport]:
				st = self.__prelabel_from_list(st, self.venue_names[sport][comp], u"_SPORTS_VENUE_", 2)
		#print("after pre-labelled venue names:",st) 

		st = self.__prelabel_from_list(st, self.AUS_THEATRE_COMPANIES, u"_THEATRE_COMPANY_", 2)
		st = self.__prelabel_from_list(st, self.AUS_OPERA_COMPANIES, u"_OPERA_COMPANY_", 2)
		# st = self.__prelabel_from_list(st, self.COUNTRIES, u"_COUNTRY_", 1)
		# st = self.__prelabel_from_list(st, self.AUS_SUBURBS, u"_AUS_LOCATION_", 1)

		st = self.__prelabel_from_list(st, self.PERFORMERS, u"_ARTIST_", 2)
		
		# st = self.__prelabel_from_list(st, self.NAMES_M, u"_NAME_", 1)

		# remove stopwords
		st = " ".join([w for w in st.split() if w not in self.STPW])
		
		st = re.compile(r"(?<!\w)\w*\d+\w*(?!\w+)").sub('', st)

		# remove the multiple white spaces again
		st = re.sub(r"\s+"," ",st)

		return st	

	def normalize_df(self, df):

		df["event"] = df["event"].str.lower().str.split().str.join(" ")
		df["event"] = df["event"].str.replace("."," ")
		df["venue"] = df["venue"].str.lower().str.split().str.join(" ")
		df["event"] = [_event.replace(_venue,"") for _event, _venue in zip(df["event"],df["venue"])]

		#print("event BEFORE normalize string:",df.iloc[:4,0])
		df['event'] = df['event'].apply(lambda x: self.normalize_string(x))		
		#print("event AFTER normalize string:",df.iloc[:4,0])

		return df


	def parallelize_dataframe(self, df, func):

   		df_split = np.array_split(df, 8)
   		pool = Pool(8)
   		rr = pool.map(func, df_split)
   		df = pd.concat(rr)
   		pool.close() #  informs the processor that no new tasks will be added to the pool
   		pool.join()  # stops and waits for all of the results to be finished and collected before proceeding with the rest of the program

   		return df


	@run_and_time
	def normalize_data(self, df, k, para="no"):

		print("[normalising {} data]".format(k))

		if para == "yes":
			print("note: you chose to run normalisation in parallel")
			df = self.parallelize_dataframe(df, self.normalize_df)	
		elif para == "no":
			df = self.normalize_df(df)
			print("note: you chose not to run normalisation in parallel")
		else:
			sys.exit("please, choose yes or no regarding the parallelisation option.")

		if k == "training":

			for col in ['event', 'venue']:
				for st_as_lst in df[col].str.split():
					for w in st_as_lst:
							self.train_dict[w] += 1

			self.words_once = [w for w in self.train_dict if self.train_dict[w] < 8]

			print("found {} words that occurr only once ({}%)...".format(len(self.words_once), round(len(self.words_once)/len(self.train_dict)*100.0),1))
			
			self.__list2file(self.rare_words_file, self.words_once)

			self.train_word_list = [str(w) for w in self.train_dict if w not in self.words_once]

			print("words to be used to extract features: {}".format(len(self.train_word_list)))
		else:
			pass
		for col in ['event', 'venue']:
			df[col] = df[col].apply(lambda x: " ".join([w for w in x.split() if  w in self.train_word_list]))
		#print("event AFTER only popular words:",df.iloc[:4,0])

		df.to_csv(self.TEMP_DATA_DIR + "normalised_" + k + "_df.csv")

		return df

	def getf_special_word(self, st):

		res_dict = defaultdict()  # to keep collected features in

		for sport in self.team_name_words:

			in_both_sets = set(self.team_name_words[sport]) & set(st.lower().split()) 

			if in_both_sets:

				res_dict["@" + sport.upper() + "_team_words"] = len(in_both_sets)

		for sport in self.comp_name_words:

			in_both_sets = set(self.comp_name_words[sport]) & set(st.lower().split()) 

			if in_both_sets:

				res_dict["@" + sport.upper() + "_comp_words"] = len(in_both_sets)

		return res_dict


	def getf_1g(self, st):

		c = Counter(st.split())

		return {"@word_[{}]".format(w.strip()): c[w] for w in c}

	def getf_2g(self, st):

		res_dict = defaultdict(int)
		
		str_list = st.split()
		
		if len(str_list) > 1:
			for i, w in enumerate(str_list):
				if i > 0:
					res_dict[("@words_[{}]->[{}]").format(str_list[i-1], w)] += 1

		return res_dict

	def getf_isvs(self, st):

		if {"vs","vs.","v"} & set(st.split()):
			return {"@word_[vs]_present": 1}
		else:
			return {"@word_[vs]_present": 0}

	def getf_sportname(self, st):

		res_dict = defaultdict(int)

		for sport in "afl soccer nrl rugby_union basketball netball tennis horse_racing".split():
			if re.sub("_", " ", sport) in st:
				res_dict["@sport_[{}]_mentioned".format(sport.upper())] += 1

		return res_dict 

	def getf_nicks(self, st):

		res_dict = defaultdict(int)

		for n in self.AUS_TEAM_NICKS:	

			nick = n.split("-")[0].strip()
			sport = n.split("-")[1].strip()

			if nick in st:
				res_dict["@nick_[{}]_present".format(sport.upper())] += 1
		return res_dict

	def getf_countries(self, st):

		res_dict = defaultdict(int)

		for country in self.COUNTRIES:	
			if country in st:
				res_dict["@country"] += 1

		return res_dict
	
	def getf_names(self, st):

		res_dict = defaultdict(int)

		for name in self.NAMES_M:	
			if re.search("(?<!\w)name(?!\w)",st):
				res_dict["@name"] += 1

		return res_dict


	def getf_burbs(self, st):

		res_dict = defaultdict(int)

		for suburb in self.AUS_SUBURBS:
			if suburb in st:
				res_dict["@city_suburb"] += 1

		return res_dict

	def getf_upper(self, st):

		res_dict = defaultdict(int)

		for w in st.split():
			if w.isupper():
				res_dict[w] += 1

		return res_dict

	# def getf_timeofday(self, hour):

	# 	if (int(hour) >= 1) and (int(hour) < 12):
	# 		time_of_day = "morning"
	# 	elif  (int(hour) >= 12) and (int(hour) < 18):
	# 		time_of_day = "afternoon"
	# 	else:
	# 		time_of_day = "evening"

	# 	return {"event_timeofday_[{}]".format(time_of_day): 1}


	@run_and_time
	def get_features(self, d, k):

		"""
		extracts features from a data frame
		"""
		
		di = defaultdict(lambda: defaultdict(int))  # will keep extracted features here	
		
		print("[extracting {} features]".format(k))


		for i, s in enumerate(d.itertuples()):  #  go by rows
			
			pk = s.Index  # recall that pks are data frame index
			
			di[pk].update(self.getf_special_word(s.event))
			di[pk].update(self.getf_nicks(s.event))
			#di[pk].update(self.getf_timeofday(s.hour))
			di[pk].update(self.getf_1g(s.event))
			#di[pk].update(self.getf_2g(s.event))
			di[pk].update(self.getf_isvs(s.event))
			di[pk].update(self.getf_sportname(s.event))
			di[pk].update(self.getf_burbs(s.event))
			di[pk].update(self.getf_countries(s.event))
			di[pk].update(self.getf_names(s.event))
			di[pk].update(self.getf_upper(s.event))
	
		# merge the original data frame with a new one created from extracted features to make one feature data frame

	
		fdf = pd.concat([d[d.columns.difference(["event", "venue"])], 
						pd.DataFrame.from_dict(di, orient='index')], axis=1, join_axes=[d.index]).fillna(0)


		if k == "training":

			print("chekcing indexes:")

			if d.index.equals(self.y_train.index):

				print("yes, the training set index is OK")
			else:
				print("index of y_train:",self.y_train.index)
				print("index of d:",d.index)
				print("difference: what is in d but not in y-train:", set(list(d.index.values)) - set(list(self.y_train.index.values)))
				sys.exit("WRONG INDEX in trainng set!!")
			
			self.model_feature_names = list(fdf)
			print("total model features...{}...ok".format(len(self.model_feature_names)))
		
			self.__list2file(self.model_fnames, self.model_feature_names)


			ff = pd.concat([d,pd.DataFrame.from_dict(di, orient='index'),self.y_train.apply(lambda _: self.sports_decoded[_])], axis=1, join_axes=[d.index]).fillna(0)
			ff.to_csv(self.TEMP_DATA_DIR + "training_w_features.csv")
		
		
		


		elif k == "testing":
			"""
			now we ignore features that are not in self.model_feature_names
			"""

			print("chekcing indexes:")
			if d.index.equals(self.y_test.index):
				print("yes, the testing set index is OK")
			else:
				sys.exit("WRONG INDEX in testing set!!")

			fdf.drop(fdf.columns.difference(self.model_feature_names), axis=1, inplace=True)

			for fch in self.model_feature_names:
				if fch not in fdf.columns:
					fdf[fch] = 0

			pd.concat([d,pd.DataFrame.from_dict(di, orient='index'),self.y_test.apply(lambda _: self.sports_decoded[_])], axis=1, join_axes=[d.index]).fillna(0).to_csv(self.TEMP_DATA_DIR + "testing_w_features.csv")

		return  fdf   # returns new data frame that has feature columns attached

	@run_and_time
	def run_classifier(self):

		print("[training model]")

		from sklearn.multiclass import OneVsRestClassifier
		from sklearn.svm import LinearSVC
		from sklearn.metrics import accuracy_score
		from sklearn.model_selection import GridSearchCV

		#classifier = OneVsRestClassifier(LinearSVC(random_state=0))
		forest_parameters = {"max_features": [None, "log2"],
							"n_estimators": [200,300,500],
							"n_jobs":[2,-1],
							"max_depth":[2]}


		forest = RandomForestClassifier()
		#forest = RandomForestClassifier(max_features=None, n_estimators=200,n_jobs=-1)
	
		#classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=50))

		classifier = GridSearchCV(forest, forest_parameters)

		classifier = forest

		print("chekcing indexes in PREDICTION:")
		if self.train.index.equals(self.y_train.index):
			print("yes, the training set index is OK")
		else:
			sys.exit("WRONG INDEX in trainng set!!")

		classifier.fit(self.train, self.y_train)

		dill.dump(classifier, open("trained_model.dill","w"))

		pred_train = pd.Series(classifier.predict(self.train), index=self.train.index, name="predicted_sport")
		print("real:",self.y_train[:5])
		print("predincted:",pred_train[:5])

		print("TRAINING accuracy:", round(accuracy_score(self.y_train, pred_train),2))


		y_pred = pd.Series(classifier.predict(self.test), index=self.test.index, name="sport_p")

		print("real:", self.y_test[:5])
		print("predicted:",y_pred[:5])

		pd.concat([self.test_nofeatures,self.test,self.y_test.apply(lambda _: self.sports_decoded[_]),y_pred.apply(lambda _: self.sports_decoded[_])], axis=1).fillna(0).to_csv(self.TEMP_DATA_DIR + "testing_report.csv")

		print("TESTING accuracy:", round(accuracy_score(self.y_test, y_pred),2))
		# .apply(lambda _: self.sports_decoded[_]
		# save the predictions to file
		pd.concat([self.test, y_pred], axis=1, join="inner").to_csv(self.predictions_test_file)

		


if __name__ == '__main__':

	# initialize classifier
	cl = SportsClassifier()
	
	cl.read_data()

	cl.create_train_test()

	x_tr = cl.normalize_data(cl.train_nofeatures, "training", para="yes")

	cl.train = cl.get_features(x_tr, "training")

	x_ts = cl.normalize_data(cl.test_nofeatures, "testing", para="yes")

	cl.test = cl.get_features(x_ts, "testing")

	cl.run_classifier()











