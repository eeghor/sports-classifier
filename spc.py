from file_fncs import file2list, list2file, setup_dirs, enc_lst, dec_dict
import pandas as pd
import re
import numpy as np

from collections import defaultdict, Counter
import time
from datetime import datetime as dt
import itertools


#from sklearn.ensemble import RandomForestClassifier


#import sys, os, shutil
#from sklearn import svm

import multiprocessing_on_dill as multiprocessing
from multiprocessing_on_dill import Pool
from functools import wraps


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

class SportsClassifier(object):

	def __init__(self):
		
		# read and encode supported sports
		self.sports = file2list("config_sports.txt")
		self.sports_encoded = enc_lst(self.sports)
		self.sports_decoded = dec_dict(self.sports_encoded)
		
		print("""
			|
			| SPORTS CLASSIFIER
			| 
			\n::: {} :::\n""".format(" ".join([w for w in self.sports if w != "nonsport"])))

		print("initialisation...")

		# specifically, team and individual sports (useful later on)
		self.TM_SPORTS = u"afl soccer nrl rugby_union basketball netball".split()
		self.SPORTS_WO_TEAMS = u"tennis horse_racing".split()
		
		self.DATA_DIR = "data/"
		self.MODEL_DIR = "model/"
		self.TEMP_DATA_DIR = "data_tmp/"
		self.AUX_DATA_DIR = "data_aux/"	

		setup_dirs([self.AUX_DATA_DIR, self.DATA_DIR])		
		setup_dirs([self.TEMP_DATA_DIR, self.MODEL_DIR], "reset")

		self.data_file = self.DATA_DIR + "data.csv.gz"  

		self.rare_words_file = self.TEMP_DATA_DIR  + "train_rare_words.txt"
		self.predictions_test_file = self.TEMP_DATA_DIR  + "predictions_on_test.csv"
		self.model_fnames = self.MODEL_DIR + "model_features.txt"
		
		self.data = pd.read_csv(self.data_file, index_col="pk_event_dim", compression="gzip")
		
		self.AUS_THEATRE_COMPANIES = file2list(self.AUX_DATA_DIR + "theatre_companies_australia.txt")
		self.AUS_OPERA_COMPANIES = file2list(self.AUX_DATA_DIR + "opera_companies_australia.txt")
		self.NAMES_M = file2list(self.AUX_DATA_DIR + "names_m_5k.txt")
		self.NAMES_F = file2list(self.AUX_DATA_DIR + "names_f_5k.txt")
		self.STPW = file2list(self.AUX_DATA_DIR + "english_stopwords.txt")
		# australian suburb/city list (by Australia Post, Sep 2016)
		self.AUS_SUBURBS = file2list(self.AUX_DATA_DIR + "aus_suburbs.txt")
		# load australian sports team nicknames
		self.AUS_TEAM_NICKS = file2list(self.AUX_DATA_DIR + "aus_team_nicknames.txt")
		# load artist/musician list (from www.rollingstone.com)
		self.PERFORMERS = file2list(self.AUX_DATA_DIR + "performer_list.txt")
		self.COUNTRIES = file2list(self.AUX_DATA_DIR + "countries.txt")

		self.train = pd.DataFrame()  
		self.test = pd.DataFrame()  

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

		"""
		create a dict of team names like {'soccer': {'england_championship': ['brighton & hove albion', 'newcastle united', 'reading',..],
											'australia_a_league': ['sydney fc', 'brisbane roar',..],..},
								  'rugby_union': {...}, ..}
		"""	

		for team_sport in self.TM_SPORTS:
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
		for team_sport in self.TM_SPORTS:
			for league in self.team_names[team_sport]:
				self.team_name_words[team_sport] = list({w.strip() for team in self.team_names[team_sport][league] for w in team.split() if w not in self.STPW})

		# print(self.team_name_words)

		"""
		create venue names just like the team names above
		"""	

		for team_sport in self.TM_SPORTS:
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
		

		print("done")  # when initialisation is done
	
	def make_train_test(self, ptest=0.3):
		"""
		split data into the training and teating sets; 
		proportion ptest left for testing;
		recall that the data has columns
		pk_event_dim | event | venue | month | weekday | sport
		and we use pk_event_dim as index

		"""
		print("training and testing sets w/o extracted features...")

		features = self.data.loc[:, u"event venue month weekday".split()]
		target = self.data[u"sport"]

		# note: featyres are YET TO BE EXTRACTED from the event and venue columns
		self.train_nofeatures, self.test_nofeatures, self.y_train, self.y_test = train_test_split(features, target, test_size=ptest, stratify = target, random_state=113)
		
		nrows_train = self.train_nofeatures.shape[0]
		nrows_test  = self.test_nofeatures.shape[0]
		nrows_orig = self.data.shape[0]

		print("training set: {} rows ({}%)".format(nrows_train, round(100*nrows_train/nrows_orig,1)))
		print("testing set: {} rows ({}%)".format(nrows_test, round(100*nrows_test/nrows_orig,1)))

	# def __prelabel_from_list(self, st, lst,lab,min_words):

	# 		c = set([v for w in lst for v in w.split()]) & set(st.split())

	# 		if c:
	# 			for s in c:
	# 				for l in lst:
	# 					if l.startswith(s):  # there is a chance..

	# 						if (l in st) and ((len(l.split()) > min_words - 1) or ("-" in l)):
	# 							st = st.replace(l,lab)
	# 						else:
	# 							pass
	# 					else:
	# 						pass
	# 		else:
	# 			pass

	# 		return st

	def extract_features(self, st):
		"""
		extract various features from a string; note that the sring is first processed
		"""
	
		st_features = defaultdict(int)
	
		if st.strip():  # if the string is not empty
			
			st = st.lower()   
			# replace some characters with white spaces
			st = st.replace(","," ")
			st = st.replace(";"," ")
			# remove duplicates as in, for example, 'swans are swans are a great team team'
			ulist = []
			[ulist.append(w) for w in st.split() if w not in ulist]  # note that comprehension or not, ulist grows
			st = " ".join(ulist)
			# team name features
			for sport in self.team_names:
				for comp in self.team_names[sport]: 
					for t1m in self.team_names[sport][comp]:
						if t1m in st:
							st_features["@{}_TEAM_NAME".format(sport)] += 1
			# comp name features
			for sport in self.comp_names:
				for comp in self.comp_names[sport]:
					if comp	in st:
						st_features["@{}_COMP_NAME".format(sport)] +=1
			# team nickname features
			# recall that the list of nicknames has items like 
			# nickname - sports, e.g. 'wombats - gymnastics'
			for nickname_line in self.AUS_TEAM_NICKS:
				nick = nickname_line.split("-")[0].strip()
				sport = nickname_line.split("-")[1].strip()
				if nick and nick in st:
					st_features["@{}_NICKNAME".format(sport)] += 1

			# theatre company features
			for th_company in self.AUS_THEATRE_COMPANIES:
				if th_company in st:
					st_features["@THEATRE_COMPANY"] += 1
			# opera company features
			for op_company in self.AUS_OPERA_COMPANIES:
				if op_company in st:
					st_features["@OPERA_COMPANY"] += 1
			# names features
			for name in set(st.split()) & set(self.NAMES_M + self.NAMES_F):
				st_features["@NAME"] += 1
			# australina suburb features
			for suburb in self.AUS_SUBURBS:
				if suburb in st:
					st_features["@AUS_LOCATION"] += 1
			# performer (music) features
			for artist in self.PERFORMERS:
				if artist in st:
					st_features["@ARTIST"] += 1
			# country features
			for country in self.COUNTRIES:
				if country in st:
					st_features["@COUNTRY"] += 1
			# check for 'vs' or 'v'
			if {"v","vs"} & set(st.split()):
				st_features["@VS"] += 1
			# sport name featurea
			for sport in self.sports:
				if sport.replace("_"," ") in st:
					st_features["@SPORT_NAME_{}".format(sport)] += 1


		return st_features

	def extract_features_from_df(self, df):

		di = defaultdict(lambda: defaultdict(int))

		# create a corpus
		from sklearn.feature_extraction.text import CountVectorizer  #  implements both tokenization and occurrence counting in a single class:
		from sklearn.feature_extraction.text import TfidfTransformer
		from sklearn.feature_extraction.text import TfidfVectorizer

		transformer = TfidfTransformer(smooth_idf=False)
		
		corpus = []

		for st in itertools.chain.from_iterable([df["event"], df["venue"]]):
			corpus.append(st.lower())

		bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
		X = bigram_vectorizer.fit_transform(corpus).toarray()

		for i, s in enumerate(df.itertuples()):  
			full_descr = s.event + " " + s.venue
			if full_descr.strip():
				di[s.Index].update(self.extract_features(full_descr))

		# columns are event | venue | month | weekday | sport

		return di
			
	@run_and_time
	def extract_features_from_df_parallel(self, df, nworkers=8):

		print("extracting features...")
		
		df_split = np.array_split(df, nworkers)
		pool = Pool(nworkers)
		res_dicts = pool.map(self.extract_features_from_df, df_split)
		pool.close() #  informs the processor that no new tasks will be added to the pool
		pool.join()  # stops and waits for all of the results to be finished and collected before proceeding with the rest of 

		big_dic = defaultdict(lambda: defaultdict(int))

		# merge feature dictionaries created for data frame splits into one big dictionary
		for dic in res_dicts:
			for k,v in dic.items():
				big_dic[k] = v

		return pd.concat([pd.get_dummies(df[df.columns.difference(["event", "venue"])], prefix="@", columns=["month","weekday"]), pd.DataFrame.from_dict(big_dic, orient='index')], axis=1, join_axes=[df.index]).fillna(0.)

	def select_features(self, feature_df):

		from sklearn.feature_selection import VarianceThreshold
		print("before variance threshold have {} features".format(feature_df.shape[1]))
		selekt = VarianceThreshold(threshold=.8*(1.-.8))
		#print(pd.isnull(feature_df))

		new_feature_df = selekt.fit_transform(feature_df)
		print("after variance threshold have {} features".format(new_feature_df.shape[1]))
		print(new_feature_df)

		return new_feature_df




	# @run_and_time
	# def run_classifier(self):

	# 	print("[training model]")

	# 	from sklearn.multiclass import OneVsRestClassifier
	# 	from sklearn.svm import LinearSVC
	# 	from sklearn.metrics import accuracy_score
	# 	from sklearn.model_selection import GridSearchCV

	# 	#classifier = OneVsRestClassifier(LinearSVC(random_state=0))
	# 	forest_parameters = {"max_features": [None, "log2"],
	# 						"n_estimators": [200,300,500],
	# 						"n_jobs":[2,-1],
	# 						"max_depth":[2]}


	# 	forest = RandomForestClassifier()
	# 	#forest = RandomForestClassifier(max_features=None, n_estimators=200,n_jobs=-1)
	
	# 	#classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=50))

	# 	classifier = GridSearchCV(forest, forest_parameters)

	# 	classifier = forest

	# 	print("chekcing indexes in PREDICTION:")
	# 	if self.train.index.equals(self.y_train.index):
	# 		print("yes, the training set index is OK")
	# 	else:
	# 		sys.exit("WRONG INDEX in trainng set!!")

	# 	classifier.fit(self.train, self.y_train)

	# 	dill.dump(classifier, open("trained_model.dill","w"))

	# 	pred_train = pd.Series(classifier.predict(self.train), index=self.train.index, name="predicted_sport")
	# 	print("real:",self.y_train[:5])
	# 	print("predincted:",pred_train[:5])

	# 	print("TRAINING accuracy:", round(accuracy_score(self.y_train, pred_train),2))


	# 	y_pred = pd.Series(classifier.predict(self.test), index=self.test.index, name="sport_p")

	# 	print("real:", self.y_test[:5])
	# 	print("predicted:",y_pred[:5])

	# 	pd.concat([self.test_nofeatures,self.test,self.y_test.apply(lambda _: self.sports_decoded[_]),y_pred.apply(lambda _: self.sports_decoded[_])], axis=1).fillna(0).to_csv(self.TEMP_DATA_DIR + "testing_report.csv")

	# 	print("TESTING accuracy:", round(accuracy_score(self.y_test, y_pred),2))
	# 	# .apply(lambda _: self.sports_decoded[_]
	# 	# save the predictions to file
	# 	pd.concat([self.test, y_pred], axis=1, join="inner").to_csv(self.predictions_test_file)

		


if __name__ == '__main__':

	# initialize classifier
	cl = SportsClassifier()
	
	cl.make_train_test()

	print(cl.extract_features("st kilda recently bought a defender from sydney fc but $50,000 was not cheap for an a-league player; kookaburras however travelled to China - clearly, Anita sent Amit away with that contract, as Josh confirmed to Alex --- India v Sri LAnka today, then followed by Russian Federation vs. USA; another rugby union team here"))

	dfx = cl.extract_features_from_df_parallel(cl.train_nofeatures)
	print(dfx.columns.values)
	new_dfx = cl.select_features(dfx)

	# x_tr = cl.normalize_data(cl.train_nofeatures, "training", para="yes")

	# cl.train = cl.get_features(x_tr, "training")

	# x_ts = cl.normalize_data(cl.test_nofeatures, "testing", para="yes")

	# cl.test = cl.get_features(x_ts, "testing")

	# cl.train = cl.select_features(cl.train)

	# cl.run_classifier()











