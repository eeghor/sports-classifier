"""
CREATE AN INPUT DATA FILE FOR SPORTS CLASSIFIER
--- note:
the resulting data table will have columns as below
pk_event_dim | event | venue | month | weekday | sport

"""

from file_fncs import file2list, list2file, setup_dirs, enc_lst, dec_dict
import sys
import numpy as np
import pandas as pd
from datetime import date, datetime

raw_data_file, data_file = sys.argv[1:3]

sports = file2list("config_sports.txt")
sports_encoded = enc_lst(sports)
sports_decoded = dec_dict(sports_encoded)
# weekdays as numbers
wkdays_decoded = {1: "monday", 2: "tuesday",  3: "wednesday",
					4: "thursday", 5: "friday", 6: "saturday", 7: "sunday"}

PKS_DIR = "pks/"   			# primary key lists are sitting here
RAW_DATA_DIR = "data_raw/"	# raw data (note: it's a gzipped CSV) 
DATA_DIR = "data/"

setup_dirs([PKS_DIR, RAW_DATA_DIR])	
setup_dirs([DATA_DIR], "reset")	

data_file_full_path = DATA_DIR + data_file
		
# all pk files MUST be called pks_[SPORT].txt!
where_pks_dict = {sport: PKS_DIR + "pks_" + sport + ".txt" for sport in sports} 

# read primary keys from files for each sport and then
# create dict {"soccer": [pk1,pk2,..], "afl": [..],..} 
pks_dic = {sport: pd.read_csv(where_pks_dict[sport], sep="\n", dtype=np.int32).drop_duplicates().ix[:,0].tolist() for sport in where_pks_dict}
		
# check if aany primary keys happen to belong to multiple sports (which is not allowed!)

for sport in pks_dic:
	for sport2 in pks_dic:
		if sport != sport2:
			cmn = set(pks_dic[sport]) & set(pks_dic[sport2])  # whick pks are in common
			if cmn:
				sys.exit("\n>> error >> primary keys on TWO lists, in both {} and {}: {}".format(sport, sport2, ",".join([str(w) for w in cmn])))
		
"""
create the (input) data frame we will work with thay contains only the events we have pks for;
note that we add a new column called "sport" containing sports codes speficied in sports_encoded
"""

data_df = pd.concat([pd.read_csv(raw_data_file, parse_dates=["performance_time"], 
										 	date_parser=lambda _: datetime.strptime(_,"%Y-%m-%d"), encoding="utf-8", index_col="pk_event_dim").drop_duplicates(),
										pd.DataFrame([(pk, sport) for sport in pks_dic for pk in pks_dic[sport]], 
											columns=["pk_event_dim", "sport"]).set_index("pk_event_dim")],
												axis=1, join="inner")
# add an extra column, month (e.g. April)
data_df = data_df.assign(month = pd.Series(data_df.performance_time.apply(lambda _: _.strftime("%B").lower())))
# another new column weekday
data_df = data_df.assign(weekday = pd.Series(data_df.performance_time.apply(lambda _: wkdays_decoded[_.isoweekday()])))

# last step: save data to a CSV file
print("saving data to {}...".format(data_file_full_path), end="")
data_df.to_csv(data_file_full_path, columns=u"event venue month weekday sport".split(), compression="gzip", encoding="utf-8")
print("ok")