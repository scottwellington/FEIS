import pandas as pd

#############################################
### Supply label and channel information: ###
#############################################

channels = ['F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4']
trial_labels = ['fleece', 'trap', 'sh', 'v', 'p', 'n', 'm', 'z', 'goose', 'k', 's', 'zh', 't', 'ng', 'f', 'thought']

header_openvibe = ['Time:256Hz','Epoch','F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4','Event Id','Event Date','Event Duration']
header = ['Time:256Hz','Epoch','F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4','Label','Stage','Flag']

hearing = pd.read_csv('stimuli.csv')
thinking = pd.read_csv('thinking.csv')
speaking = pd.read_csv('speaking.csv')

hearing.columns = header
thinking.columns = header
speaking.columns = header

#################################################################################################
### Set functions to process OpenVibe event-related columns to relevant time/epoch/event data ###
#################################################################################################

def average_data(stage, seconds_per_epoch = 0):
	i = 0
	s = pd.DataFrame(columns=channels)
	for epoch in range (0,256*seconds_per_epoch):
		s = pd.concat([s,pd.DataFrame(stage[i::int(len(stage['F3'])/10)].mean(axis = 0)).T])
		i += 1
	return s

def time(stage):
	i = 0
	s = []
	for x in range(0,len(stage['F3'])):
		s.append(i)
		i = i + 1/256
	return s
	
def epoch(stage):
	i = 0
	s = []
	for x in range(0,int(len(stage['F3']/32)/32)):
		for xx in range(0,32):
			s.append(i)
		i = i + 1
	return s
	
def misc(df, p, s):
	df.insert(0, 'Time:256Hz', time(df))
	df.insert(1, 'Epoch', epoch(df))
	df['Event Id'] = [p] * len(df['F3'])
	df['Event Date'] = [s] * len(df['F3'])
	df['Event Duration'] = ['n/a'] * len(df['F3'])

################################################################################
### Create separate dataframes (.csvs) for each separate condition (phoneme) ###
################################################################################

hearing_label = hearing.loc[hearing['Label'] == 'fleece', channels]
thinking_label = thinking.loc[thinking['Label'] == 'fleece', channels]
speaking_label = speaking.loc[speaking['Label'] == 'fleece', channels]

hearing_average = average_data(hearing_label, seconds_per_epoch = 5)
thinking_average = average_data(thinking_label, seconds_per_epoch = 5)
speaking_average = average_data(speaking_label, seconds_per_epoch = 5)

misc(hearing_average, 'fleece', 'stimuli')
misc(thinking_average, 'fleece', 'thinking')
misc(speaking_average, 'fleece', 'speaking')

#########################################################################################
### Create total dataframe (.csv) combining all separate condition (phoneme) averages ###
#########################################################################################

for phoneme in trial_labels[1:]:

	hearing_label = hearing.loc[hearing['Label'] == phoneme, channels]
	thinking_label = thinking.loc[thinking['Label'] == phoneme, channels]
	speaking_label = speaking.loc[speaking['Label'] == phoneme, channels]

	hearing_average_next = average_data(hearing_label, seconds_per_epoch = 5)
	thinking_average_next = average_data(thinking_label, seconds_per_epoch = 5)
	speaking_average_next = average_data(speaking_label, seconds_per_epoch = 5)
	
	misc(hearing_average_next, phoneme, 'stimuli')
	misc(thinking_average_next, phoneme, 'thinking')
	misc(speaking_average_next, phoneme, 'speaking')
	
	hearing_average = pd.concat([hearing_average, hearing_average_next])
	thinking_average = pd.concat([thinking_average, thinking_average_next])
	speaking_average = pd.concat([speaking_average, speaking_average_next])

################################################################################################
###  Reconstruct experiment with false time and epoch values for OpenViBE processing only:   ###
################################################################################################

hearing_average['Time:256Hz'] = time(hearing_average)   # Overwrite the timestamps (easier for OpenVibe to process contiguous timestamps)
thinking_average['Time:256Hz'] = time(thinking_average)
speaking_average['Time:256Hz'] = time(speaking_average)

hearing_average['Epoch'] = epoch(hearing_average) # Overwrite the epoch labels (easier for OpenVibe to process 1 epoch per second)
thinking_average['Epoch'] = epoch(thinking_average)
speaking_average['Epoch'] = epoch(speaking_average)

hearing_average.to_csv('hearing_average.csv', index=False)
thinking_average.to_csv('thinking_average.csv', index=False)
speaking_average.to_csv('speaking_average.csv', index=False)