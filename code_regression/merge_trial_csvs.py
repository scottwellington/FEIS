import pandas as pd

#############################################
### Supply label and channel information: ###
#############################################

trial_1_labels = ['goose','thought','zh','p','sh','n','k','fleece','trap','s','ng','t','f','z','m','v','v','ng','s','k','trap','thought','sh','zh','z','m','fleece','t','p','goose','f','n','f','trap','m','goose','z','fleece','t','k','p','v','ng','sh','thought','n','s','zh','goose','fleece','n','t','v','trap','z','k','zh','f','sh','s','ng','thought','m','p']
trial_2_labels = ['ng','m','sh','goose','trap','fleece','v','k','f','thought','t','p','z','n','zh','s','ng','fleece','trap','thought','f','m','sh','s','goose','p','n','v','z','t','zh','k','zh','fleece','goose','p','t','k','z','v','sh','m','s','ng','f','n','trap','thought']
trial_3_labels = ['k','trap','t','v','fleece','zh','p','f','n','m','sh','z','thought','ng','goose','s','n','sh','s','t','p','k','m','v','goose','ng','trap','z','f','fleece','thought','zh','trap','fleece','goose','f','k','zh','n','p','m','z','ng','thought','v','t','sh','s']

header_openvibe = ['Time:256Hz','Epoch','F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4','Event Id','Event Date','Event Duration']
header = ['Time:256Hz','Epoch','F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4','Label','Stage','Flag']

stimuli_1 = pd.read_csv('stimuli_1.csv')
articulators_1 = pd.read_csv('articulators_1.csv')
thinking_1 = pd.read_csv('thinking_1.csv')
speaking_1 = pd.read_csv('speaking_1.csv')
resting_1 = pd.read_csv('resting_1.csv')
stimuli_2 = pd.read_csv('stimuli_2.csv')
articulators_2 = pd.read_csv('articulators_2.csv')
thinking_2 = pd.read_csv('thinking_2.csv')
speaking_2 = pd.read_csv('speaking_2.csv')
resting_2 = pd.read_csv('resting_2.csv')
stimuli_3 = pd.read_csv('stimuli_3.csv')
articulators_3 = pd.read_csv('articulators_3.csv')
thinking_3 = pd.read_csv('thinking_3.csv')
speaking_3 = pd.read_csv('speaking_3.csv')
resting_3 = pd.read_csv('resting_3.csv')

#################################################################################################
### OpenVibe recorded 2 seconds' worth here; we only want 1 second, so remove erroneous data: ###
#################################################################################################

def clip_articulators(stage):

	i = 0 # exclusive 256Hz
	s = []
	for x in range(0,int((len(stage['Time:256Hz'])/256)/2)):
		s.append(stage[i:i+256])
		i = i + 512
	return s
	
articulators_1 = pd.concat(clip_articulators(articulators_1))
articulators_2 = pd.concat(clip_articulators(articulators_2))
articulators_3 = pd.concat(clip_articulators(articulators_3))

##################################################################################################
### Assert that expected iterations of data have been recorded in all trials (sanity checking) ###
##################################################################################################

trial_1_no_of_epochs = ((
len(stimuli_1['Time:256Hz']) +
len(articulators_1['Time:256Hz']) +
len(thinking_1['Time:256Hz']) +
len(speaking_1['Time:256Hz']) +
len(resting_1['Time:256Hz'])
) / 256 # Cycles per second
) / 21 # no of seconds in an epoch

trial_2_no_of_epochs = ((
len(stimuli_2['Time:256Hz']) +
len(articulators_2['Time:256Hz']) +
len(thinking_2['Time:256Hz']) +
len(speaking_2['Time:256Hz']) +
len(resting_2['Time:256Hz'])
) / 256 # Cycles per second
) / 21 # no of seconds in an epoch

trial_3_no_of_epochs = ((
len(stimuli_3['Time:256Hz']) +
len(articulators_3['Time:256Hz']) +
len(thinking_3['Time:256Hz']) +
len(speaking_3['Time:256Hz']) +
len(resting_3['Time:256Hz'])
) / 256 # Cycles per second
) / 21 # no of seconds in an epoch

assert int(trial_1_no_of_epochs) == len(trial_1_labels)
assert int(trial_2_no_of_epochs) == len(trial_2_labels)
assert int(trial_3_no_of_epochs) == len(trial_3_labels)

#####################################################################################
### Set redundant OpenVibe event-related columns to relevant stage and label data ###
#####################################################################################

stimuli_1.columns = header
articulators_1.columns = header
thinking_1.columns = header
speaking_1.columns = header
resting_1.columns = header

stimuli_2.columns = header
articulators_2.columns = header
thinking_2.columns = header
speaking_2.columns = header
resting_2.columns = header

stimuli_3.columns = header
articulators_3.columns = header
thinking_3.columns = header
speaking_3.columns = header
resting_3.columns = header

stimuli_1['Stage'] = 'stimuli'
articulators_1['Stage'] = 'articulators'
thinking_1['Stage'] = 'thinking'
speaking_1['Stage'] = 'speaking'
resting_1['Stage'] = 'resting'

stimuli_2['Stage'] = 'stimuli'
articulators_2['Stage'] = 'articulators'
thinking_2['Stage'] = 'thinking'
speaking_2['Stage'] = 'speaking'
resting_2['Stage'] = 'resting'

stimuli_3['Stage'] = 'stimuli'
articulators_3['Stage'] = 'articulators'
thinking_3['Stage'] = 'thinking'
speaking_3['Stage'] = 'speaking'
resting_3['Stage'] = 'resting'

def label(stage, labels = None):
	
	stage_secs = int((len(stage['Label'])/256)/len(labels))

	i = 0
	s = []
	for x in range(0,int((len(stage['Label'])/256)/stage_secs)):
		for xx in range(0,256*stage_secs):
			s.append(labels[i])
		i = i + 1
	return s

stimuli_1['Label'] = label(stimuli_1, labels = trial_1_labels)
articulators_1['Label'] = label(articulators_1, labels = trial_1_labels)
thinking_1['Label'] = label(thinking_1, labels = trial_1_labels)
speaking_1['Label'] = label(speaking_1, labels = trial_1_labels)
resting_1['Label'] = label(resting_1, labels = trial_1_labels)

stimuli_2['Label'] = label(stimuli_2, labels = trial_2_labels)
articulators_2['Label'] = label(articulators_2, labels = trial_2_labels)
thinking_2['Label'] = label(thinking_2, labels = trial_2_labels)
speaking_2['Label'] = label(speaking_2, labels = trial_2_labels)
resting_2['Label'] = label(resting_2, labels = trial_2_labels)

stimuli_3['Label'] = label(stimuli_3, labels = trial_3_labels)
articulators_3['Label'] = label(articulators_3, labels = trial_3_labels)
thinking_3['Label'] = label(thinking_3, labels = trial_3_labels)
speaking_3['Label'] = label(speaking_3, labels = trial_3_labels)
resting_3['Label'] = label(resting_3, labels = trial_3_labels)

##########################################################################################################
### Hardcode correct time and epoch data across trials (time and epochs were reset to 0 per recording) ###
##########################################################################################################

stimuli_2['Time:256Hz'] = [i for i in stimuli_2['Time:256Hz'] + 1408]
stimuli_2['Epoch'] = [i for i in stimuli_2['Epoch'] + 64]
articulators_2['Time:256Hz'] = [i for i in articulators_2['Time:256Hz'] + 1408]
articulators_2['Epoch'] = [i for i in articulators_2['Epoch'] + 64]
thinking_2['Time:256Hz'] = [i for i in thinking_2['Time:256Hz'] + 1408]
thinking_2['Epoch'] = [i for i in thinking_2['Epoch'] + 64]
speaking_2['Time:256Hz'] = [i for i in speaking_2['Time:256Hz'] + 1408]
speaking_2['Epoch'] = [i for i in speaking_2['Epoch'] + 64]
resting_2['Time:256Hz'] = [i for i in resting_2['Time:256Hz'] + 1408]
resting_2['Epoch'] = [i for i in resting_2['Epoch'] + 64]

stimuli_3['Time:256Hz'] = [i for i in stimuli_3['Time:256Hz'] + 2464]
stimuli_3['Epoch'] = [i for i in stimuli_3['Epoch'] + 112]
articulators_3['Time:256Hz'] = [i for i in articulators_3['Time:256Hz'] + 2464]
articulators_3['Epoch'] = [i for i in articulators_3['Epoch'] + 112]
thinking_3['Time:256Hz'] = [i for i in thinking_3['Time:256Hz'] + 2464]
thinking_3['Epoch'] = [i for i in thinking_3['Epoch'] + 112]
speaking_3['Time:256Hz'] = [i for i in speaking_3['Time:256Hz'] + 2464]
speaking_3['Epoch'] = [i for i in speaking_3['Epoch'] + 112]
resting_3['Time:256Hz'] = [i for i in resting_3['Time:256Hz'] + 2464]
resting_3['Epoch'] = [i for i in resting_3['Epoch'] + 112]

#############################################################################
### Create separate dataframes (.csvs) for each separate experiment stage ###
#############################################################################

stimuli = pd.concat([stimuli_1, stimuli_2, stimuli_3])
articulators = pd.concat([articulators_1, articulators_2, articulators_3])
thinking = pd.concat([thinking_1, thinking_2, thinking_3])
speaking = pd.concat([speaking_1, speaking_2, speaking_3])
resting = pd.concat([resting_1, resting_2, resting_3])

stimuli = stimuli.sort_values(by=['Time:256Hz'])
articulators = articulators.sort_values(by=['Time:256Hz'])
thinking = thinking.sort_values(by=['Time:256Hz'])
speaking = speaking.sort_values(by=['Time:256Hz'])
resting = resting.sort_values(by=['Time:256Hz'])

stimuli.to_csv('stimuli.csv', index=False)
articulators.to_csv('articulators.csv', index=False)
thinking.to_csv('thinking.csv', index=False)
speaking.to_csv('speaking.csv', index=False)
resting.to_csv('resting.csv', index=False)

##############################################################################
### Create total dataframe (.csv) combining all separate experiment stages ###
##############################################################################

concatenated = pd.concat([
stimuli_1,
articulators_1,
thinking_1,
speaking_1,
resting_1,
stimuli_2,
articulators_2,
thinking_2,
speaking_2,
resting_2,
stimuli_3,
articulators_3,
thinking_3,
speaking_3,
resting_3,
])

concatenated = concatenated.sort_values(by=['Time:256Hz'])
concatenated.to_csv('full_eeg.csv', index=False) # reconstructed experiment with real values

##################################################################################################
###  Reconstruct experiment with false time and epoch values for visualisation (video) only:   ###
##################################################################################################

def time(stage):
	i = 0
	s = []
	for x in range(0,len(stage['Time:256Hz'])):
		s.append(i)
		i = i + 1/256
	return s
	
def epoch(stage):
	i = 0
	s = []
	for x in range(0,int(len(stage['Epoch']/32)/32)):
		for xx in range(0,32):
			s.append(i)
		i = i + 1
	return s

concatenated['Time:256Hz'] = time(concatenated)  # Overwrite the timestamps (easier for OpenVibe to process contiguous timestamps)
concatenated['Epoch'] = epoch(concatenated)		 # Overwrite the epoch labels (easier for OpenVibe to process 1 epoch per second)
concatenated.columns = header_openvibe			 # Change column labels back to default (otherwise OpenVibe cannot read in the data)
concatenated = concatenated[:60*256]			 # Just get the first minute

concatenated.to_csv('full_eeg_for_video_only.csv', index=False)