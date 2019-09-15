import random

ids = {
'b':'OVTK_StimulationId_Label_01',
'ch':'OVTK_StimulationId_Label_02',
'd':'OVTK_StimulationId_Label_03',
'dh':'OVTK_StimulationId_Label_04',
'f':'OVTK_StimulationId_Label_05',
'g':'OVTK_StimulationId_Label_06',
'h':'OVTK_StimulationId_Label_07',
'j':'OVTK_StimulationId_Label_08',
'jh':'OVTK_StimulationId_Label_09',
'k':'OVTK_StimulationId_Label_0A',
'l':'OVTK_StimulationId_Label_0B',
'm':'OVTK_StimulationId_Label_0C',
'n':'OVTK_StimulationId_Label_0D',
'ng':'OVTK_StimulationId_Label_0E',
'p':'OVTK_StimulationId_Label_0F',
'r':'OVTK_StimulationId_Label_10',
's':'OVTK_StimulationId_Label_11',
'sh':'OVTK_StimulationId_Label_12',
't':'OVTK_StimulationId_Label_13',
'th':'OVTK_StimulationId_Label_14',
'v':'OVTK_StimulationId_Label_15',
'w':'OVTK_StimulationId_Label_16',
'x':'OVTK_StimulationId_Label_17',
'z':'OVTK_StimulationId_Label_18',
'zh':'OVTK_StimulationId_Label_19',
'bath':'OVTK_StimulationId_Label_1A',
'comma':'OVTK_StimulationId_Label_1B',
'dress':'OVTK_StimulationId_Label_1C',
'fleece':'OVTK_StimulationId_Label_1D',
'foot':'OVTK_StimulationId_Label_1E',
'goose':'OVTK_StimulationId_Label_1F',
'kit':'OVTK_StimulationId_Number_00',
'lot':'OVTK_StimulationId_Number_01',
'strut':'OVTK_StimulationId_Number_02',
'thought':'OVTK_StimulationId_Number_03',
'trap':'OVTK_StimulationId_Number_04'
}

# empty preset if wanting to specify phonemes
preset = ['fleece', 'goose', 'trap', 'thought', 'm', 'n', 'ng', 'f', 's', 'sh', 'v', 'z', 'zh', 'p', 't', 'k']

if preset:
	stims = [ids[stim] for stim in preset]
else:
	print('\nType which phonemes you want to use in your experiment:\nFor example, type "f r bath d fleece b goose kit"\n')
	print('Consonants:\tb\tch\td\tdh\tf\n\t\tg\th\tj\tjh\tk\n\t\tl\tm\tn\tng\tp\n\t\tr\ts\tsh\tt\tth\n\t\tv\tw\tx\tz\tzh\n')
	print('Vowels:\t\tbath\tcomma\tdress\tfleece\n\t\tfoot\tgoose\tkit\tlot\n\t\tstrut\tthought\ttrap\n')

	try:
		stims = [ids[stim] for stim in input('--> ').split(' ')]
	except:
		print('input phonemes are invalid')
		exit(1)

print('\nType how many (randomised) repetitions of each phoneme you want:\nFor example, type "10"\n')
try:
	reps = int(input('--> '))
except:
	print('input repetitions is invalid')
	exit(1)

with open('experiment-timeline.lua', 'w') as f: 
	f.write('''
function initialize(box)

	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	-- each stimulation sent that gets rendered by Display Cue Image box 
	-- should probably have a little period of time before the next one or the box wont be happy
	
	baseline_duration = 5
	cue_duration = 5
	prepare_articulators_duration = 1
	thinking_duration = 5
	speaking_duration = 5
	rest_duration = 5
	post_trial_duration = 1
	
	sequence = {
''')	
	with open('experiment_epochs.txt', 'w') as g:
		inv_ids = {v: k for k, v in ids.items()}
		for i in range(0,reps):
			random.shuffle(stims)	
			for j in stims:
				f.write('\t\t' + j + ',\n')
				g.write(inv_ids[j] + '\n')
	f.write( '''	}

end

function process(box)

	local t = 0

	-- Delays before the trial sequence starts

	box:send_stimulation(1, OVTK_StimulationId_BaselineStart, t, 0)
	t = t + baseline_duration

	-- creates each trial
	for i = 1, #sequence do

		box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, t, 0)
			
		--phoneme
		box:send_stimulation(1, OVTK_StimulationId_Label_00, t, 0)
		box:send_stimulation(1, sequence[i], t, 0)
		t = t + cue_duration
		
		--prepare_articulators
		box:send_stimulation(1, OVTK_GDF_Tongue_Movement, t, 0)
		t = t + prepare_articulators_duration

		--thinking
		box:send_stimulation(1, OVTK_GDF_Feedback_Continuous, t, 0)
		t = t + thinking_duration
		
		--speaking
		box:send_stimulation(1, OVTK_GDF_Tongue, t, 0)
		t = t + speaking_duration
		
		--rest
		box:send_stimulation(1, OVTK_StimulationId_RestStart, t, 0)
		t = t + rest_duration
		
		-- end of thinking epoch and trial
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_RestStop, t, 0)
		t = t + post_trial_duration
		box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)	
	end

	-- send end for completeness	
	box:send_stimulation(1, OVTK_GDF_End_Of_Session, t, 0)
	t = t + 5

	-- used to cause the acquisition scenario to stop and denote final end of file
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, t, 0)
		
end
''')