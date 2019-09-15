import os
import numpy as np
import soundfile
import subprocess
import time
from shutil import copyfile

name = input('Participant name: ')
wavs_dir = '.\\wavs\\{}\\wavs'
combined_wavs_dir = '.\\wavs\\{}\\combined_wavs'
sox_rec = 'sox -t waveaudio -c 1 -r 44100 -d "{}\\temp.wav" silence -l 1 0.1 0.1% 1 1.0 5%'
sox_proc = 'sox "{}\\temp.wav" "{}\\{}.wav" trim 0 1'
sox_play = 'sox "{}\\{}.wav" -t waveaudio -d'

try:
	os.makedirs(wavs_dir.format(name))
	os.makedirs(combined_wavs_dir.format(name))
except FileExistsError:
	pass
	
phonemes = ['f','fleece','goose','k','m','n','ng','p','s','sh','t','thought','trap','v','z','zh']

for i in phonemes:
	while True:
		user = input("\nCurrently recording '{}.wav' ({} out of {}). Press 'enter' to start recording."\
			.format(i, phonemes.index(i)+1, len(phonemes)))
		subprocess.run(sox_rec.format(wavs_dir.format(name)))
		subprocess.run(sox_proc.format(wavs_dir.format(name), wavs_dir.format(name), i))
		subprocess.run(sox_play.format(wavs_dir.format(name), i))
		while True:
			user = input("\nDo you want to (p)lay again, (r)ecord again, or proceed to (n)ext?\n--> ").lower()
			if user == 'r':
				break
			elif user == 'p':
				subprocess.run(sox_play.format(wavs_dir.format(name), i))
			elif user == 'n':
				user = ''
				break
			else:
				pass
		if not user:
			break
	waveform, samplerate = soundfile.read('{}{}'.format(wavs_dir.format(name),'\\{}.wav'.format(i)))
	combined = np.concatenate([waveform,waveform,waveform,waveform,waveform])
	soundfile.write('{}{}'.format(combined_wavs_dir.format(name),'\\{}.wav'.format(i)), combined, samplerate)
	copyfile('{}{}'.format(combined_wavs_dir.format(name),'\\{}.wav'.format(i)), '.\\wavs\\{}.wav'.format(i))
os.remove('{}\\temp.wav'.format(wavs_dir.format(name)))