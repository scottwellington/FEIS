import os
import subprocess

 # -G flag to guard against clipping samples. Remove silence from beg/mid/end.
downsample = 'sox -G {} -c 1 -r 16k {} silence -l 1 0.1 1% -1 0.1 1%'

in_path = 'VCTK-Corpus'
out_path = 'VCTK'

try:
	os.mkdir(out_path)
except FileExistsError:
	pass

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(in_path):
    for file in f:
        if '.wav' in file:
            files.append(os.path.join(r, file))

count = 0
total = len(files)
for f in files:
	subprocess.run(downsample.format(f,os.path.join('VCTK', os.path.split(f)[1])))
	count += 1
	print('done {} [{}/{}]'.format(os.path.split(f)[1], count, total))