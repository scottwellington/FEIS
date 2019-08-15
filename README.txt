
><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

Welcome to the FEIS (Fourteen-channel EEG with Imagined Speech) dataset.

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

The FEIS dataset comprises Emotiv EPOC+ [1] EEG recordings of:

* 21 participants listening to, imagining speaking, and then actually speaking
  16 English phonemes (see supplementary, below)

* 2 participants listening to, imagining speaking, and then actually speaking
  16 Chinese syllables (see supplementary, below)

For replicability and for the benefit of further research, this dataset
includes the complete experiment set-up, including participants' recorded
audio and 'flashcard' screens for audio-visual prompts, Lua script and .mxs
scenario for the OpenVibe [2] environment, as well as all Python scripts
for the preparation and processing of data as used in the supporting
studies (submitted in support of completion of the MSc Speech and Language
Processing with the University of Edinburgh):

* J. Clayton, "Towards phone classification from imagined speech using
  a lightweight EEG brain-computer interface," M.Sc. dissertation,
  University of Edinburgh, Edinburgh, UK, 2019.
* S. Wellington, "An investigation into the possibilities and limitations
  of decoding heard, imagined and spoken phonemes using the Emotiv EPOC+
  mobile EEG headset," M.Sc. dissertation, University of Edinburgh,
  Edinburgh, UK, 2019.

Each participant's data comprise 5 .csv files -- these are the 'raw'
(unprocessed) EEG recordings for the 'stimuli', 'articulators' (see
supplementary, below) 'thinking', 'speaking' and 'resting' phases per epoch
for each trial -- alongside a 'full' .csv file with the end-to-end
experiment recording (for the benefit of calculating deltas).

To guard against software deprecation or inaccessability, the full repository
of open-source software used in the above studies is also included.

We hope for the FEIS dataset to be of some utility for future researchers,
due to the sparsity of similar open-access databases. As such, this dataset
is made freely available for all academic and research purposes (non-profit).

><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

REFERENCING

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

If you use the FEIS dataset, please reference:

* S. Wellington and J. Clayton, "Fourteen-channel EEG with Imagined Speech
  (FEIS) dataset," v1.0, University of Edinburgh, Edinburgh, UK, 2019.
  Available: https://github.com/FEIS-dataset/FEIS

><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

LEGAL

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

The research supporting the distribution of this dataset has been approved by
the PPLS Research Ethics Committee, School of Philosophy, Psychology and
Language Sciences, University of Edinburgh (reference number: 435-1819/2).

This dataset is made available under the Open Data Commons Attribution License
(ODC-BY): http://opendatacommons.org/licenses/by/1.0

><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

ACKNOWLEDGEMENTS

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

The FEIS database was compiled by:

Scott Wellington (MSc Speech and Language Processing, University of Edinburgh)
Jonathan Clayton (MSc Speech and Language Processing, University of Edinburgh)

Principal Investigators:

Oliver Watts (Senior Researcher, CSTR, University of Edinburgh)
Cassia Valentini-Botinhao (Senior Researcher, CSTR, University of Edinburgh)

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

METADATA

><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

For participants, dataset refs 01 to 21:

01 - NNS
02 - NNS
03 - NNS, Left-handed
04 - E
05 - E, Voice heard as part of 'stimuli' portions of trials belongs to
     particpant 04, due to microphone becoming damaged and unusable prior to
     recording
06 - E
07 - E
08 - E, Ambidextrous
09 - NNS, Left-handed
10 - E
11 - NNS
12 - NNS, Only sessions one and two recorded (out of three total), as
     particpant had to leave the recording session early
13 - E
14 - NNS
15 - NNS
16 - NNS
17 - E
18 - NNS
19 - E
20 - E
21 - E

E = native speaker of English
NNS = non-native speaker of English (>= C1 level)

For participants, dataset refs chinese-1 and chinese-2:

chinese-1 - C
chinese-2 - C, Voice heard as part of 'stimuli' portions of trials belongs to
            participant chinese-1

C = native speaker of Chinese

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><

SUPPLEMENTARY

><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

Under the international 10-20 system, the Emotiv EPOC+ headset 14 channels:

F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4

The 16 English phonemes investigated in dataset refs 01 to 21:

/i/ /u:/ /æ/ /ɔ:/ /m/ /n/ /ŋ/ /f/ /s/ /ʃ/ /v/ /z/ /ʒ/ /p /t/ /k/

The 16 Chinese syllables investigated in dataset refs chinese-1 and chinese-2:

mā má mǎ mà mēng méng měng mèng duō duó duǒ duò tuī tuí tuǐ tuì

All references to 'articulators' (e.g. as part of filenames) refer to the
1-second 'fixation point' portion of trials. The name is a layover from
preliminary trials which were modelled on the KARA ONE database
(http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) [3].

<>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <>< <><
><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>

[1] Emotiv EPOC+. https://emotiv.com/epoc. Accessed online 14/08/2019.

[2] Y. Renard, F. Lotte, G. Gibert, M. Congedo, E. Maby, V. Delannoy,
    O. Bertrand, A. Lécuyer. “OpenViBE: An Open-Source Software Platform
    to Design, Test and Use Brain-Computer Interfaces in Real and Virtual
    Environments”, Presence: teleoperators and virtual environments,
    vol. 19, no 1, 2010.

[3] S. Zhao, F. Rudzicz. "Classifying phonological categories in imagined
    and articulated speech." In Proceedings of ICASSP 2015, Brisbane
    Australia, 2015.
