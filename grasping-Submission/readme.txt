Note all the commands shoukld be run from vtprl folder in the docker container
example: home/vtprl# python3 agent/main_advanced.py
(because otheriwse there would be relative path error)

NOTE: to save the image for the dataset, see iiwa_sample_env.py for more details
NOTE: the datset saved would not be normalized, a.k.a the saved datset labels would be unnormalized
BY home/vtprl# python3 agent/main.py
NOTE: check move.py to split dataset


NOTE we added 2 new parameters in the config_advanced.py which are use_mlp and backbone
for more information on the parameters check the config_advanced.py
