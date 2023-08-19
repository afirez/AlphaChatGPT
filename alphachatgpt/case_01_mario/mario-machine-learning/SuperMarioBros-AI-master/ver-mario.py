import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random
import os
import csv

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config

import shutil

    
def save_mario(population_folder: str, individual_name: str, mario) -> None:
    # Make population folder if it doesnt exist
    fdir=os.path.join(population_folder, 'generations')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    save_data={'config':{},
               'net_params':mario
               }
    np.save(os.path.join(fdir,individual_name), save_data)
    
def load_mario(population_folder: str, individual_name: str):
    # Make sure individual exists inside population folder
    if not os.path.exists(os.path.join(population_folder, individual_name)):
        raise Exception(f'{individual_name} not found inside {population_folder}')

    chromosome: Dict[str, np.ndarray] = {}
    # Grab all .npy files, i.e. W1.npy, b1.npy, etc. and load them into the chromosome
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))
        
    return chromosome

for target_dir in os.listdir('./'):
    if 'Example world' in target_dir:
        for name in os.listdir(target_dir):
            if 'best_ind_gen' in name:
                mario=load_mario(target_dir,name)
                ex_dir=os.path.join('./save',target_dir)
                save_mario(ex_dir,name,mario)
                shutil.copy(target_dir+'\settings.config',ex_dir+'\settings.config')


