import pandas as pd
import os
from collections import defaultdict
from scipy.stats import bootstrap
import numpy as np
import math
import sys
from helper_functions import parse_file_name

RANDOM_SEED = 7

data_folder = sys.argv[1]
accuracy_mappings = {"gpt-3.5-turbo": defaultdict(list), 
                     "gpt-4o": defaultdict(list), 
                     "llama-3-8b": defaultdict(list), 
                     "llama-3-70b": defaultdict(list), 
                     "mixtral-8x7b-instruct": defaultdict(list)
    }

for file in os.listdir(data_folder):
    model, task, seed = parse_file_name(file)
    data = pd.read_csv(f"{data_folder}/{file}")
    accuracy = len(data[data["new_extracted_pred"] == data["gt"]]) / len(data)
    accuracy_mappings[model][task].append(accuracy)

for model in accuracy_mappings:
    for task in accuracy_mappings[model]:
        data = accuracy_mappings[model][task] 
        ci_95 = bootstrap((data,), np.mean, n_resamples=2000, confidence_level=0.95,
                random_state=RANDOM_SEED)#uses bias-corrected and accelerated bootstrap confidence interval
        low, high = ci_95.confidence_interval
        if math.isnan(low):
            low, high = np.mean(data), np.mean(data)
        print(f"Model: {model}, Task: {task}, 95% CI: ({low=:.3f},{high=:.3f}), Mean: {np.mean(data)=:.3f}")
