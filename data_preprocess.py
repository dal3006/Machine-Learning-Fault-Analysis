#%%

from nptdms import TdmsFile
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%

def plot_signal(signal):
    # print sig min and max
    # print(f"sig min: {signal.min()}")
    # print(f"sig max: {signal.max()}")
    # create a matplotlib plot
    plt.figure(figsize=(20,2))
    plt.plot(signal)
    plt.show()

paths = []
paths += glob.glob("dataset/mandelli/test_H0/01_prove_lunghe_acc_cuscinetto_alto_basso/*/*accelerometer*.tdms")
paths += glob.glob("dataset/mandelli/test_H0/03_su_e_giu_acc_cuscinetto/*/*accelerometer*.tdms")
paths += glob.glob("dataset/mandelli/test_H0/09_prove_lunghe_acc_cuscinetto_basso_alto/*/*accelerometer*.tdms")
paths += glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/01_prove_lunghe_acc_cuscinetto/*/*accelerometer*.tdms")
paths += glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/02_prove_lunghe_4corse_su_giu/*/*accelerometer*.tdms")

# convert CAL dataset from tqdm to npy ready
for filepath in paths:
    print("==== Reading file ====")
    print(filepath)
    tdms_file = TdmsFile.read(filepath)
    for group in tdms_file.groups():
        group_name = group.name
        # print(f"group: {group_name}")

        # extract elevation and signal from group
        elevation = None
        acc_x = None
        for channel in group.channels():
            channel_name = channel.name
            # if channel is Acceleration_X save it
            if channel_name == "Acceleration_X":
                acc_x = channel[:]
            # ellse if is untitled save it as elevation
            elif channel_name == "Untitled":
                elevation = channel[:]
            # else print it
            else:
                print(f"ignoring channel:{channel_name}")

        # Clean signal where the contraption is not moving (start and end of the experiment)
        # elevation and acc_x are parallel arrays. We need to filter out acc_x where
        # elevation is constant

        # extract min and max elevation
        elevation_min = elevation.min()
        elevation_max = elevation.max()
        # calculate difference between min and max elevation and increase margins by 10% of that amount
        elevation_diff = elevation_max - elevation_min
        elevation_diff_margin = elevation_diff * 0.1
        elevation_bound_lo = elevation_min + elevation_diff_margin
        elevation_bound_hi = elevation_max - elevation_diff_margin
        # Find where elevation is between margins and
        # keep only the signal where elevation is between them
        elevation_bound = (elevation >= elevation_bound_lo) & (elevation <= elevation_bound_hi)


        # crop acc_x to only between 10% and 50% of the length
        acc_x = acc_x[int(len(acc_x)*0.13):int(len(acc_x)*0.5)]
        df = pd.DataFrame({'x': acc_x[0:]})
        # downsample to 1/10 of the original size
        df = df.iloc[::100]
        df.to_csv('artifacts/mandelli_fullsig.csv')

        elevation = elevation[int(len(elevation)*0.13):int(len(elevation)*0.5)]
        df = pd.DataFrame({'x': elevation[0:]})
        df = df.iloc[::100]
        df.to_csv('artifacts/mandelli_fullpos.csv')
        break

        # plot_signal(acc_x)
        # plot_signal(elevation)

        #TODO: split files with multiple up-down runs into different signals instead of merging all into one signal
        acc_x = acc_x[elevation_bound]
        elevation = elevation[elevation_bound]
        plot_signal(acc_x)
        plot_signal(elevation)


        # save signal to file with same name as tdms file appending _x_cleaned.npy
        savepath = filepath.replace(".tdms", "_x_cleaned.npy")
        print(f"saving to {savepath}")
        np.save(savepath, acc_x)
    break



#%%
from data import minmax_normalization
import torch

VELOCITY="1000"
VELOCITY="30000"

paths = glob.glob("dataset/mandelli/test_H0/01_prove_lunghe_acc_cuscinetto_alto_basso/"+VELOCITY+"/*accelerometer*.npy")
# paths = glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/01_prove_lunghe_acc_cuscinetto/1000/*accelerometer*.npy")

for filepath in paths:
    # read numpy file
    data = np.load(filepath)
    sig = data[10000:10256]
    # plot signal
    plt.figure()
    plt.plot(sig)
    plt.show()

    df = pd.DataFrame({'x': sig})
    df.to_csv('mandelli_sample_'+VELOCITY+'.csv')


    sig_tensor = torch.tensor(sig).unsqueeze(0)
    sig_tensor = minmax_normalization(sig_tensor)
    # plot signal
    plt.figure()
    plt.plot(sig_tensor.squeeze(0))
    plt.show()

# %%
