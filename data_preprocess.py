#%%
#

from nptdms import TdmsFile
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

#%%

def plot_signal(signal):
    # print sig min and max
    # print(f"sig min: {signal.min()}")
    # print(f"sig max: {signal.max()}")
    # create a matplotlib plot
    plt.figure()
    plt.plot(signal)
    plt.show()

paths = glob.glob("dataset/mandelli/test_H0/01_prove_lunghe_acc_cuscinetto_alto_basso/1000/*accelerometer*.tdms")
# paths = glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/01_prove_lunghe_acc_cuscinetto/1000/*accelerometer*.tdms")

# convert CAL dataset from tqdm to npy ready
for filepath in paths:
    print(filepath)
    tdms_file = TdmsFile.read(filepath)
    for group in tdms_file.groups():
        group_name = group.name
        print(f"group: {group_name}")

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
        acc_x = acc_x[elevation_bound]
        elevation = elevation[elevation_bound]

        plot_signal(acc_x)
        plot_signal(elevation)

        # save signal to file with same name as tdms file appending _x_cleaned.npy
        savepath = filepath.replace(".tdms", "_x_cleaned.npy")
        print(f"saving to {savepath}")
        np.save(savepath, acc_x)



#%%
from data import minmax_normalization
import torch

# paths = glob.glob("dataset/mandelli/test_H0/01_prove_lunghe_acc_cuscinetto_alto_basso/1000/*accelerometer*.npy")
paths = glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/01_prove_lunghe_acc_cuscinetto/1000/*accelerometer*.npy")

for filepath in paths:
    # read numpy file
    data = np.load(filepath)
    sig = data[10000:10256]
    # plot signal
    plt.figure()
    plt.plot(sig)
    plt.show()

    sig_tensor = torch.tensor(sig).unsqueeze(0)
    sig_tensor = minmax_normalization(sig_tensor)
    # plot signal
    plt.figure()
    plt.plot(sig_tensor.squeeze(0))
    plt.show()

# %%
