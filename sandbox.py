#%%
#

from nptdms import TdmsFile
import glob
import os
import matplotlib.pyplot as plt

#%%

def plot_signal(signal):
    # print sig min and max
    # print(f"sig min: {signal.min()}")
    # print(f"sig max: {signal.max()}")
    # create a matplotlib plot
    plt.figure()
    plt.plot(signal)
    plt.show()

paths = glob.glob("dataset/mandelli/test_H0/01_prove_lunghe_acc_cuscinetto_alto_basso/1000/*.tdms")
# paths = glob.glob("dataset/mandelli/test_A_fault_cuscinetto_pitting/01_prove_lunghe_acc_cuscinetto/1000/*.tdms")

for filepath in paths:
    print(filepath)
    tdms_file = TdmsFile.read(filepath)
    for group in tdms_file.groups():
        group_name = group.name
        print(f"group: {group_name}")
        for channel in group.channels():
            print(f"channel:{channel}")

            channel_name = channel.name
            # Access dictionary of properties:
            properties = channel.properties
            # for prop in properties:
            #     print(f"\t\t\t property: {prop}")
            # Access numpy array of data for channel:
            signal = channel[:]
            plot_signal(signal)


#%%




# %%
