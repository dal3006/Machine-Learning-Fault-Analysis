# Machine-Learning-Fault-Analysis

This repo contains research code related to my master's theses.
The project consists in a PyTorch model which is able to perform fault detection and isolation using vibrations coming from ball bearings.

The project aims at performing fault detection on unlabeled datasets never seen by the model. We achieve this by making heavy use of state-of-the-art transfer learning techniques and semi-supervised training.

**Note**: Currently the project is in a work-in-progress state, so breaking changes may happen at any time.

# Usage

Run training in a docker container leveraging all GPUs

```sh
docker run --rm -v $(pwd):/workspace/ --gpus all -it pytorch-gpu python trainer_main.py --accelerator gpu --source CWRUA --target CWRUB --num_classes 4 --batch_size 128 --save_embeddings false --alpha 0.01 -n debug --learning_rate 1e-3
```

Run training in the cloud using [Grid.AI](https://grid.ai)

```sh
grid run --datastore_name "cwru" \
trainer_main.py --data_dir "/datastores/" --accelerator cpu --autorestore false  \
    --grid_search true --save_embeddings false --experiment_name grid_search --learning_rate 1e-3 --max_epochs 80
```

# Credits

This project is heavily inspired on the paper [Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery](https://ieeexplore.ieee.org/abstract/document/9301443)
For training/testing we are using the CWRU dataset from [Case Western Reserve University, Cleveland, Ohio](https://engineering.case.edu/bearingdatacenter/download-data-file)

# Dataset

#### Bearing fault diagnosis

Transfer against different loads

- CWRU
  - Load: 0HP for CWRUA, 3HP for CWRUB
  - Sensor: DE
  - Faults: B IR OR N
  - Fault diameter: any
  - OR position: any

Classify: 4 classes (B IR OR N)

#### Fault severity diagnosis

Transfer against different sensors/severities

- CWRU
  - Load: 0 to 3HP
  - Sensor: DE, FE
  - Faults: B IR OR N
  - Fault diameter: 007,014,021
  - OR position: any

Classify: 4 classes (B IR OR N)
e.g. train on FE007, test on DE014
