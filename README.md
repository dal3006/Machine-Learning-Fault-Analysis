# Machine-Learning-Fault-Analysis

docker run --rm -v $(pwd):/workspace/ --gpus all -it pytorch-gpu python trainer_main.py --accelerator gpu --source CWRUA --target CWRUB --num_classes 4 --batch_size 128 --save_embeddings false --alpha 0.01 -n debug --learning_rate 1e-3

# todo:

- balance classes
- plot accu/recall/precision

## FRAN

- 2 conv layer (kernel=2, filter=[32,64])
- 2 max pool
- 1000 FC (WHAT)
- 3 FC (not 4?)

### Dataset

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
