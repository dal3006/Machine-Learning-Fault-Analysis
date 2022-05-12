# Machine-Learning-Fault-Analysis

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
