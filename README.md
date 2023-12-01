# L4DC Project
Train online RL policies (e.g. PPO, SAC) in robosuite with stable_baselines3

## Dependencies
- robosuite v1.4.1 (source)
- robomimic v0.3 (source)
- stable_baselines3 >= 2.2.1
- see requirements.txt

## Setup
Create virtual environment and install dependencies:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
Install robomimic v0.3 and robosuite v1.4.1 from source:
```
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout v0.3
pip3 install -e .
cd ${PATH_TO_DIR}/L4DC_Project

git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.4.1
pip3 install -e .
cd ${PATH_TO_DIR}/L4DC_Project
```

Setup private macro file for robosuite and robomimic.

Test setup:
```
python3 tests/test_setup.py
```

## Run

### Training

```
python scripts/train.py
```

- `--env [str]`: the environment to train on (currently only supports `robosuite.ALL_ENVIRONMENTS`)
- `--robot [str]`: the robot to train with (currently only supports `robosuite.ALL_ROBOTS`)
- `--controller [str]`: the low-level joint controller for the robot (currently only supports `robosuite.ALL_CONTROLLERS`)
- `--policy [str]`: the RL policy to use (SAC or PPO)
- `--continue_training [store_true]`: whether to load previously trained model and continue training
- `--dir [str]`: directory of previously trained model (only used when `--continue_training`)

Example: train SAC policy in robosuite Lift environment with IIWA robot
```
python scripts/train.py --env=Lift --robot=IIWA --policy=SAC --controller=OSC_POSITION
```

## Evaluate

```
python scripts/eval.py
```
- `--env [str]`: the environment to evaluate on (currently only supports `robosuite.ALL_ENVIRONMENTS`)
- `--robot [str]`: the robot to evaluate with (currently only supports `robosuite.ALL_ROBOTS`)
- `--controller [str]`: the low-level joint controller for the robot (currently only supports `robosuite.ALL_CONTROLLERS`)
- `--policy [str]`: the RL policy used (SAC or PPO)
- `--dir [str]`: path to saved model (zip)
- `--eval_eps [Optional[int]]`: number of evaluation episodes

Example: evaluate SAC policy trained for robosuite Lift environment with IIWA robot
```
python scripts/eval.py --env=Lift --robot=IIWA --policy=SAC --controller=OSC_POSITION --dir=PATH_TO_MODEL 
```