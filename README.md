# L4DC Project

## Dependencies
- robosuite v1.4.1 (source)
- robomimic v0.3 (source)
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
```
python scripts/train.py --env=ROBOSUITE_ENV_NAME
```