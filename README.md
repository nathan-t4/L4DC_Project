# L4DC Project

## Setup
Create virtual environment and install dependencies:
```
    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt
```
Setup private macro file (recommended by robosuite):
```
    python3 ${PATH_TO_DIR}/L4DC_Project/env/lib/python3.9/site-packages/robosuite/macros_private.py
```
Change python3.9 to python version

Test setup:
```
    python3 tests/test_setup.py
```

Test simulation:
```
    python3 tests/test_simulation.py
```