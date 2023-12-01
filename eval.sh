# Lift, IIWA, OSC_POSITION, PPO
python scripts/eval.py --env=Lift --policy=PPO --robot=IIWA --dir=exps/Lift-IIWA/OSC_POSITION/PPO/20231128-125035/log/best_model.zip
# Lift, IIWA, OSC_POSITION, SAC
python scripts/eval.py --env=Lift --policy=SAC --robot=IIWA --dir=exps/Lift-IIWA/OSC_POSITION/SAC/20231128-150250/log/best_model.zip