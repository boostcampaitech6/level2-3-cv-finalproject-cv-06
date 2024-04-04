python train_MIL.py --model_size="small" --num_workers=8 --drop_rate=0.3 --thr=0.4 --wandb_mode="online" --wandb_run_name="MIL_nl_onlyMIL_no_extra_small" --patience=1000 --use_extra
python train_MIL.py --model_size="small" --num_workers=8 --drop_rate=0.3 --thr=0.4 --wandb_mode="online" --wandb_run_name="MIL_nl_onlyMIL_in_t1loop_small" --patience=1000
