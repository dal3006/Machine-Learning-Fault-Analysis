## GRID PARAMS
# --use_spot --auto_resume
# --instance_type "g4dn.xlarge"

## TRAINER PARAMS
# --accelerator gpu --gpus 1
# --accelerator cpu


# grid run --datastore_name "cwru" \
# trainer_main.py --data_dir "/datastores/" --accelerator cpu --autorestore false  \
#     --grid_search true --save_embeddings false --experiment_name grid_search --learning_rate 1e-3 --max_epochs 80


# grid run --datastore_name "cwru" --instance_type "g4dn.xlarge" \
#     trainer_main.py --data_dir "/datastores/" --resume_chkp_last --max_epochs 70 \
#     --accelerator gpu --gpus 1 \
#     --experiment_name debug_interrupt \
#     --alpha 0 --beta 0



grid run --datastore_name "cwru" --use_spot --auto_resume --instance_type "g4dn.xlarge" \
    trainer_main.py --data_dir "/datastores/" --resume_chkp_last --max_epochs 70 \
    --accelerator gpu --gpus 1 \
    --experiment_name grid_search_resnext \
    --alpha "[0, 0.001, 0.01, 0.1, 1, 10]" --beta "[0, 0.001, 0.01, 0.1, 1, 10]"
