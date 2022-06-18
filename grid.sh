## GRID PARAMS
# --use_spot --auto_resume
# --instance_type "g4dn.xlarge"

## TRAINER PARAMS
# --accelerator gpu --gpus 1
# --accelerator cpu


grid run --datastore_name "cwru" \
trainer_main.py --data_dir "/datastores/" --accelerator cpu --autorestore false  \
    --grid_search true --save_embeddings false --experiment_name grid_search --learning_rate 1e-3 --max_epochs 80
