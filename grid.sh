grid run --instance_type "g4dn.xlarge" --datastore_name "cwru" \
trainer_main.py --data_dir "/datastores/" --accelerator gpu --gpus 1 --source CWRUA3 --target CWRUB3 --num_classes 3 --reuse_target true --autorestore false


#--use_spot --auto_resume
