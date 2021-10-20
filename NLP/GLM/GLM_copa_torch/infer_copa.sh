export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True

bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_base.sh \
     config_tasks/task_copa.sh