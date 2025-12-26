#!/usr/bin/env python3
import yaml
import shutil
import os
from pathlib import Path

KUBE_SCHEDULER_YAML_PATH = "/etc/kubernetes/manifests/kube-scheduler.yaml"
SCHEDULER_POLICY_CONFIG_PATH = "/etc/kubernetes/scheduler-policy-config.yaml"

def add_gpushare_config():
    if not os.path.exists(KUBE_SCHEDULER_YAML_PATH):
        print(f"Error: {KUBE_SCHEDULER_YAML_PATH} not found.")
        return

    # Backup original file
    kube_scheduler_yaml_path = Path(KUBE_SCHEDULER_YAML_PATH)
    backup_path = kube_scheduler_yaml_path.parent.parent / (kube_scheduler_yaml_path.name + '.bak')
    if not backup_path.exists():
        print(f"Backing up {KUBE_SCHEDULER_YAML_PATH} to {backup_path}")
        shutil.copy(KUBE_SCHEDULER_YAML_PATH, backup_path)
    else:
        print(f"Backup file {backup_path} already exists. Skipping backup.")

    try:
        with open(KUBE_SCHEDULER_YAML_PATH, 'r') as f:
            scheduler_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {KUBE_SCHEDULER_YAML_PATH}")
        print(e)
        print("\nThis might be due to a malformed YAML file, possibly from previous script runs.")
        print(f"Please restore from backup '{backup_path}' or manually fix the file.")
        return

    # Check and add command line argument
    command = scheduler_config['spec']['containers'][0]['command']
    config_arg = f'--config={SCHEDULER_POLICY_CONFIG_PATH}'
    if config_arg not in command:
        command.insert(command.index('--leader-elect=false') + 1, config_arg)
        print("Added scheduler config argument.")

    # Check and add volume mount
    volume_mounts = scheduler_config['spec']['containers'][0]['volumeMounts']
    policy_mount_exists = any(vm['name'] == 'scheduler-policy-config' for vm in volume_mounts)
    if not policy_mount_exists:
        volume_mounts.insert(0, {
            'mountPath': SCHEDULER_POLICY_CONFIG_PATH,
            'name': 'scheduler-policy-config',
            'readOnly': True
        })
        print("Added scheduler policy volume mount.")

    # Check and add volume
    volumes = scheduler_config['spec']['volumes']
    policy_volume_exists = any(v['name'] == 'scheduler-policy-config' for v in volumes)
    if not policy_volume_exists:
        volumes.insert(0, {
            'hostPath': {
                'path': SCHEDULER_POLICY_CONFIG_PATH,
                'type': 'FileOrCreate'
            },
            'name': 'scheduler-policy-config'
        })
        print("Added scheduler policy volume.")

    # Check and add dnsPolicy
    if 'dnsPolicy' not in scheduler_config['spec']:
        scheduler_config['spec']['dnsPolicy'] = 'ClusterFirstWithHostNet'
        print("Added dnsPolicy: ClusterFirstWithHostNet")

    with open(KUBE_SCHEDULER_YAML_PATH, 'w') as f:
        yaml.dump(scheduler_config, f, default_flow_style=False)

    print(f"✅ Settings have been added to {KUBE_SCHEDULER_YAML_PATH}. (Backup: {backup_path})")

if __name__ == "__main__":
    add_gpushare_config() 