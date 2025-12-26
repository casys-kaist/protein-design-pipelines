#!/bin/bash

set -e
set -x

sudo sysctl fs.protected_regular=0

minikube delete
# rm -rf /home/jwhwang/.minikube # This is unnecessary

# sudo -E minikube start \
#   --driver=none
minikube start --driver=none
sudo chown -R $USER:$USER $HOME/.minikube

sudo scripts/set-nvidia-docker.py
sudo systemctl restart docker

# Check if nvidia-container-runtime is installed
docker info | grep -i runtime

echo "Waiting for Kubernetes API server to be ready..."
while ! kubectl get --raw='/readyz?verbose' > /dev/null 2>&1; do
  echo "API server is not ready yet. Retrying in 5 seconds..."
  sleep 5
done
echo "API server is ready."

kubectl apply -f https://raw.githubusercontent.com/AliyunContainerService/gpushare-scheduler-extender/master/config/gpushare-schd-extender.yaml

kubectl -n kube-system patch svc gpushare-schd-extender \
  -p '{"spec":{"type":"ClusterIP"}}'

sudo scripts/install-gpushare.py

kubectl create -f https://raw.githubusercontent.com/AliyunContainerService/gpushare-device-plugin/master/device-plugin-rbac.yaml
kubectl create -f https://raw.githubusercontent.com/AliyunContainerService/gpushare-device-plugin/master/device-plugin-ds.yaml

kubectl -n kube-system patch ds gpushare-device-plugin-ds \
  --type=json \
  -p='[
     {"op":"add","path":"/spec/template/spec/hostNetwork","value":true},
     {"op":"add","path":"/spec/template/spec/dnsPolicy",
      "value":"ClusterFirstWithHostNet"}
  ]'

kubectl label node $(hostname) gpushare=true
kubectl label node $(hostname) node-role.kubernetes.io/master=

kubectl inspect gpushare

# Set up nextflow user
kubectl create namespace nextflow
kubectl create serviceaccount nextflow -n nextflow

kubectl create rolebinding nextflow-edit \
  --clusterrole=edit \
  --serviceaccount=nextflow:nextflow \
  -n nextflow

kubectl get serviceaccount nextflow -n nextflow
kubectl apply -f quota-memory.yaml