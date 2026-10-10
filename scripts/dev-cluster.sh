#!/usr/bin/env bash
# Local kind cluster serving a real knowledge base, with Prometheus + Grafana.
#   VAULT_DIR=~/knowledge-base MODE=full  scripts/dev-cluster.sh
#   VAULT_DIR=~/knowledge-base MODE=light scripts/dev-cluster.sh
# Requires docker (e.g. colima), kind, kubectl and helm.
set -euo pipefail
cd "$(dirname "$0")/.."
VAULT_DIR="${VAULT_DIR:?set VAULT_DIR to the directory to serve}"
MODE="${MODE:-full}"
CLUSTER="${CLUSTER:-vaultkv}"
VAULT_DIR="$(cd "$VAULT_DIR" && pwd)"
VALUES_FILE="${VALUES_FILE:-deploy/helm/values-local.yaml}"

if [ ! -f "$VALUES_FILE" ]; then
  echo "Missing $VALUES_FILE. Copy deploy/helm/values-local.example.yaml and edit its batches." >&2
  exit 1
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  sed "s#VAULT_DIR#${VAULT_DIR}#" deploy/kind/kind-config.yaml | kind create cluster --name "$CLUSTER" --config -
fi
docker build -t vaultkv:0.1.0 .
# `kind load docker-image` can fail on newer Docker image stores (containerd I/O
# errors); loading a saved archive is reliable.
archive="$(mktemp -t vaultkv-image).tar"
docker save vaultkv:0.1.0 -o "$archive"
kind load image-archive "$archive" --name "$CLUSTER"
rm -f "$archive"

extra=()
if [ "$MODE" = "light" ]; then
  extra=(--set api.generator=extractive --set api.retrieval=bm25 --set ollama.enabled=false)
fi
helm upgrade --install vk deploy/helm/vaultkv -f "$VALUES_FILE" "${extra[@]}" \
  --wait --timeout 20m
kubectl get pods
