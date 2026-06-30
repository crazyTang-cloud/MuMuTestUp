#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="mumutestup:latest"
HOST_JAVA_DIR="${HOST_JAVA_DIR:-/data/david/java}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"


docker build -t "${IMAGE_NAME}" .
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${HOST_JAVA_DIR}:/data/david/java:ro" \
  -v "${PROJECT_ROOT}:/data/david/project/mumutestup" \
  "${IMAGE_NAME}" "$@"
