#!/bin/bash

mkdir -p /var/app/pip-tmp
chmod 777 /var/app/pip-tmp

export TMPDIR=/var/app/pip-tmp
export TMP=/var/app/pip-tmp
export TEMP=/var/app/pip-tmp

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
