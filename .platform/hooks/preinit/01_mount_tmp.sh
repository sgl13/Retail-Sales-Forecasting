#!/bin/bash

mkdir -p /opt/eb-tmp
chmod 1777 /opt/eb-tmp

mount --bind /opt/eb-tmp /tmp
chmod 1777 /tmp
