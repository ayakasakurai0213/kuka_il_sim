#!/bin/bash
docker-compose -f build/linux-compose.yaml build
docker-compose -f build/linux-compose.yaml up