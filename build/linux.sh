#!/bin/bash
docker compose -f build/linux-compose.yml build
docker compose -f build/linux-compose.yml up