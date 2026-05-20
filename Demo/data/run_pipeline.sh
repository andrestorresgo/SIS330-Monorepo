#!/bin/bash

echo "=================================================="
echo " Starting Golden Path Demo Data Pipeline           "
echo "=================================================="

# Spin up the docker compose pipeline which will mount 
# the local directory and build the .npy files inside it.
docker compose up demo-pipeline --build

echo ""
echo "=================================================="
echo " Pipeline Finished. Verify the following files:   "
echo " - demo_hemo_X.npy                                "
echo " - demo_hemo_Y.npy                                "
echo " - demo_vent_X.npy                                "
echo " - demo_vent_Y.npy                                "
echo "=================================================="
