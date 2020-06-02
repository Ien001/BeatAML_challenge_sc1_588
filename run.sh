SYNAPSE_PROJECT_ID=syn21767065
docker build -t docker.synapse.org/$SYNAPSE_PROJECT_ID/sc1_model:ensemble2 .
docker run \
    -v "$PWD/training/:/input/" \
    -v "$PWD/output:/output/" \
    docker.synapse.org/$SYNAPSE_PROJECT_ID/sc1_model:ensemble2
