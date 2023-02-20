#!/bin/zsh

TIMESTAMP=$(date +%y-%m-%d-%H-%M-%S)

PCM=/home/yuang/Programs/pcm/pcm-core.x
LIK=./scripts/likwid.sh
VTUNE=./scripts/vtune.sh  # ${O_DIR}/${PREFIX}
likwid=./scripts/likwid.sh

#parameters
THREAD=40

ITER=100
DATA_DIR=/data1/csr


declare -A root
root=( ["tr"]=24264010 ["lj"]=488056 ["wl"]=14725342 ["tw"]=23934132 ["mpi"]=779958  )



EXP_NAME="time"
O_DIR=./logs/${EXP_NAME}
mkdir -p ${O_DIR}

for DATA in  roadusa ; do # wl tw sd kr lj pld mpi orkut tr uk02 ur23
      echo " ${APP}: data: ${DATA} thread: ${THREAD}, iter: ${ITER}"
      PREFIX=pava.${DATA}
      ./pava /data1/csr/${DATA}.csr ${THREAD} > ${O_DIR}/${PREFIX}.log 2>&1   
done

