#!/usr/bin/env bash

cd /home/root3/Tuning/benchbase-mysql
BENCHNAME=$1
TIMESTAMP=$2
OUTPUTDIR=$3
OUTPUTLOG=$4
java -jar benchbase.jar -b $BENCHNAME -c config/mysql/sample_${BENCHNAME}_config.xml --execute=true --directory=$OUTPUTDIR > ${OUTPUTLOG}/${BENCHNAME}_${TIMESTAMP}.log