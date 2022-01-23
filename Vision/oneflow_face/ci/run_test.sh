#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-r50}
BZ_PER_DEVICE=${2:-128}
DTYPE=${4:-'fp32'}
TEST_NUM=${5:-1}
ITER_NUM=${6:-400}




i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1,2,3  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    sleep 10s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} ${ITER_NUM} 0,1,2,3,4,5,6,7  1    $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    sleep 10s
done