#! /usr/bin/env bash

FILELIST=/tmp/filelist
MONITOR_DIR=$1

scp -r $1 davhost:~/cookbook/tensorboard/data/.

[[ -f ${FILELIST} ]] || ls ${MONITOR_DIR} > ${FILELIST}

while : ; do
    cur_files=$(ls ${MONITOR_DIR})
    diff <(cat ${FILELIST}) <(echo $cur_files) || \
         { echo "Alert: ${MONITOR_DIR} changed" ;
           scp -r $1 davhost:~/cookbook/tensorboard/data/. ;
           # Overwrite file list with the new one.
           echo $cur_files > ${FILELIST} ;
         }

    # echo "Waiting for changes."
    sleep 120
done

# echo "$1"
# while true
# do
#     echo "Uploading logs..."
#     scp -r $1 davhost:~/cookbook/tensorboard/data/.
#     sleep $2
# done