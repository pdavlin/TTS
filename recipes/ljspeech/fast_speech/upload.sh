while true
do
    echo "Uploading logs..."
    echo "$1"
    scp -r $1 davhost:~/cookbook/tensorboard/data/.
    sleep 1800
done
