
END=$2
PRL=$1
# echo "Running exe $2 in parallel with $1 threads each"
for ((i=1;i<=END;i++)); do
    /home/dbelgrod/bruteforce-gpu/build/cpusweep_bin /mnt/ssd2/UNC-Dynamic-Scene-Benchmarks/cloth-ball/cloth_ball92.ply /mnt/ssd2/UNC-Dynamic-Scene-Benchmarks/cloth-ball/cloth_ball93.ply -p $((PRL)) &
done