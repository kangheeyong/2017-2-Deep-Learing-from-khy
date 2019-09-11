

StartTime=$(date +%s)
# ./[exe file][learning rate] [betch size] [max norm] [gpu set] [result txt]
./train_experiment 0.00005 128 2.0 0 result1.txt
./train_experiment 0.00030 128 2.0 0 result1.txt
./train_experiment 0.00007 128 2.0 0 result1.txt
./train_experiment 0.00002 128 2.0 0 result1.txt
./train_experiment 0.00001 128 2.0 0 result1.txt


EndTime=$(date +%s)
echo "it takes $(($EndTime - $StartTime)) seconds to complete this task"
















