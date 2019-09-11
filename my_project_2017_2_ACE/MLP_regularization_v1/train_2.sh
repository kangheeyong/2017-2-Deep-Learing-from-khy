

StartTime=$(date +%s)
# ./[exe file][learning rate] [betch size] [max norm] [gpu set] [result txt]

./train_experiment 0.00009 200 200.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt
./train_experiment 0.00009 200 2.0 1 result2.txt

EndTime=$(date +%s)
echo "it takes $(($EndTime - $StartTime)) seconds to complete this task"
















