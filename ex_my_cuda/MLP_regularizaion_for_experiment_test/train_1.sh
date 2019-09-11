

StartTime=$(date +%s)
# ./[exe file][learning rate] [betch size] [max norm] [gpu set] [result txt]

./train_experiment_elu 0.0001 128 2000.0 0 result1.txt
./train_experiment_elu 0.0001 128 2000.0 0 result1.txt
./train_experiment_elu 0.0001 128 2000.0 0 result1.txt
./train_experiment_elu 0.0001 128 2000.0 0 result1.txt

EndTime=$(date +%s)
echo "it takes $(($EndTime - $StartTime)) seconds to complete this task"
















