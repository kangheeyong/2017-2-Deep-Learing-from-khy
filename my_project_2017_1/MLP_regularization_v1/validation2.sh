
StartTime=$(date +%s)
# ./[exe file][learning rate] [betch size] [max norm] [gpu set] [result txt]

for i in {1..100}
do
    ./validation_experiment 1 validation_result2.txt
done

EndTime=$(date +%s)
echo "it takes $(($EndTime - $StartTime)) seconds to complete this task"







