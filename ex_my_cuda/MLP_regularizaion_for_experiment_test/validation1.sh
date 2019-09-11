
StartTime=$(date +%s)
# ./[exe file][learning rate] [betch size] [max norm] [gpu set] [result txt]

for i in {1..30}
do
    ./validation_experiment 0 validation_result1.txt $(($i*1000))
done

EndTime=$(date +%s)
echo "it takes $(($EndTime - $StartTime)) seconds to complete this task"







