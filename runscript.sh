for p in 0.0 0.2 0.4 0.6 0.8 0.9 1.0
do
  for k in 1 10 50
  do
    th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -depth 50 -nEpochs 256 -LR 0.05 -experimentType most -k $k -noisyLabelProbability $p -logFilePath tmp/cifar100/mostConfusing_$k/$p/
    sleep 15s
  done
done
exit 0
