for p in $(seq 0.0 0.2 1.0)
do
  th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 50 -nEpochs 256 -LR 0.05 -experiment_type most -noisyLabelProbability $p
  sleep 30s
done
exit 0
