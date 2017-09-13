for p in $(seq 0.0 0.2 1.0)
do
  th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -depth 50 -nEpochs 256 -LR 0.05 -experiment_type random -noisyLabelProbability $p
  sleep 30s
done
exit 0
