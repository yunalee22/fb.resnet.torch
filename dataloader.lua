--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
      if split == 'train' then
        loaders[i]:injectNoise(dataset, opt)
      end
      loaders[i].num_classes = dataset:getNumClasses()
   end

   return table.unpack(loaders)
end

function DataLoader:getNumClasses()
  return self.num_classes
end

function DataLoader:injectNoise(dataset, opt)
  local p = opt.noisyLabelProbability
  print ("Noisy label probability: " .. tostring(p))
  math.randomseed(os.time())

  local numClasses = dataset:getNumClasses()
  local validOtherChoices = {}
  for i = 1, numClasses do
    table.insert(validOtherChoices, {})
  end

  -- Determine valid choices based on experiment type
  if (opt.experimentType == 'random') then
    for i = 1, numClasses do
      for j = 1, numClasses do
        table.insert(validOtherChoices[i], j)
      end
      table.remove(validOtherChoices[i], i)
    end

  else
    local cm = torch.load('conf/cf_' .. opt.dataset .. '.dat')
    local k = opt.k

    for i = 1, numClasses do
      -- Remove original label
      local idx = {}
      for j = 1, numClasses do
        table.insert(idx, j)
      end
      table.remove(idx, i)
      idx = torch.Tensor(idx):long()
      local choices = cm[i]:index(1, idx)

      -- Select topk most or least confusing labels
      if (opt.experimentType == 'most') then
        _, topk = torch.topk(choices, k, 1, true)
      elseif (opt.experimentType == 'least') then
        _, topk = torch.topk(choices, k, 1, false)
      end
      for j = 1, k do
        if topk[j] >= i then
          topk[j] = topk[j] + 1
        end
      end

      validOtherChoices[i] = torch.totable(topk)
    end
  end

  -- Write valid other choices matrix to file
  os.execute('mkdir -p ' .. opt.logFilePath)
  local file = assert(io.open(opt.logFilePath .. 'candidates.txt', 'w'))
  for i = 1, numClasses do
    file:write(string.format('Class %d -> {', i))
    for j = 1, #(validOtherChoices[i]) do 
      file:write(string.format('%d ', validOtherChoices[i][j]))
    end
    file:write('}\n')
  end
  file:close()

  -- Table to count corruptions from label i to label j
  local noiseStats = {}
  for i = 1, numClasses do
    noiseStats[i] = {}
    for j = 1, numClasses do
      noiseStats[i][j] = 0
    end
  end

  -- Inject label noise
  local labels = {}
  for i = 1, dataset:size() do
    local sample = dataset:get(i)
    if (math.random() < p) then
      local originalTrainingLabel = sample.target
      local choices = validOtherChoices[originalTrainingLabel]
      local newTrainingLabel = choices[math.random(#choices)]
      dataset:set(i, newTrainingLabel)
      -- print("Label " .. originalTrainingLabel .. " -> " .. dataset:get(i).target)
      table.insert(labels, newTrainingLabel)
      noiseStats[originalTrainingLabel][newTrainingLabel] = noiseStats[originalTrainingLabel][newTrainingLabel] + 1 
    end
  end

  -- Write labels to file
  torch.save(opt.logFilePath .. 'labels.txt', labels)

  -- Write noise injection statistics to file
  torch.save(opt.logFilePath .. 'noiseStatistics.txt', noiseStats)
  -- for i = 1, numClasses do
  --   file:write(string.format('Class %d -> {', i))
  --   for j = 1, numClasses do
  --     file:write(string.format('Class %d: %d ', j, noiseStats[i][j]))
  --   end
  --   file:write('}\n')
  -- end
  -- file:close()

end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
    end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
    end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   local function getCPUType(tensorType)
      if tensorType == 'torch.CudaHalfTensor' then
         return 'HalfTensor'
      elseif tensorType == 'torch.CudaDoubleTensor' then
         return 'DoubleTensor'
      else
         return 'FloatTensor'
      end
   end
   self.cpuType = getCPUType(opt.tensorType)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops, cpuType)
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch[cpuType](sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            self.cpuType
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
