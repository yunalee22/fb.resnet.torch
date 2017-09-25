require 'gnuplot'

print ('Path: ' .. arg[1])
local noiseStats = torch.load(arg[1])

for i = 1, #noiseStats do
  io.write('Class ' .. i .. ' -> ')
  for j = 1, #noiseStats[i] do
    if noiseStats[i][j] ~= 0 then
      io.write(j .. ': ' .. noiseStats[i][j] .. ', ')
    end
  end
  io.write('\n')
end

gnuplot.hist(torch.randn(100000), 100)
