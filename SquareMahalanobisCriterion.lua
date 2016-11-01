local M, parent = torch.class('nn.SquareMahalanobisCriterion', 'nn.Criterion')

-- Input: x, mu, sigma. Calculates (x-mu)^T*sigma^-1*(x-mu)
function M:__init(sizeAverage)
  parent.__init(self)
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
end

function M:setCov(cov)
  self.invcov = cov.new(cov:size())
  for i=1,cov:size(1) do
    self.invcov[i] = torch.inverse(cov[i])
  end
end

function M:updateOutput(input, target)
  if input:dim() > 2 then
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  if input:dim() == 1 then input = input:view(1, -1) end

  self.output = 0
  for i=1,input:size(1) do
    local diff = (input[i] - target[i]):view(-1, 1)
    self.output = self.output + diff:t() * self.invcov[i] * diff
  end
  if self.sizeAverage then
    self.output = self.output / input:size(1)
  end
  return self.output
end

function M:updateGradInput(input, target)
  if input:dim() > 2 then
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  if input:dim() == 1 then input = input:view(1, -1) end

  self.gradInput:resizeAs(input)
  for i=1,input:size(1) do
    local diff = (input[i] - target[i]):view(-1, 1)
    self.gradInput[i] = 2 * self.invcov[i] * diff
  end
  if self.sizeAverage then
    self.gradInput = self.gradInput / input:size(1)
  end
  return self.gradInput
end

return M
