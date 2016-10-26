local SoftCrossEntropyCriterion, parent = torch.class('nn.SoftCrossEntropyCriterion', 'nn.Criterion')

function SoftCrossEntropyCriterion:__init(T, weights, sizeAverage)
  parent.__init(self)
  self.T = T
  self.criterion = nn.CrossEntropyCriterion(weights, sizeAverage)
  self.sizeAverage = self.criterion.sizeAverage
end

local function sm(input)
  if input:dim() == 1 then
    return torch.exp(input) / torch.exp(input):sum()
  elseif input:dim() == 2 then
    return torch.cdiv(torch.exp(input), torch.exp(input):sum(2):expandAs(input))
  else
    error('matrix or vector expected')
  end
end

function SoftCrossEntropyCriterion:updateOutput(input, target)
  self.criterion:updateOutput(input / self.T, target)
  self.output = self.criterion.output
  return self.output
end

function SoftCrossEntropyCriterion:updateGradInput(input, target)
  self.criterion:updateGradInput(input / self.T, target)
  self.gradInput = self.criterion.gradInput / self.T
  return self.gradInput
end

return SoftCrossEntropyCriterion
