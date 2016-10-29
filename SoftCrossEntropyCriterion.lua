local SoftCrossEntropyCriterion, parent = torch.class('nn.SoftCrossEntropyCriterion', 'nn.Criterion')

function SoftCrossEntropyCriterion:__init(T, sizeAverage)
  parent.__init(self)
  self.T = T
  self.sm = nn.SoftMax()
  self.lsm = nn.LogSoftMax()
  self.criterion = nn.DistKLDivCriterion(sizeAverage)
  self.sizeAverage = self.criterion.sizeAverage
end

function SoftCrossEntropyCriterion:updateOutput(input, target)
  self.softTarget = self.sm:updateOutput(target / self.T)
  self.softInput = self.lsm:updateOutput(input / self.T)
  self.output = self.criterion:updateOutput(self.softInput, self.softTarget)
  return self.output
end

function SoftCrossEntropyCriterion:updateGradInput(input, target)
  self.criterion:updateGradInput(self.softInput, self.softTarget)
  self.gradInput = self.lsm:updateGradInput(input / self.T, self.criterion.gradInput) / self.T
  return self.gradInput
end

return SoftCrossEntropyCriterion
