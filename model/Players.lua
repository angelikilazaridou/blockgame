require 'dpnn'
require 'nn'

local player1 = require 'player1'
local player2 = require 'player2'

local players, parent =  torch.class('nn.Players', 'nn.Module')


function players:__init(opt)
	parent.__init(self)

	-- params
	self.batch_size = opt.batch_size
	self.vocab_size = opt.vocab_size

	--defining the two players
	self.player1 = player1.model() 
	self.player2 = player2.model()
	
	if opt.gpuid == 0 then
		-- categorical for selection of action on object
		self.object_action = nn.ReinforceCategorical(true):cuda()
		-- baseline 
	        self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1)):cuda()
	else
                self.object_action = nn.ReinforceCategorical(true)
                self.baseline = nn.Sequential():add(nn.Constant(1,1)):add(nn.Add(1))

	end


end

--called from game:forward()
function players:updateOutput(input)

	-- input: 	
	
	--player 1 receives start world and end words
	-- does a forward and gives back distribution over objects to move
	self.probs_object = self.player1:forward()

	--sample the object
	self.object = self.object_selection:forward(self.probs_object)


	-- player 2 receives the start world and the object to get moved
	self.prediction = self.player2:forward()

	
	-- baseline
	local baseline = self.baseline:forward(torch.CudaTensor(self.batch_size,1))

	local outputs = {}

	return outputs
end


--called from game:backward()
function players:updateGradInput(input, gradOutput)

end

function players:evaluate()
	self.player1:evaluate()
        self.player2:evaluate()
	self.baseline:evaluate()
        self.object_selection:evaluate()
end

function players:training()
	self.player1:training()
        self.player2:training()
        self.baseline:training()
	self.object_selection:training()

end

function players:reinforce(reward)
	self.object_selection:reinforce(reward)
end

function players:parameters()
  	local p1,g1 = self.player1:parameters()
        local p2,g2 = self.player2:parameters()
	local p3,g3 = self.baseline:parameters()
	
	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
  
	local grad_params = {}
 	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end
	for k,v in pairs(g3) do table.insert(grad_params, v) end

	return params, grad_params
end



