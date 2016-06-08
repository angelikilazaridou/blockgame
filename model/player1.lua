require 'misc.LinearNB'
require 'misc.Peek'
require 'nngraph'
require 'dp'

local player1 = {}
function player1.model(game_size, feat_size, vocab_size, hidden_size, dropout, gpu)

    local startWorld = nn.Identity()()
    local endWorld = nn.Identity()()

    local inputs = {}
    table.insert(inputs, startWorld)
    table.insert(inputs, endWorld)

	local probs = nn.SoftMax()(result)
	--take out discriminative 
    	local outputs = {}
	table.insert(outputs, probs)
	

	local model = nn.gModule(inputs, outputs)

   
	return model
end

return player1
