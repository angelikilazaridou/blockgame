local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	local j  = utils.read_json(opt.json_file)
	self.data = j.data
	self.num_obj = j.num_obj
	self.gpu = opt.gpu
  	self.grid_size = j.grid_size 
  	print('vocab object size is ' .. self.num_obj)
  
  	-- separate out indexes for each of the provided splits
  	self.split_ix = {}
  	self.iterators = {}
  	for i,item in pairs(self.data) do
    		local split = item.split
	    	if not self.split_ix[split] then
			-- initialize new split
   			self.split_ix[split] = {}
	   		self.iterators[split] = 1
    		end
	    	table.insert(self.split_ix[split], i)
  	end
  
	for k,v in pairs(self.split_ix) do
    	print(string.format('assigned %d images to split %s', #v, k))
  	end
end

function DataLoader:resetIterator(split)
  	self.iterators[split] = 1
end

function DataLoader:getNumObj()
	return self.num_obj
end

function DataLoader:getGridSize()
	return self.grid_size
end

--[[
  Split is a string identifier (e.g. train|val|test)
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
	local split = utils.getopt(opt, 'split') 
	local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many games to receive

	-- pick an index of the datapoint to load next
  	local split_ix = self.split_ix[split]
  	assert(split_ix, 'split ' .. split .. ' not found.')


	-- start end end states are represented with a 1-hot grid
    	local startWorld = torch.FloatTensor(batch_size, self.num_obj,  self.grid_size, self.grid_size):fill(0)
	local endWorld = torch.FloatTensor(batch_size, self.num_obj,  self.grid_size, self.grid_size):fill(0)

	local label_batch =  torch.FloatTensor(batch_size, 1)
	
	local max_index = #split_ix
	local wrapped = false
	local infos = {}
  
  	for i=1,batch_size do

    		local ri = self.iterators[split] -- get next index from iterator
    		local ri_next = ri + 1 -- increment iterator
   		if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
   		self.iterators[split] = ri_next
   		ix = split_ix[ri]
   		assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

		
		for j=1,self.num_obj do
			cord = self.data[ix].startW[j]
			x = cord[1]	
			y = cord[2]
			if x ~= -1 and y ~= -1 then
				print(i..' '..j..' '..x..' '..y..' ' )
				startWorld[i][j][x][y] = 1
			end
		end
		-- make a of game and change o,x,y = self.games[ix].end
		endWorld[i] = startWorld[i]:clone()
		d = self.data[ix].endW[1]
		o = d[1]
		x = d[2]
		y = d[3]
		endWorld[i][o]:fill(0)
		endWorld[i][o][x][y] = 1
	end

	if self.gpu<0 then
		data.objects = label_batch:contiguous() -- note: make label sequences go down as columns
		data.startWorld = startWorld
		data.endWorld = endWorld
	else
                data.objects = label_batch:cuda():contiguous()
		data.startWorld = startWorld:cuda()
		data.endWorld = endWorld:cuda()
	end
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  	return data
end

