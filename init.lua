require 'nn'
require 'bnn'

local optnet = require 'optnet.env'
require 'optnet.countUsedMemory'
require 'optnet.tests'

local utils = require 'optnet.utils'

local kNotUsed = 10000---1
local kNotDefined = 0
local kMinimumForSharing = 2
local kAlwaysLive = 10000

local function analyse(net, input, opts)
  opts = opts or {}
  local mode = opts.mode or 'inference'

  local track = {}
  local analysis = {}

  local function entry_fun(t, args)
    local ptr = torch.pointer(t:storage())
    local info = {used=kNotUsed, defined=kNotDefined,
                  name=args.name, ptr=ptr, tensor=t}
    table.insert(analysis, info)
    return info
  end

  local function fun(t, track, args)
    local ptr = torch.pointer(t:storage())
    local val = track[ptr][args.var]
    if val == args.notUsed then
      track[ptr][args.var] = args.c
    else
      track[ptr][args.var] = args.f(args.c,val)
    end
  end

  local c = 1
  local function apply_func(m)
    local func = 'updateOutput'
    local basefunc = m[func]
    m[func] = function(self, input)
      local opts = {
        analysis=analysis, c=c, name=tostring(m),
        kNotUsed=kNotUsed, kNotDefined=kNotDefined
      }
      if mode == 'inference' then
        -- always keep track of the input
        opts.var = 'used'; opts.f = math.max; opts.notUsed = kNotUsed
        utils.keepTrack(input, track, entry_fun, fun, opts)

        if not m.modules then
          -- always keep track of the outputs of non-containers
          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(self.output, track, entry_fun, fun, opts)
        elseif torch.typename(m) == 'nn.Concat' or
          torch.typename(m) == 'nn.Parallel' or
          torch.typename(m) == 'nn.DepthConcat' then

          -- for containers that do some operations on the input, need to keep
          -- track of each output of its branches uppon entry on the module,
          -- as well as to keep track of it's own output (as it's a non-trivial
          -- operation on the childs output, contrary to nn.Sequential for
          -- example)
          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(self.output, track, entry_fun, fun, opts)

          for i,branch in ipairs(m.modules) do
            local last_module
            -- if brach is a container, get its last element, if not, take it
            if branch.modules then
              last_module = branch:get(branch:size())
            else
              last_module = branch
            end
            local out = last_module.output
            opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
            utils.keepTrack(out, track, entry_fun, fun, opts)
          end
        end
      end
      c = c + 1
      return basefunc(self,input)
    end

    for _, func in ipairs({'updateGradInput','accGradParameters','backward'}) do
      local basefunc = m[func]
      m[func] = function(self, input, gradOutput, scale)
        local opts = {
          analysis=analysis, c=c, name=tostring(m),
          kNotUsed=kNotUsed, kNotDefined=kNotDefined
        }

        -- always keep track of the input
        --opts.var = 'used'; opts.f = math.max; opts.notUsed = kNotUsed
        --utils.keepTrack(input, track, entry_fun, fun, opts)
        if not torch.isTypeOf(m, 'nn.Sequential') then
        -- always keep track of the gradOutput
	  opts.var = 'used'; opts.f = math.max; opts.notUsed = kNotUsed
          utils.keepTrack(gradOutput, track, entry_fun, fun, opts)

          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(self.gradInput, track, entry_fun, fun, opts)
        end

        --[[
        if not m.modules then
          -- always keep track of the gradInput of non-containers
          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(self.gradInput, track, entry_fun, fun, opts)
        elseif torch.typename(m) == 'nn.Concat' or
          torch.typename(m) == 'nn.Parallel' or
          torch.typename(m) == 'nn.DepthConcat' then

          -- for containers that do some operations on the gradOutput, need to keep
          -- track of each gradInput of its branches uppon entry on the module,
          -- as well as to keep track of it's own gradInput (as it's a non-trivial
          -- operation on the childs output, contrary to nn.Sequential for
          -- example)
          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(self.gradInput, track, entry_fun, fun, opts)

          for i,branch in ipairs(m.modules) do
            local first_module = branch:get(1)
            local out = first_module.gradInput
            opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
            utils.keepTrack(out, track, entry_fun, fun, opts)
          end
        end
        --]]
        c = c + 1
        return basefunc(self,input,gradOutput,scale)
      end

    end

  end
  net:apply(apply_func)
  local out = net['forward'](net, input)
  local grad
  if mode == 'training' then
    grad = utils.recursiveClone(out)
    net['backward'](net, input, grad)
  end
  local function trackInputs(t, name)
    if torch.isTensor(t) then
      local f = function(a,b) return a end
      utils.keepTrack(t, track, entry_fun, fun,
      {var='used', c=kAlwaysLive,
      f=f, notUsed=0, name=name})
      utils.keepTrack(t, track, entry_fun, fun,
      {var='defined', c=-kAlwaysLive,
      f=f, notUsed=0, name=name})
    else
      for k,v in ipairs(t) do
        trackInputs(v)
      end
    end
  end
  trackInputs(input,'input')
  if grad then
    trackInputs(grad,'grad')
  end
  -- clean up the modified function
  net:apply(function(x)
    x['updateOutput'] = nil
    x['updateGradInput'] = nil
    x['accGradParameters'] = nil
    x['backward'] = nil
  end)

  -- disable backward pass if in evaluation mode
  if mode == 'inference' then
    net:apply(function(m)
      m.updateGradInput = function(self, input, gradInput)
        error([[Backward pass disabled!
          You are using inference optimization.
          Call optnet.removeOptimization(net) to enable backward again]])
      end
    end)
  end
  return analysis
end

local function isCompatible(candidate, assignment)
  if candidate.used == kNotUsed then
    return false
  end
  if candidate.tensor:numel() < kMinimumForSharing then
    return false
  end
  local a_used = assignment[#assignment].used
  return candidate.defined > a_used
end

local function assign(net, analysis)
  table.sort(analysis, function(a,b)
    local x = a.used
    local y = b.used
    return x < y
  end)
  local assignments = {}
  for _,candidate in ipairs(analysis) do
    local assigned = false
    local bestAssignment = 0
    local minDist = math.huge
    local candidateSize = candidate.tensor:numel()
    for idx, assignment in ipairs(assignments) do
      if isCompatible(candidate, assignment) then
        assigned = true
        local dist = math.abs(assignment.maxSize-candidateSize)
        if dist < minDist then
          minDist = dist
          bestAssignment = idx
        end
      end
    end
    if assigned then
      local assignment = assignments[bestAssignment]
      table.insert(assignment, candidate)
      assignment.maxSize = math.max(assignment.maxSize, candidateSize)
    else
      table.insert(assignments, {candidate, maxSize=candidateSize})
    end
  end
  return assignments
end

local function applyAssignments(net, assignments)
  for _, assignment in ipairs(assignments) do
    local storage
    for k, v in ipairs(assignment) do
      if v.used == kAlwaysLive and v.defined == -kAlwaysLive then
        break
      end
      storage = storage or v.tensor.new(1):storage()
      v.tensor:set(storage)
    end
  end
end

local function defaultValue(var, val)
  if var == nil then
    var = val
  end
  return var
end

-- set to inplace modules that allows it
local function setInplace(net, opts)
  local inplace = defaultValue(opts.inplace, true)
 
  if inplace then
    net:apply(function(m)
      if m.inplace ~= nil then
        -- inplace is not always supported for threshold,
        -- depending on the values. Disabling it for the moment
        if torch.typename(m) ~= 'nn.Threshold' then
          m.inplace = true
        end
      end
    end)
  end
end

local reusableBuffers = {
['nn.SpatialConvolution'] = {{'finput','fgradInput'},{'fgradInput'}},
['bnn.SpatialConvolution'] = {{'finput','fgradInput','columns_binary'},{}},
['nn.SpatialConvolutionMM'] = {{'finput','fgradInput'},{'fgradInput'}},
['nn.Normalize'] = {{'norm','buffer','normp','_indices'},{}},
['nn.SpatialCrossMapLRN'] = {{'scale'},{}},
['nn.SpatialMaxPooling'] = {{'indices'},{}},
}
-- basic reusing scheme: keeps a list of all possible buffers
-- that can be reused in evaluation mode and also in training
-- mode.
local function reuseStateBuffers(net, opts)
  local reuseBuffers = defaultValue(opts.reuseBuffers, true)
  local mode = defaultValue(opts.mode, 'inference')
  local mode_idx = 1
  if mode == 'training' then
    mode_idx = 2
  end
  -- workaround SpatialMaxUnpooling corner case
  -- https://github.com/fmassa/optimize-net/issues/14
  local reusableBuffers = utils.copyTable(reusableBuffers)
  if #net:findModules('nn.SpatialMaxUnpooling') > 0 then
    reusableBuffers['nn.SpatialMaxPooling'] = {{},{}}
  end
  if reuseBuffers then
    local reusedBuffers = {}
    net:apply(function(m)
      local name = torch.typename(m)
      if reusableBuffers[name] then
        local rb = reusableBuffers[name][mode_idx]
        for k, v in ipairs(rb) do
          if m[v] then
            reusedBuffers[name..','..v] = reusedBuffers[name..','..v] or m[v]:storage()
            if reusedBuffers[name..','..v] then
              m[v]:set(reusedBuffers[name..','..v])
            end
          end
        end
      end
    end)
  end
end

-- needed for cudnn
local function resetInputDescriptors(net)
  net:apply(function(m)
    if torch.typename(m):find('cudnn') and
       torch.typename(m.iSize) == 'torch.LongStorage' then
      m.iSize:fill(0)
    end
  end)
end

-- need to keep a list of shared gradParams
-- to avoid problems when removing the optimization
local function removeGradParams(net, opts)
  local removeGradParams = defaultValue(opts.removeGradParams, true)
  local mode = defaultValue(opts.mode, 'inference')
  if not removeGradParams then return end
  if mode == 'training' then return end
  local storages = {}
  net:apply(function(m)
    for _, k in ipairs({'gradWeight','gradBias'}) do
      if m[k] and m[k]:storage() then
        local strPtr = torch.pointer(m[k]:storage())
        local strOffset = m[k]:storageOffset()
        local strSize = m[k]:storage():size()
        local tPtr = torch.pointer(m[k])
        storages[tPtr] = {storage=strPtr, offSet=strOffset,
                          size=strSize, stride=m[k]:stride()}
        m[k]:set()
      end
    end
    -- disabling getParameters
    m.getParameters = function(self)
      error('getParameters was disabled by optnet '..
            '(by option removeGradParams=true). '..
            'Call optnet.removeOptimization(net) to enable it back.')
    end
  end)
  net.__gradParamsInfo = storages
end

local function addGradParams(net)
  local storages = net.__gradParamsInfo
  if not storages then return end
  local createdStorages = {}
  net:apply(function(m)
    for k, v in pairs({weight='gradWeight',bias='gradBias'}) do
      if m[v] then
        local tPtr = torch.pointer(m[v])
        local info = storages[tPtr]
        if not createdStorages[info.storage] then
          local strSize = info.size
          createdStorages[info.storage] = m[v].new(strSize):storage()
        end
        local storage = createdStorages[info.storage]
        local tSize = m[k]:size()
        local tStride = info.stride
        local tOffset = info.offSet
        m[v]:set(storage, tOffset, tSize, tStride)
      end
    end
    -- add back original getParameters
    m.getParameters = nil
  end)
  net.__gradParamsInfo = nil
end


function optnet.optimizeMemory(net, input, opts)
  opts = opts or {}

  if net.__memoryOptimized then
    print('Skipping memory optimization. '..
          'Network is already optimized for '..net.__memoryOptimized..' mode.')
    return
  end

  local mode = defaultValue(opts.mode,'inference')

  local out = net['forward'](net, input)
  local grad
  if mode == 'training' then
    grad = utils.recursiveClone(out)
    net['backward'](net, input, grad)
  end

  setInplace(net, opts)
  reuseStateBuffers(net, opts)
  removeGradParams(net, opts)

  -- share outputs
  local analysis = analyse(net, input, opts)
  --print(analysis)
  local assignments = assign(net,analysis)
  --print(assignments)
  applyAssignments(net, assignments)
  resetInputDescriptors(net)

  -- add flag to mention that it was optimized
  net.__memoryOptimized = mode
end

function optnet.removeOptimization(net)

  if not net.__memoryOptimized then
    print('Skipping memory optimization removal, as the network was not optimized.')
    return
  end

  local function rem(m)
    if torch.isTensor(m) then
      m:set()
    end
    if torch.type(m) == 'table' then
      for k, v in ipairs(m) do
        rem(v)
      end
    end
  end
  
  net:apply(function(m)
    rem(m.output)
    rem(m.gradInput)
    local name = torch.typename(m)
    if reusableBuffers[name] then
      local rb = reusableBuffers[name][1]
      for k, v in ipairs(rb) do
        if m[v] then
          m[v]:set()
        end
      end
    end
    -- remove backward blocking
    m.updateGradInput = nil
  end)
  resetInputDescriptors(net)
  addGradParams(net)

  net.__memoryOptimized = nil
end

return optnet

