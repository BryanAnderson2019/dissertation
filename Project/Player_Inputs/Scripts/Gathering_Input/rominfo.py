import numpy as np
from itertools import product


'''
Extraction of attributes from the RAM memory of the game Super Mario World
Information taken from: https://www.smwcentral.net/?p=nmap&m=smwram
'''
def getXY(ram):
    '''
    getXY(ram): returns information about the agent's position
    although layer1? is not currently used, it may be useful in some
    changes to the learning algorithm.
    '''
    
    # Coordinates x, y relative to the entire level
    # They are stored in 2 bytes each
    # in little endian format
    marioX = ram[0x95]*256 + ram[0x94]
    marioY = ram[0x97]*256 + ram[0x96]
    
    # Coordinate of the visible part of the level
    layer1x = ram[0x1B]*256 + ram[0x1A]
    layer1y = ram[0x1D]*256 + ram[0x1C]
    
    return marioX.astype(np.int16), marioY.astype(np.int16), layer1x.astype(np.int16), layer1y.astype(np.int16)

def getSprites(ram):
  '''
  getSprites(ram): returns the sprites (blocks, enemies, items) displayed on the screen.
  '''
  
  sprites = []
  extsprites = []
  
  # There can be up to 12 sprites on the screen
  for slot in range(12):
    # if the status is 0, there is no sprite in this slot
    status = ram[0x14C8+slot]
    if status != 0:
      # x,y position of the sprite
      spriteX    = ram[0xE4+slot] + ram[0x14E0+slot]*256
      spriteY    = ram[0xD8+slot] + ram[0x14D4+slot]*256
      
      spriteSize = ram[0x0420+ram[0x15EA+slot]]  # sprite size
      spriteId   = ram[0x15EA+slot]              # which sprite is it?
      
      # if it's an item (44) or block? (216), don't include it in the information
      if spriteId != 44 and spriteId != 216:
        # either it's 1x1 or 4x4 blocks in our window
        size = 1
        if spriteSize == 0:
          size = 4
        sprites.append({'x': spriteX, 'y': spriteY, 'size': size})
      
  return sprites
      
def getTile(dx,dy,ram):
  '''
  getTile(dx, dy, ram): returns whether there is a block the Mario can step on at position dx, dy
  '''
  x = np.floor(dx/16)
  y = np.floor(dy/16)
  
  # 0x1C800 indicates for each pixel whether it is an obstacle or not
  # how to get the right point was taken from here: https://www.smwcentral.net/?p=viewthread&t=78887
  # return ram[0x1C800 + int(np.floor(x/16)*432 + y*16 + x%16)]

  # The correct address is 0x1F000, contribution by Fernando Teixeira
  if (0x1F000 + int(np.floor(x / 16) * 432 + y * 16 + x % 16)) > 131071:
    return ram[131071]
  else:
    return ram[0x1F000 + int(np.floor(x / 16) * 432 + y * 16 + x % 16)]
  
def getInputs(ram, radius=6):
  '''
  getInputs(ram): returns an nd.array of enemies, obstacles within a radius around the agent
  '''
  
  marioX, marioY, layer1x, layer1y = getXY(ram)
  sprites = getSprites(ram)

  # vector length
  maxlen = (radius*2+1)*(radius*2+1)
  inputs = np.zeros(maxlen, dtype=int)
  
  # each image block represents 16x16 pixels
  # so having a reference x,y from Mario
  # we should walk 16 at a time
  window = (-radius*16, radius*16 + 1, 16)
  j = 0

  def withinLimits(idx, ds1, ds2, r, maxlen):
    return (idx%(2*r + 1) + ds2 < 2*r + 1) and (idx + ds1*(2*r + 1) + ds2 < maxlen)
  
  for dy, dx in product(range(*window), repeat=2):
    # checks if there is an obstacle at position x+dx, y+dy
    # the +8 is to start measuring from the middle of Mario
    tile = getTile(marioX+dx+8, marioY+dy, ram)
    
    # Mario is always in the middle, 
    # it must check if y is within the limit
    if tile==1 and marioY+dy < 0x1B0:
      inputs[j] = 1
    
    # For each sprite  
    for i in range(len(sprites)):
      # If it's within the 16 x 16 block (-8, +8)
      distx = np.abs(sprites[i]['x'] - marioX - dx)
      disty = np.abs(sprites[i]['y'] - marioY - dy)
      size = sprites[i]['size']
      if distx <= 8 and disty <= 8:
        # if it's within the limits, insert -1
        for s1, s2 in product(range(size), repeat=2):
          if withinLimits(j, s1, s2, radius, maxlen):
            inputs[j + s1*(radius*2 + 1) + s2] = -1
    j = j + 1
  return inputs, marioX, marioY
  
# Retrieves the current state as a string
def getState(ram, radius):
  state, x, y = getInputs(ram, radius=radius)
  return ','.join(map(str,state)), x, y  
  
def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)
    #return np.array(list(env.data.memory.blocks[8257536]))
 
