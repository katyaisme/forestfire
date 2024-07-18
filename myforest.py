# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:45:58 2024

@author: gelbo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
plt.rcParams['font.size'] = 15

#0 is burnt, 1 is tree, 2 is fire

class forest:
    
    @classmethod
    def makegrid(cls, gridsize, startgr, draw):
        self = cls.__new__(cls)
        self.array = np.zeros(shape=(gridsize, gridsize))
        self.size = gridsize
        self.draw = draw
        
        ind = np.arange(0, gridsize**2 - 1)
        gr = np.random.choice(ind,
                              size=startgr,
                              replace = False)
        for i in gr:
            self.array[self.twoD(i)] = 1
        
        self.getpos()
        self.getcounts()
        
        if draw == True:
            self.setup()
            
        return self

    def twoD(self, x): #convert 1d index to 2d
        return (x//self.size, x%self.size)
    
    def setup(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.gca()
        self.cmap = colors.ListedColormap(['brown',
                                           'green',
                                           'yellow'])
        self.board_img = self.ax.imshow(self.array,
                                        cmap=self.cmap,
                                        vmin = 0,
                                        vmax = 2)
        self.board_img.norm.autoscale([0,1,2])
        self.fig.canvas.draw()
        plt.pause(0.5)
    
    def update(self):
        if self.draw == True:
            self.board_img.set_data(self.array)
            
            self.fig.canvas.draw()
            plt.pause(0.1)
        
        self.getpos()
        self.getcounts()

    def getpos(self):
        bn_1, bn_2 = np.where(self.array == 0) #indexes
        self.burnspos = list(zip(bn_1[::],
                                 bn_2[::]))
        tr_1, tr_2 = np.where(self.array == 1)
        self.treespos = list(zip(tr_1[::],
                                 tr_2[::]))
        fr_1, fr_2 = np.where(self.array == 2) #indexes
        self.firespos = list(zip(fr_1[::],
                                 fr_2[::]))
    
    def getcounts(self):
        self.burntno = np.count_nonzero(self.array == 0)
        self.treeno = np.count_nonzero(self.array == 1)
        self.fireno = np.count_nonzero(self.array == 2)
        
    def burn(self): #rule 1, needs starting fire positions
        for i in self.firespos:
            self.array[i] = 0

    def propagate(self): #rule 2, needs starting fire positions
        for i in self.treespos:
            nbrs = self.getneighbourhood(i[0], i[1])
            if 2.0 in nbrs:
                self.array[i[0], i[1]] = 2
        
    def ignite(self, ignitprob): #rule 3, needs new tree positions
        p_ig = [1-ignitprob, ignitprob]
        new = np.random.choice([1, 2],
                                self.treeno,
                                p=p_ig)
        for i in np.arange(0,self.treeno):
            pos = self.treespos[i]
            self.array[pos] = new[i]
        
    def grow(self, growthprob): #rule 4, needs starting burn positions
        p_gr = [1-growthprob, growthprob]
        new = np.random.choice([0, 1],
                               self.burntno,
                               p=p_gr)
        for i in np.arange(0,self.burntno):
            pos = self.burnspos[i]
            self.array[pos] = new[i]
            
    def getneighbourhood(self, r, c):
        neighbours = []
        nbrind = [(r+1,c),
                  (r-1,c),
                  (r,c+1),
                  (r,c-1)]
        for j in nbrind:
            try:
                neighbours.append(self.array[j])
            except IndexError:
                continue
        return neighbours
            
test = forest.makegrid(200, 10, True)

runcount = 50
countarray = np.zeros(shape=(3,runcount))

for i in np.arange(0,runcount):
    try: 
        test.propagate()
        test.burn()
        test.grow(0.1)
        
        test.getpos() #new positions needed to avoid resetting propagation
        test.getcounts()
        test.ignite(0.001)
        
        test.update()
        print(f'\n\nRun {i}\nTrees: {test.treeno}\nFires: {test.fireno}\nBurnt: {test.burntno}')
        
        countarray[0,i] = test.burntno
        countarray[1,i] = test.treeno
        countarray[2,i] = test.fireno
        
        count = i
        
    except KeyboardInterrupt:
        plt.close('all')
        
        break
    
fig = plt.figure(figsize=(10,10))
plt.plot(countarray[0,:count],
         label = 'burnt',
         color='brown')
plt.plot(countarray[1,:count],
         label = 'trees',
         color='green')
plt.plot(countarray[2,:count],
         label = 'fires',
         color='orange')
plt.legend()
plt.xlabel('Runcount')
plt.ylabel('State cell count')
plt.show()
