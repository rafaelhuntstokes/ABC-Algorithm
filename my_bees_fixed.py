from __future__ import division

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:24:12 2018


This is a first attempt at creating a single objective Artificial Bee Colony Algorithm.


@author: rafael
"""


import random 
import sys    #access to interpretor commands 
import copy   #allows deepcopy when attempting to replicate mutable objects (lists etc) 

#require two classes: the bee, including functions for evaluating cost functions etc, & the hive itself, which 
#contains the functions for running the algorithm (ie each stage of initiation, employed, onlookers, waggle dance and scout)


class bee(object):
    
    def __init__(self, lower, upper, func):
        """This function is used to initialise a bee with a random solution vector, evaluate the fitness of the solution
           and update the bees abandonment counter"""
        
        self.fitness = None 
	   
        #initialises random solution vector between the lower and upper bounds of problem by calling random function
        self.random(lower, upper)
           
        self.value = func(self.vector) 

        #computes fitness of solution vector by calling fitness function 
        self.calcFitness()
           
        #starts abandonment counter
        self.counter = 0 
           

    def random(self, upper, lower):
       """Creates empty self.vector and then appends it the length of lower bound number of times with the ABC algorith 
       equation for random vector"""
       self.vector = []
       for i in range(len(lower)):
           self.vector.append(lower[i] + random.random() * (upper[i] - lower[i]))
       
       
    def calcFitness(self):
        """Evaluates fitness of solution according to the fitness function (abritarily determined by me). Must give high 
        score for good solutions, low score to bad solutions and exists as a selection method"""
        if self.value >= 0:
            self.fitness = 1 / (self.value + 1)
        else:
            self.fitness = 1 + abs(1/self.value)
            
            
class beehive(object):
    """Initialises a beehive object.This class includes the functions which actually run the algorithm:
        1. Initialissation
        2. Employee
        3. Onlookers
        4. Waggle dance
        5. Scout 
    """

        
        
    def __init__(self, func, lower, upper, num_bees = 30, max_its = 100, verbose = False, seed = None, max_trials = None):
        """Intialises beehive object. Randomises individuals betweeen upper and lower bounds of search space"""
        
        #checks lower and upper bounds are of the same length 
        assert (len(lower) == len (upper))
        
        #activates seed for random number generator 
        if seed == None: 
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed 
        random.seed(self.seed) 
        
        #computes number of employees. ORIGINAL = num_bees + num_bees % 2 --> WHY??? Got rid of it.  
        self.size = int(num_bees) 
        
        #assigns basic properties of algorithm
        self.dims = len(lower)
        self.max_its = max_its
        
        #ensures there's an abandonment counter even if max trials set to "None" - stops getting stuck
        if max_trials == None: 
            self.max_trials = 0.6 * self.size * self.dims
        else:
            self.max_trials = max_trials 
            
            
        #assigns properties of optimisation problem
        self.evaluate = func 
        self.lower = lower
        self.upper = upper 
        
        #initialises current best and its solution vector
        self.best = sys.float_info.max                                         #think this creates a structure to hold
        self.solution = None                                                   #float info 
        
        #creates a beehive object
        self.population = [bee(lower, upper, func) for i in range(self.size)]
        
        #initialises best solution vector 
        self.find_best()

        
        #computes selection probability 
        self.compute_prob()
        
        #sets verbosity 
        self.verbose = verbose 
        
        
    def run(self):
        """This function is called to run the ABC algorithm"""

        #an empty dictionary with two unassigned keys is created. Keys relate to global best value and mean value of sol.
        cost = {} ; cost["best"] = [] ; cost["mean"] = []           

        #runs the algorithm for the specified number of max iterations 
        for i in range(self.max_its):
            
            #employee phase - occurs size number of times - defined below in __init__. Calls send_employees function.  
            for j in range(self.size):
                self.send_employees(j)
            
            
            #onlookers phase. Calls onlookers function defined below
            self.send_onlookers()
            
            #scouts phase. Calls scout function defined below
            self.send_scouts()
            
            #computes best path. Calls function defined below
            self.find_best()
            
            #stores convergence information 
            cost["best"].append(self.best)
            cost["mean"].append(sum(bee.value for bee in self.population) / self.size)

            #prints progress information by calling verbose function 
            if self.verbose:
                self.verbose(i, cost)

        return cost        
        
        
        
    def find_best(self):
        """finds current best bee candidate"""
        
        values = [bee.value for bee in self.population]
        index = values.index(min(values))                                      #finds index of lowest value (ie best) within
        if values[index] < self.best:
            self.best = values[index]
            self.solution = self.population[index].vector
                                                                      
        
    def compute_prob(self):
        """computes realtive chance solution will be chosen by onlooker bees after waggle dance"""
        
        #retrieves fitness of each bee within hive 
        values = [bee.fitness for bee in self.population]
        
        total = 0 
        for i in range(self.size):
            total = total + values[i]
        #computes probability solution is chosen in same way as original Karaboga paper circa. 2005
        #ie via "roulette wheel" selection. This use naive implimentation. See paper for upgrades. 
        
        self.selection_prob = [v / total for v in values]
        
    def send_employees(self, index):
        """This function uses crossover and mutation to create new solutions from employed bee initial values. If mutated
        values are better than originals, mutated values are taken forward"""
    
   
        #deepcopies current bee solution vector - avoids decimating program due to mutability of lists etc. 
        ghost_bee = copy.deepcopy(self.population[index])                      #index passed to function in run fn. 
        
        #random crossover point selected from ghost_bee solution vector. "self.dim -1" to avoid exact copies 
        d = random.randint(0, self.dims -1)
        
        #selects other 'parent' bee randomly  
        ghost_bee2 = index 
        while ghost_bee2 == index:
            ghost_bee2 = random.randint(0, self.size-1)                        #pop defined 0 --> 1-self.size previously   
            
        #produces mutant offspring by calling mutate function, defined below 
        ghost_bee.vector[d] = self.mutate(d, index, ghost_bee2)
        
        #checks boundaries for mutant by calling check function 
        ghost_bee.vector = self.check(ghost_bee.vector, dim = d)
        
        #computes fitness of mutant using previous fitness function 
        ghost_bee.value = self.evaluate(ghost_bee.vector)                      #passes mutant value to function 
        ghost_bee.calcFitness()
        
        #'deterministic crowding' - method to stop removal of parents if mutants not that good 
        if ghost_bee.fitness > self.population[index].fitness:
            self.population[index] = copy.deepcopy(ghost_bee)
            self.population[index].counter = 0                                 #no need to change counter as replacement 
        else:
            self.population[index].counter += 1 
            
    def send_onlookers(self):
        """Function defines as many onlookers as employed bees. Onlookers attempt to locally improve employed bees 
        solutions, following selection via waggle dance. If they find improvement, they communicate this with original 
        employed bee."""
        
        #send onlookers 
        num_onlookers = 0 ; beta = 0 
        while num_onlookers < self.size:
            
            #draws random number bwetween 0 & 1
            phi = random.uniform(0,1)
            
            #increments roulette wheel parameter beta 
            beta += phi * max(self.selection_prob)
            beta %= max(self.selection_prob)
            
            #selects new onlooker based on waggle dance 
            index = self.select(beta)                                          #value of beta fed to select() 
            
            #sends the onlooker using same code as employee 
            self.send_employees(index)
            
            #increments number of onlookers 
            num_onlookers += 1 
            
            
    def select(self, beta):
        """This is the function which implements the waggle dance phase. """

        #calculates probability intervals online - ie recalculated for each onlooker
        # probas = self.compute_prob()
        self.compute_prob()

        #selects new potential onlooker bee 
        for index in range(self.size):
            if beta < self.selection_prob[index]:
                return index 
            
            
    def send_scouts(self):
        """Scout phase. Identifies bees who's abandonment count exceeds present max trials limit.""" 
        
        #retrieves total number of trials for all bees 
        trials = [self.population[i].counter for i in range(self.size)]
        
        #identifies bee with greatest number of trials 
        index = trials.index(max(trials))
        
        #checks if this nimber exceeds preset limit max_trials 
        if index > self.max_trials:
            
            #randomly creates new scout 
            self.population[index] =  bee(self.lower, self.upper, self.evaluate)
            
            #sends scout to exploit its new solution vector 
            self.send_employees(index)
            
    
    def mutate(self, dim, current_bee, other_bee):
        """Function that handles the crossover & mutation process in employee bee processing"""
        
        
        #fairly arbritrary - takes original, then adds a random amount on and takes the difference between them as 
        #the mutation 
        return self.population[current_bee].vector[dim] + (random.random() - 0.5) *2 * \
        (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim])
        
    def check(self, vector, dim = None):
        """Checks solution vector is contained within the lower and upper bounds"""
        
        if dim == None:
            _range = range(self.dim)
        else:
            _range = [dim]

        for i in _range:
            
            #checks lower bound
            if vector[i] < self.lower[i]:
                vector[i] = self.lower[i]
        
            #checks upper bound
            elif vector[i] > self.upper[i]:
                vector[i] = self.upper[i]
            
        return vector 
    
    def verbose(self, its, cost):
        """Displays computation information"""
        
        msg = "# Iter = {} | Best Evaluation Value = {} | Mean Evaluation Value = {} "
        print(msg.format(int(its), cost["best"][its], cost["mean"][its]))

# ---- END
        
                        
            
        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
