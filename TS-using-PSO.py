import numpy as np
import matplotlib.pyplot as plt
import copy
##############################################################################33
def initpop(npop,x_max,x_min,v_max,dim):
    # Use this function to generate the initial population for the PSO
    #npop: population size
    #x_max: the upper limit for each decision variable (positions). [10,12]
    #x_min: the lower limit for each decision variable (positions). [1,2]
    #v_max: the upper limit for each decision variable (velocity). [2,4]
    #consider that the lower limit of the speed is equal to negative the upper limit
    #dim: number of decision variables
    x_id = np.zeros((npop,dim),dtype=int)
    v_id = np.zeros((npop,dim),dtype=int)
    for i in range(dim):
        x_id[:,i] = np.random.randint(x_min[i],x_max[i],(npop))
        v_id[:,i] = np.random.randint(-1*v_max[i],v_max[i],(npop))
    return x_id,v_id
    #x_id: the initial positions. Array of arrays of npop*dim
    #v_id: the initial velocities. Array f arrays of npop*dim
###############################################################################3
def fitCalc(x_i,w):
    # Use this function to calculate the fitness for the particle(schedule)
    #The function is Min schedule fitness which is calculated by sum of (constraint weight * number of violations)
    #x_i: single particle(schedule) position
    fitness  = (w[0]*x_i[0])+((x_i[1]*w[1])+(x_i[2]*w[2])+(x_i[3]*w[3])+(x_i[4]*w[4]))
    return fitness
    #fitness: the fitness value of a signle particle(schedule).
############################3#####################################################
def updatePid(x_i,x_fitness,p_i,particle_bestFit):
    #Use this function to find single particle best position (particle own history)
    #x_i: single particle position.
    #p_i: the particle best position across all the previous generations.
    #particle_best: particles best fintess values across all the previouse generations.
    if (x_fitness < particle_bestFit):
        p_i = copy.deepcopy(x_i)
    return p_i
    #pi: the particle best position.
####################################################################################3
def updatePgd(p_i,particle_bestFit,p_g,global_bestFit):
    #Use this function to find the best position in the population
    #p_i: a single particle best position
    #particle_bestFit: a particle fitness value associated to p_i.
    #p_g: a vector of 1*dim of representing the best position in the population across all the previouse generations
    #global_bestFit: fitness value associated to the p_g in the same neighborhood
    if (particle_bestFit < global_bestFit):
        p_g = copy.deepcopy(p_i)
        global_bestFit = particle_bestFit
    return p_g,global_bestFit
    #p_g: the best position in the neighborhood.
    #global_bestFit: the best fitness in the neighborhood.
########################################################################################
def updateVidXid(p_i,p_g,x_i,v_i,c_cog,c_soc,dim):
    v_min = -1*v_max[0]
    vmax = v_max[0]
    xmin = x_min[0]
    xmax = x_max[0]
    #Use this function to calculate new particle velocity and position\n",
    #p_i: the particle best position across all the previouse generations.\n",
    #p_g: a vector of 1*d of the best position in the neighborhood across all the previouse generations\n",
    #x_i: single particle position.\n",
    #v_i: single particle velocity.\n",
    #c_cog: cognitive component accerlaration constant\n",
    #c_soc: social component accerlaration constant\n",
    r_cog = np.random.random(dim)
    r_soc = np.random.random(dim)
    v_i = np.array(v_i) + (c_cog * np.multiply(r_cog, np.subtract(p_i,x_i))) + (c_soc * np.multiply(r_soc, np.subtract(p_g,x_i)))
    v_i = v_i.astype(int)
    v_i = np.where(v_i>v_min,v_i,v_min)
    v_i = np.where(v_i<vmax,v_i,vmax)
    x_i = np.array(x_i) + v_i
    x_i = x_i.astype(int)
    x_i = np.where(x_i<xmax,x_i,xmax)
    x_i = np.where(x_i>xmin,x_i,xmin)
    return x_i,v_i
#################################################################################################
def PSO(numItr,npop,nsize,w,x_max,x_min,v_max,dim,c_cog,c_soc):
    #Use this function to put all the PSO algorithm together for number of iterations\n",
    #numItr: number of iterations.(generations)\n",
    #npop: population size\n",
    #x_max: the upper limit for each decision variable (positions). [10,12]\n",
    #x_min: the lower limit for each decision variable (positions). [1,2]\n",
    #v_max: the upper limit for each decision variable (velocity). [2,4]\n",
    #c_cog: cognitive constant (c1)
    #c_soc: social constant (c2)
    #dim: the number of decision variable.
    #nsize : the neighborhood size (i assume it equal to (3))
    #Intialize\n",
    best_hist = np.zeros(numItr,dtype=float)
    dn  =npop//nsize # Represent the number of neighborhoods.\n",
    x,v= initpop(npop,x_max,x_min,v_max,dim)
    x = np.array_split(x, dn) # pop split into number of x's (neighborhoods)\n",
    v = np.array_split(v, dn) # velocity split as same as X\n",
    p = np.zeros((nsize,dim), dtype = float)  # initial particle best position across all the previous generations\n",
    xp = x[0]
    p[0] = xp[0] #to set p = x in the first iteration\n",
    p[1] = xp[1]
    p[2] = xp[2]
    p_g = np.zeros((dn,dim),dtype = float) #best position across all particles in all neighborhood
    bestfit_neighborhood = np.ones(dn) #best_fit across the particles in same neighborhood\n",
    for i in range(len(bestfit_neighborhood)):
        bestfit_neighborhood[i] = 10000 #Set a initial value to the bestfit across the particles in same neighborhood (to be minimized)
    local_bestfit = 1000000000 #Set a initial value to the local_bestfit across all neighborhoods(all pop)  (to be minimized)
    #repeat till number of iterations\n",
    for iteration in range(numItr):
        for j in range(dn): # to loop over the number of neighborhoods
            vd = v[j] # to take every neighborhood velocity
            xd = x[j] # to take every neighborhood posistion
        
       
        #Update particle best position and global best position\n",
        
            for i in range(nsize): #loop over all particles in same the neighborhood with the neighborhood size(nsize)\n",
                p[i] = updatePid(xd[i],fitCalc(xd[i],w),p[i],fitCalc(p[i],w))
                p_g[j],bestfit_neighborhood[j] = updatePgd(p[i],fitCalc(p[i],w),p_g[j],bestfit_neighborhood[j])
        #Update velocity and position\n",
            for i in range(nsize):  #loop over all particles in same the neighborhood with the neighborhood size(nsize)\n",
                xd[i],vd[i] = updateVidXid(p[i],p_g[j],xd[i],vd[i],c_cog,c_soc,dim)
        local_bestfit  = min(bestfit_neighborhood)  #get the bestfit over all neighborhoods and set it as local_fit for the problem\n",
        index = bestfit_neighborhood.argmin() # to get the position of  P_g which matching the local best PSO\n",
        best_hist[iteration] = local_bestfit #append it to the best_history\n",
        
    return  p_g[index], local_bestfit, best_hist
        #p_g[index]: the position with the best fitness in the final generation.\n",
        #local_bestfit: value associated to p_g[index]"
#######################################################################################################
numItr = 100
npop = 50
nsize = 3
w=[1.5,0.5,0.24,0.24,0.3]
x_max = [10,10,10,10,10]
x_min = [0,0,0,0,0]
v_max = [8,8,8,8,8]
dim = 5
c_cog = 1.7
c_soc = 1.7
      
      
     
p_g, local_bestfit,best_hist = PSO(numItr,npop,nsize,w,x_max,x_min,v_max,dim,c_cog,c_soc)

print(p_g)
print(local_bestfit)
print(best_hist)
plt.plot(best_hist)
plt.xlabel("iterations")
plt.ylabel("fitness value")
plt.show()
        
#Based on what i understood(that we have to search & get the local best PSO over the particles in same neighborhood(with ring topology) And The social component reflects information exchanged within the\n",
#neighborhood of the particle).
#So simply : i splitted the whole pop into n neighborhoods and every neighborhood have a n particles and we can get the local best PSO by getting the best_fit over every neighborhood and by comparing them we can get the local best PSO.\n",
#I hope I have explained to you a good explanation about what i did
