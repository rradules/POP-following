# This file contains planning code for the Fish-Wood problem under ESR.
# The Fish-Wood problem is parameterised by:
### pfish : the probability of getting fish while in the river state (0)
### pwood : the probability of getting timber while in the woods state (1)
### the planning horizon
### the utility function u (here defined as one standard function)

#Author: Diederik M. Roijers (Vrije Universiteit Brussel & HU University of Applied Sciences Utrecht)
#Written in September, 2020AD


#This is a helper decorator to memoize the dynamic programming results in the recursive implementation
def memoize(func):
    cache = dict()
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func

#The utility of the optimal policy, expressed as a recursive policy
#the first four arguments are fixed, but time (t), state (s), and accrued fish (rfish) and wood (rwood) are what is conditioned on.
@memoize
def conditioned_optimal(pfish, pwood, horizon, u, t=0, s=0, rfish=0, rwood=0):
    if(t>=horizon):
        return u(rfish,rwood)
    if (s==0):
        #the difference between x and y is not moving or moving (to 1), i.e., the action
        x = pfish*conditioned_optimal(pfish, pwood, horizon, u, t+1, 0, rfish+1, rwood)
        x = x + (1-pfish)*conditioned_optimal(pfish, pwood, horizon, u, t+1, 0, rfish, rwood)
        y = pfish * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 1, rfish + 1, rwood)
        y = y + (1 - pfish) * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 1, rfish, rwood)
        return max([x,y])
    if (s==1):
        # the difference between x and y is moving (to 0) or not moving, i.e., the action
        x = pwood * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 0, rfish, rwood+1)
        x = x + (1 - pwood) * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 0, rfish, rwood)
        y = pwood * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 1, rfish, rwood+1)
        y = y + (1 - pwood) * conditioned_optimal(pfish, pwood, horizon, u, t + 1, 1, rfish, rwood)
        return max([x, y])

def unconditioned_policy(pfish, pwood, horizon, tswitch, u):
    #Because in an unconditioned policy the only thing that counts is the number of timesteps
    #that is spent in each state (the rest can't be conditioned on) we can express the
    #policy as /the timestep where the agent will move to the woods/ (so once, because it does not matter)
    #and the return distributions as two independent distributions over the returns of fish, and the returns of wood
    #the product of these distributions are used to calculate the utility of the unconditioned policy
    #where the time to move to the woods is tswtich.
    pr_fish =  [0]*(tswitch+1) #for zero timesteps, the fish-returns for anything bigger than 0 has probability 0
    pr_fish[0] = 1.0 #and zero fish has probability one
    for j in range(tswitch): #now for each timestep spent at the river
        for i in range(tswitch, -1, -1):
            x = pr_fish[i]*(1.0-pfish) #we can either catch nothing
            y = 0 if i==0 else pr_fish[i-1]*pfish #or we can
            pr_fish[i] = x+y
    #and the same for wood for the rest of the timesteps until the horizon:
    from_tswitch = horizon-tswitch
    pr_wood = [0] * (from_tswitch+1)
    pr_wood[0] = 1
    for j in range(from_tswitch):
        for i in range(from_tswitch, -1, -1):
            x = pr_wood[i] * (1 - pwood)
            y = 0 if i == 0 else pr_wood[i - 1] * pwood
            pr_wood[i] = x + y
    #and then marginalising over the joint distribution to get to the utility:
    sum_result = 0
    for i in range(len(pr_fish)):
        for j in range(len(pr_wood)):
            sum_result = sum_result + u(i,j)*pr_fish[i]*pr_wood[j]
    return sum_result

#The standard (non-linear) utility function for Fish-Wood, requiring two wood to bake 1 fish, baked fish being the utility.
def fishwood_util_std(fish, wood):
    return min(wood//2, fish)

def fishwood_util_linear_w(fish, wood, wf, ww):
    return fish*wf + wood*ww

if __name__ == '__main__':
    pfish = 0.25
    pwood = 0.65
    horiz = 13
    linu = lambda f,w : fishwood_util_linear_w(f, w, 0.9, 0.1) #this is how you define a linear utility function
    #note that for a linear utility function the utility conditioned and unconditioned is the same
    #only for nonlinear utility functions, such as fishwood_util_std is there a difference.
    #u = linu
    u = fishwood_util_std

    #First, let's find the utility of the best policy that does not condition on the accrued rewards
    util = max([unconditioned_policy(pfish, pwood, horiz, sw, u) for sw in range(horiz+1)])
    print("Utility best unconditioned policy: "+ str(util))
    #And then let's find the utility of the optimal policy /with/ conditioning on the accrued rewards
    util2 = conditioned_optimal(pfish, pwood, horiz, u)
    print("Utility best conditioned policy: "+ str(util2))

    print("Proportional difference: "+ str((util2-util)/util2) )

