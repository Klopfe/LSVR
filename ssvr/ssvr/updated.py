""" Generalized SMO that solves the Simplex Support Vector Regression"""

import numpy as np
import warnings


""" The problem we seek to solve is the following quadratic optimization problem
Given X, y, C, nu
n = number of rows of the matrix X

1/2 theta^{T}*Q*theta + l^{T}theta
subject to 0 <= theta_{i} <= C/n for i in {1,...,2n}
                theta_{i}>= 0 for i in {2n+1,..., 2n+p}
"""


""" The following function is the core of the SMO algorithm, it decides at each iteration
the block of variables that will be updated and change the corresponding gradient and variables

Input :
Q <- matrix of the quadratic term of the objective function, size (2n+p+1) x (2n+p+1)
p <- vector that is the linear term in the objective function, size (2n+p+1)
X <- the design matrix of the regression problem, size (n) x (p)
y <- variable to be explained by the regression problem, size (n)
tau <- tolerance of the algorithm, ie 10e-3
version <- can take three values "greedy", "random", "cyclic" which decides the strategy of the block selection at each step
C <- C hyperparameter of the Simplex nuSVR optimization problem, default value = 1
nu <- nu hyperparameter of the Simplex nuSVR optimization problem, default value = 0.5
max_it <- Threshold of the number of iterations of the algorithm, a warning pops if max_it is reached
"""

def SMO_nuSVR_Simplex(Q, l, X, y, tau, version, C = 1, nu = 0.5, max_it = 1000):

    n, p = X.shape

    # Initialization of theta in the feasible domain and number of iterations k

    theta = np.repeat(0.0, 2*n+p+1)
    somme = (C*nu)/2

    for i in range(n):
        theta[i] = min(somme, C/n)
        theta[i+n] = min(somme, C/n)
        somme = somme - theta[i]

    k = 0 #keeping track of the number of iterations


    # Initialization of empty lists that allows us to keep track of different values
    # dual energy, primal energy, theta at each iteration, selected block and so on

    energie_dual=[]
    energie_primal = []
    dual_coef = []
    dual_coef.append(theta)
    sub=[]
    delta=[]
    primal = []

    Delta = 10 # Initialization of the optimality score which has to be lower than tau to stop the iterations

    F = np.dot(Q,theta) + l # Initialization of the gradient of the objective function
    # loop that allows the algorithm to update the variables and converges to the solution of the Simplex NuSVR

    while (Delta > tau and k < max_it):
        # Checking optimality conditions defining the set of indices I_up, I_low, ...

        I_up = np.argwhere(theta[0:n]<(C/n))
        I_low = np.argwhere(theta[0:n]>0)

        I_up_star = np.argwhere(theta[n:(2*n)]<(C/n))+n
        I_low_star = np.argwhere(theta[n:(2*n)]>0)+n

        i = I_up[np.argmin(F[I_up])]
        j = I_low[np.argmax(F[I_low])]

        i_star = I_up_star[np.argmin(F[I_up_star])]
        j_star = I_low_star[np.argmax(F[I_low_star])]


        # Optimality score for each block of variables

        Delta1 = F[j]-F[i]
        Delta2 = F[j_star]-F[i_star]
        Delta3 = -np.min((np.min(F[(2*n):(2*n+p)]),0.0))
        Delta4 = np.abs(F[(2*n+p)])
        
        Delta = np.max((Delta1,Delta2,Delta3,Delta4)) #highest score of the 4 blocks


        # Block selection strategies

        if(version == "random"): # random
            case = np.random.randint(0,4)
        elif(version == "cyclic"): #cyclic
            case = k%4
        else: #greedy 
            case = np.argmax((Delta1,Delta2,Delta3,Delta4)) 


        # updating the selected block 

        if(case == 0): # update in block 0, corresponding to indices 0,...,n-1
            if(Delta1 > tau):
                theta, t = subproblem12(theta,F,Q,X,i,j,C,nu)
                F = F + np.multiply(t,(Q[:,i]-Q[:,j]))
                
        elif(case == 1): #update in block 1, corresponding to indices n,...,2n-1
            if(Delta2 > tau):
                theta, t = subproblem12(theta,F,Q,X,i_star,j_star,C,nu)
                F = F + np.multiply(t,(Q[:,i_star]-Q[:,j_star]))
                
        elif(case == 2): #update in block 2, corresponding to indices 2n,...,2n+p-1
            if(Delta3 > tau):
                s = (np.argmin(F[(2*n):(2*n+p)])+2*n)
                theta = subproblem3(theta,X,s)
                F = F + np.multiply((theta[s]-dual_coef[k][s]),Q[:,s])
                
        elif(case == 3): #update in block 2, corresponding to indices 2n+p
            if(Delta4 > tau):
                theta = subproblem4(theta,X)
                F = F+np.multiply((theta[-1]-dual_coef[k][-1]),Q[:,-1])


        sub.append(case) #keeping track of selected block
        dual_coef.append(theta) #keeping track of dual iterates theta
        delta.append(Delta) #Keeping track of optimality score

        k = k+1
        # block dual variables
        alpha = np.copy(theta[0:n])
        alpha_star = np.copy(theta[(n):(2*n)])
        sigma = np.copy(theta[(2*n+p)])
        gamma = np.copy(theta[(2*n):(2*n+p)])


        #Computing corresponding primal variables using dual-primal equations relationship
        temp = alpha.flatten() * np.transpose(X) - alpha_star.flatten() * np.transpose(X)
        prim_opt = gamma.flatten() - np.repeat(sigma,n) - np.sum(temp,axis=1)
        
        epsilon = get_epsilon(alpha, alpha_star, C, X, y, prim_opt)
        beta0 = get_beta0(alpha, alpha_star, C, X, y, prim_opt, epsilon)
        xi, xi_star = get_xi(alpha, alpha_star, C, X,y, prim_opt, epsilon, beta0)
        prim = np.hstack([prim_opt,xi,xi_star,epsilon,beta0])
        primal.append(prim)
 
        # Keeping track of the value of the objective function primal and dual
        energie_dual.append(0.5*np.dot(np.transpose(theta),np.dot(Q,theta))+np.dot(np.transpose(l),theta))
        energie_primal.append(0.5*np.dot(np.transpose(prim[0:n]),prim[0:n])+(C*nu*prim[(2*n+p)]+(C/n)*np.sum(prim[p:(p+2*n)])))
          
    support = np.where((alpha.flatten()-alpha_star.flatten())!=0)
    support_vectors = X[support,:]
    

   
        #Implement the intercept calculation
    intercept = beta0
    if(k == max_it): #return warning if max_it is reached
        warnings.warn('SMO algorithm for SSVR reached maximal iteration')
        
    return [support,support_vectors,dual_coef,intercept, energie_dual,k,sub,delta, primal,energie_primal]



""" Function that solves the optimization problem considering that only theta_{i}
and theta_{j} are the variables, it corresponds to the update for blocks 0 and 1"""
"""
Input : 
theta <- Dual variable iterate, size (2n+p+1)
F <- gradient of the objective function of Simplex nuSVR problem, size (2n+p+1)
Q <- matrix of the quadratic term of the objective function, size (2n+p+1) x (2n+p+1)
X <- the design matrix of the regression problem, size (n) x (p)
i, j <- indices of the selected variables to update
C <- C hyperparameter of the Simplex nuSVR optimization problem, default value = 1
nu <- nu hyperparameter of the Simplex nuSVR optimization problem, default value = 0.5
"""

def subproblem12(theta, F, Q, X, i, j, C = 1.0, nu = 0.5):
    n = X.shape[0]
    
    # solution of unconstrained minimization problem with two variables to optimized
    denom = (Q[i,i]+Q[j,j]-2*Q[i,j])
    num = F[i]-F[j]
    theta2=-num/denom
    
    # Performing clipping to make sure that we stay in feasible domain
    
    A = max(-theta[i],theta[j]-C/n)
    B = min(theta[j],C/n-theta[i])
    temp = min(max(A,theta2),B)
   
    # Updating the variables

    theta[i] = theta[i]+temp
    theta[j] = theta[j]-temp

    return [theta, temp]   



""" Function that solves the optimization problem considering that only theta_{k}
is the variable, it corresponds to the update for blocks 2"""
"""
Input : 
theta <- Dual variable iterate, size (2n+p+1)
X <- the design matrix of the regression problem, size (n) x (p)
k <- index of the selected variable to update
"""

def subproblem3(theta, X, k):
    n, p = X.shape
    x = X[:,(k-(2*n))]
    gamma = theta[(2*n+p)]+(np.inner(theta[0:n]-theta[n:(2*n)],x))
    if(gamma < 0):
        gamma = 0
    
    theta[k] = gamma

    return theta


""" Function that solves the optimization problem considering that only theta_{2n+p+1}
is the variable, it corresponds to the update for blocks 3"""
"""
Input : 
theta <- Dual variable iterate, size (2n+p+1)
X <- the design matrix of the regression problem, size (n) x (p)
"""

def subproblem4(theta, X):
    n, p = X.shape
    sigma = np.sum(theta[(2*n):(2*n+p)])-np.inner(theta[0:n],np.sum(X,axis=1))+np.inner(theta[n:(2*n)],np.sum(X,axis=1))-1.0
    sigma = sigma/p
    
    theta[(2*n+p)] = sigma

    return theta



def SMO_solver(X, y, version, C = 1.0, nu = 0.5, max_it = 1000, tau = 0.001):
    n, p = X.shape

    # Construction of the quadratic matrix of the optimization problem
    Q=np.zeros(((2*n+p+1),(2*n+p+1)))

    Q[0:n,0:(2*n+p)]=np.concatenate((np.dot(X,np.transpose(X)),-np.dot(X,np.transpose(X)),-X),axis=1)
    Q[n:(2*n),0:(2*n+p)]=np.concatenate((-np.dot(X,np.transpose(X)),np.dot(X,np.transpose(X)),X),axis=1)
    Q[(2*n):(2*n+p),0:(2*n+p)]=np.concatenate((-np.transpose(X),np.transpose(X),np.identity(p)),axis=1)


    Q[:,(2*n+p)]=np.hstack((X.sum(axis=1),-X.sum(axis=1),np.repeat(-1.0,p),p))
    Q[(2*n+p),:]=np.transpose(np.hstack((X.sum(axis=1),-X.sum(axis=1),np.repeat(-1.0,p),p)))
    
    #Linear term vector
    l=np.hstack((y,-y,np.repeat(0.0,p),1.0))
    l=np.transpose(l)
    support, support_vectors,dual_coef, intercept, energie_dual, k,sub,delta, primal, energie_primal = SMO_nuSVR_Simplex(Q, l, X, y, tau, version, C, nu, max_it)
    return [support, support_vectors,dual_coef, intercept, energie_dual, k,sub,delta, primal, energie_primal]
                                                                                                                    


def get_epsilon(alpha, alpha_star, C, X, y, primal):
    i = np.argwhere((alpha>0) & (alpha<C/len(alpha)))
    
    i = i[0]
    j = np.argwhere((alpha_star > 0) & (alpha_star < C/len(alpha)))
    j = j[0]
    eps = (-y[i]+y[j]+np.inner(X[i,:],primal)-np.inner(X[j,:],primal))/2

    return eps

def get_beta0(alpha, alpha_star, C, X, y, primal, epsilon):
    i = np.argwhere((alpha>0) & (alpha<C/len(alpha)))
    i = i[0]
    beta0 = epsilon + y[i]  - np.inner(X[i,:],primal)

    return beta0


def get_xi(alpha, alpha_star, C, X, y, primal, eps, beta0):
    xi = np.zeros(len(alpha))
    xi_star = np.zeros(len(alpha_star))

    for k in range(len(alpha)):
        if alpha[k]== C/len(alpha):
            xi[k] = - eps -y[k] + np.inner(X[k,:],primal) + beta0
        if alpha_star[k] == C/len(alpha):
             xi_star[k] = - eps +y[k] - np.inner(X[k,:],primal) - beta0

    return [xi, xi_star]                                                                                                         