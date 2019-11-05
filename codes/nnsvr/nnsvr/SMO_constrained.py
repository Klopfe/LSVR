import numpy as np
import warnings


""" The problem we seek to solve is the following quadratic optimization problem
Given X, y, C, nu
n = number of rows of the matrix X

1/2 theta^{T}*Q*theta + p^{T}theta
subject to 0 <= theta_{i} <= C/l for i in {1,...,2l}
                theta_{i}>= 0 for i in {2l+1,..., 2l+p}
"""


""" The following function is the core of the SMO algorithm, it decides at each iteration
the block of variables that will be updated and change the corresponding gradient and variables

Input :
Q <- matrix of the quadratic term of the objective function, size (2l+n+1) x (2l+n+1)
p <- vector that is the linear term in the objective function, size (2l+n+1)
X <- the design matrix of the regression problem, size (l) x (n)
y <- variable to be explained by the regression problem, size (l)
tau <- tolerance of the algorithm, ie 10e-3
version <- can take three values "greedy", "random", "cyclic" which decides the strategy of the block selection at each step
C <- C hyperparameter of the Simplex nuSVR optimization problem, default value = 1
nu <- nu hyperparameter of the Simplex nuSVR optimization problem, default value = 0.5
max_it <- Threshold of the number of iterations of the algorithm, a warning pops if max_it is reached
"""
def SMO_nuSVR_constrained(p,Q,X,y,tau,version,C=1,nu=0.5,max_it=1000):
    

    l, n = X.shape
    # Initialization of theta in the feasible domain and number of iterations k

    theta = np.repeat(0.0,2*l+n)
    
    somme = (C*nu)/2
    for i in range(l):
        theta[i] = min(somme,C/l)
        theta[i+l] = min(somme,C/l)
        somme = somme - theta[i]    


    k = 0 #keeping track of the number of iterations

    # Initialization of empty lists that allows us to keep track of different values
    # dual energy, primal energy, theta at each iteration, selected block and so on

    energie_dual = []
    energie_primal = []
    sub = []
    delta = []
    primal = []

    Delta = np.inf # Initialization of the optimality score which has to be lower than tau to stop the iterations

    F = np.dot(Q,theta) + p # Initialization of the gradient of the objective function
    
    
    
    # loop that allows the algorithm to update the variables and converges to the solution of the Simplex NuSVR
 
    while (Delta>tau and k<max_it):

        # Checking optimality conditions defining the set of indices I_up, I_low, ...

        I_up = np.argwhere(theta[0:l]<(C/l))
        I_low = np.argwhere(theta[0:l]>0)

        I_up_star = np.argwhere(theta[l:(2*l)]<(C/l))+l
        I_low_star = np.argwhere(theta[l:(2*l)]>0)+l

        i = I_up[np.argmin(F[I_up])]
        j = I_low[np.argmax(F[I_low])]

        i_star = I_up_star[np.argmin(F[I_up_star])]
        j_star = I_low_star[np.argmax(F[I_low_star])]

        # Optimality score for each block of variables

        Delta1 = (F[j]-F[i])[0]
        Delta2 = (F[j_star]-F[i_star])[0]
        Delta3 = -np.min((np.min(F[(2*l):(2*l+n)]),0.0))
        Delta = np.max((Delta1,Delta2,Delta3))
    
        #Selecting the block in which the update will take place  
        case = np.argmax((Delta1,Delta2,Delta3)) 
    
    
    
        beta = np.copy(theta)


        # block dual variables
        alpha = np.copy(theta[0:l])
        alpha_star = np.copy(theta[(l):(2*l)])
        gamma = np.copy(theta[(2*l):(2*l+n)])


        #Computing corresponding primal variables using dual-primal equations relationship

        temp = alpha*np.transpose(X)-alpha_star*np.transpose(X)
        prim_opt = gamma-np.sum(temp,axis=1)
        
        epsilon = get_epsilon(alpha, alpha_star, C, X, y, prim_opt)
        beta0 = get_beta0(alpha, alpha_star, C, X, y, prim_opt, epsilon)
        xi, xi_star = get_xi(alpha, alpha_star, C, X,y, prim_opt, epsilon, beta0)
        prim = np.hstack([prim_opt,xi,xi_star,epsilon,beta0])
        primal.append(prim)
 

        # Keeping track of the value of the objective function primal and dual

        energie_dual.append(0.5*np.dot(np.transpose(theta),np.dot(Q,theta))+np.dot(np.transpose(p),theta))
        energie_primal.append(0.5*np.dot(np.transpose(prim[0:n]),prim[0:n])+(C*nu*prim[(2*l+n)]+(C/l)*np.sum(prim[n:(n+2*l)])))
          
        # updating the selected block 

        if(case == 0):
            if(Delta1 > tau):
                theta, t = subproblem12(np.copy(theta),F,Q,X,i,j,C,nu)
                F = F + np.multiply(t,(Q[:,i]-Q[:,j]))[:,0]
                
        elif(case == 1):
            if(Delta2 > tau):
                theta, t = subproblem12(np.copy(theta),F,Q,X,i_star,j_star,C,nu)
                F = F + np.multiply(t,(Q[:,i_star]-Q[:,j_star]))[:,0]
                
        elif(case == 2):
            if(Delta3 > tau):
                s = np.argmin(F[(2*l):(2*l+n)])+2*l
                theta = subproblem3(F, np.copy(theta),s)
                F = F + np.multiply((theta[s]-beta[s]),Q[:,s])
                
        
        sub.append(case)    

        delta.append(Delta)
        k = k + 1
            
            
  
   

    support = np.where((alpha-alpha_star)!=0)
    support_vectors = X[support,:]
    dual_coef = theta

   
    #return the intercept beta0
    intercept = beta0
    if(k == max_it):
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
def subproblem12(theta,F,Q,X,i,j,C=1.0,nu=0.5):
    l = X.shape[0]
    
    # solution of unconstrained minimization problem with two variables to optimized 
    denom = (Q[i,i]+Q[j,j]-2*Q[i,j])
    num = F[i]-F[j]
    theta2 = -num/denom
    
       
    # Performing clipping to make sure that we stay in feasible domain
    A = max(-theta[i],theta[j]-C/l)
    B = min(theta[j],C/l-theta[i])
    temp = min(max(A,theta2),B)

    # Updating the variables
    theta[i] = theta[i]+temp
    theta[j] = theta[j]-temp
    return [theta, temp]   


""" Function that solves the optimization problem considering that only theta_{k}
is the variable, it corresponds to the update for blocks 2"""
"""
Input : 
theta <- Dual variable iterate, size (2l+n+1)
F <- Gradient of objective function, size (2l+n+1)
k <- index of the selected variable to update
"""
def subproblem3(F, theta, k):
    
    gamma = -F[k]+theta[k]
    if(gamma < 0):
        gamma = 0
    theta[k] = gamma
    return theta



""" Function that solves the optimization problem considering that only theta_{2n+p+1}
is the variable, it corresponds to the update for blocks 3"""
"""
Input :
F <-  Gradient of objective function, size (2l+n+1)
theta <- Dual variable iterate, size (2l+n+1)
n <- int number of columns of X
"""
def subproblem4(F ,theta, n):
    
    sigma = -F[-1]/n+theta[-1]
    theta[-1] = sigma
    return theta


def SMO_solver(X,y,version,C=1.0,nu=0.5,max_it=1000,precision=0.001):
    l, n = X.shape

    # Construction of the quadratic matrix of the optimization problem
    Q = np.zeros(((2*l+n),(2*l+n)))

    Q[0:l,0:(2*l+n)] = np.concatenate((np.dot(X,np.transpose(X)),-np.dot(X,np.transpose(X)),-X),axis=1)
    Q[l:(2*l),0:(2*l+n)] = np.concatenate((-np.dot(X,np.transpose(X)),np.dot(X,np.transpose(X)),X),axis=1)
    Q[(2*l):(2*l+n),0:(2*l+n)] = np.concatenate((-np.transpose(X),np.transpose(X),np.identity(n)),axis=1)


   
    #Linear term vector

    L = np.hstack((y,-y,np.repeat(0.0,n)))
    L = np.transpose(L)
    support, support_vectors,dual_coef, intercept, energie_dual, k,sub,delta, primal, energie_primal = SMO_nuSVR_constrained(L,Q,X,y,precision,version,C,nu,max_it)
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


