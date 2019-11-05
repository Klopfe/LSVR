""" SMO algorithm for Isotonic SVR"""
import numpy as np
import warnings
from numba import jit
import cProfile

def SMO_nuSVR_constrained(A,p,Q,X,tau,version,C=1,nu=0.5,max_it=1000):
    

    l,n=X.shape
    #STEP 1:
    d=Q.shape[0]
    theta=np.repeat(0.0,d)
    theta=theta.flatten()
    somme=(C*nu)/2
    for i in range(l):
        theta[i]=min(somme,C/l)
        theta[i+l]=min(somme,C/l)
        somme=somme-theta[i]    
    #STEP 2:
 
    F=np.dot(Q,theta)+p
   
    I_up=np.argwhere(theta[0:l]<(C/l))
    I_low=np.argwhere(theta[0:l]>0)

    I_up_star=np.argwhere(theta[l:(2*l)]<(C/l))+l
    I_low_star=np.argwhere(theta[l:(2*l)]>0)+l

    i=I_up[np.argmin(F[I_up])]
    j=I_low[np.argmax(F[I_low])]

    i_star=I_up_star[np.argmin(F[I_up_star])]
    j_star=I_low_star[np.argmax(F[I_low_star])]


    
    Delta1=F[j]-F[i]
    Delta2=F[j_star]-F[i_star]
    Delta3=-np.min((np.min(F[(2*l):d]),0.0))

    Delta=np.max((Delta1,Delta2,Delta3))
    
    k=0
    if(version=="random"):
        case=np.random.randint(0,3)
    elif(version=="cyclic"):
        case=k%3
    else:
        case=np.argmax((Delta1,Delta2,Delta3))
    
    
    #STEP 3:
    energie_dual=[]
    #energie_dual.append(0.5*np.dot(np.transpose(theta),np.dot(Q,theta))+np.dot(np.transpose(p),theta))

    sub=[]
    delta=[]
    
    beta=np.array(theta)

    while (Delta>tau and k<max_it):
         
        if(case==0):

            theta, t=subproblem12(theta,F,Q,X,i,j,C,nu)
            
            F=F+np.multiply(t,(Q[:,i]-Q[:,j])).flatten()
            
        elif(case==1):
            
            theta, t=subproblem12(theta,F,Q,X,i_star,j_star,C,nu)
            F=F+np.multiply(t,(Q[:,i_star]-Q[:,j_star])).flatten()
           
        elif(case==2):
            s=(np.argmin(F[(2*l):d])+2*l)
            theta=subproblem3(Q,F,A,theta,s)
            F=F+np.multiply((theta[s]-beta[s]),Q[:,s]).flatten()
           
 
        sub.append(case)    
        
        beta=np.array(theta)
        I_up=np.argwhere(theta[0:l]<(C/l))
        I_low=np.argwhere(theta[0:l]>0)

        I_up_star=np.argwhere(theta[l:(2*l)]<(C/l))+l
        I_low_star=np.argwhere(theta[l:(2*l)]>0)+l
        i=I_up[np.argmin(F[I_up])]
        j=I_low[np.argmax(F[I_low])]
        i_star=I_up_star[np.argmin(F[I_up_star])]
        j_star=I_low_star[np.argmax(F[I_low_star])]

        Delta1=F[j]-F[i]
        Delta2=F[j_star]-F[i_star]
        Delta3=-np.min((np.min(F[(2*l):d]),0.0))
        
        
        
      
        delta.append(Delta)
        k=k+1
        if(version=="random"):
            case=np.random.randint(0,3)
        elif(version=="cyclic"):
            case=k%3
        else:
            case=np.argmax((Delta1,Delta2,Delta3))
    
        Delta=np.max((Delta1,Delta2,Delta3))
        
        energie_dual.append(0.5*np.dot(np.transpose(theta),np.dot(Q,theta))+np.dot(np.transpose(p),theta))
       

            #computing the primal solution given at each step and its energie

  
    alpha=theta[0:l]
    alpha_star=theta[(l):(2*l)]

    support=np.where((alpha.flatten()-alpha_star.flatten())!=0)
    support_vectors=X[support,:]
    dual_coef=theta
   
        #Implement the intercept calculation
    intercept=0.0
    if(k==max_it):
        warnings.warn('SMO algorithm for SSVR reached maximal iteration')
        
    return [support,support_vectors,dual_coef,intercept, energie_dual,k,sub,delta]


def subproblem12(theta,F,Q,X,i,j,C=1.0,nu=0.5):
    l,n=X.shape
    
    #STEP 2 :
    
    denom=(Q[i,i]+Q[j,j]-2*Q[i,j])
 
    if(denom>0 and i<l and j<l):
        
        num=F[i]-F[j]
        theta2=-num/denom
    elif(denom>0 and i>=l and j>=l):
       
        num=F[i]-F[j]
        theta2=-num/denom
    
    if(denom == 0):
        theta2 = 0   
        #STEP 3 :  
    
    A=max(-theta[i],theta[j]-C/l)
   
    B=min(theta[j],C/l-theta[i])
    
    theta_new=theta
    temp=min(max(A,theta2),B)
    if (denom == 0):
        temp = 0
    theta_new[i]=theta[i]+temp
    theta_new[j]=theta[j]-temp
    return [theta_new, temp]   


def subproblem3(Q,F,A,theta,k):

    gamma = (-F[k]/2.0)+theta[k]
    if(gamma<0):
        gamma=0
    theta_new=theta
    theta_new[k]=gamma
    return theta_new



def SMO_solver(X,y,version,C=1.0,nu=0.5,max_it=1000,precision=0.001):
    l, n=X.shape
    
    y=y.flatten()
    A = np.zeros(((n-1),n))
    A[0:(n-1),0:(n-1)]=np.identity(n-1)
    for i in range(n-1):
        for j in range(n):
            if(j==(i+1)):
                A[i,j]=-1
      #quadratic term matrix
    Q=np.zeros(((2*l+n-1),(2*l+n-1)))

    Q[0:l,0:(2*l+n-1)]=np.concatenate((np.dot(X,np.transpose(X)),-np.dot(X,np.transpose(X)),np.dot(X,np.transpose(A))),axis=1)
    Q[l:(2*l),0:(2*l+n-1)]=np.concatenate((-np.dot(X,np.transpose(X)),np.dot(X,np.transpose(X)),-np.dot(X,np.transpose(A))),axis=1)
    Q[(2*l):(2*l+n-1),0:(2*l+n-1)]=np.concatenate((np.dot(A,np.transpose(X)),-np.dot(A,np.transpose(X)),np.dot(A,np.transpose(A))),axis=1)

#Linear term vector

    L=np.hstack((y,-y,np.repeat(0.0,n-1)))
    L=np.transpose(L)
    support, support_vectors,dual_coef, intercept, energie_dual, k,sub,delta = SMO_nuSVR_constrained(A,L,Q,X,precision,version,C,nu,max_it)
    return [support, support_vectors,dual_coef, intercept, energie_dual, k,sub,delta, A]












