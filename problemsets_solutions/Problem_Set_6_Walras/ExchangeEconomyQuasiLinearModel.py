from types import SimpleNamespace
import numpy as np
from scipy import optimize
from ExchangeEconomyModel import ExchangeEconomyModelClass

class ExchangeEconomyModelQuasiLinearClass(ExchangeEconomyModelClass):

    # uA(x1A,x2A) = ln(x1A) + alpha*x2A
    # uB(x1B,x2B) = ln(x1B) + beta*x2B

    ######################
    # utility and demand # 
    ######################

    def utility_A(self,x1A,x2A):
        """
        Utility function for agent A.
        """        
        return super().utility_A(x1A, x2A)
    
    def x2A_indifference(self,uA,x1A):

        return super().x2A_indifference(uA, x1A)   
     
    def utility_B(self,x1B,x2B):
        """
        Utility function for agent B.
        """

        return super().utility_B(x1B, x2B)
    
    def x2B_indifference(self,uB,x1B):

        return super().x2B_indifference(uB, x1B)   

    def demand_A(self,p1):
        """
        Demand for good 1 and 2 for agent A.
        """

        return super().demand_A(p1)
        
    def demand_B(self,p1):
        """
        Demand for good 1 and 2 for agent B.
        """

        return super().demand_B(p1)
            
    def solve_dictator_B(self):
        """ 
        Solve the dictator problem for agent A.
        """

        return super().solve_dictator_B()            
    
    def indifference_curve_A(self,ax,w1,w2,color='blue',label='A'):
        
        ua = self.utility_A(w1,w2)

        x1A = np.linspace(0.001,0.999,1000)
        x2A = self.x2A_indifference(ua, x1A)

        mask_inside_box = (x2A < 1) & (x2A > 0)
        ax.plot(x1A[mask_inside_box],x2A[mask_inside_box],
                label = label,color='orange')

    def indifference_curve_B(self,ax,w1,w2,color='blue',label='A'):

        ub = self.utility_B(w1,w2)

        x1B = np.linspace(0.001,0.999,1000)
        x2B = self.x2B_indifference(ub,x1B)

        mask_inside_box = (x2B < 1)

        
        ax.plot(x1B[mask_inside_box],x2B[mask_inside_box], color=color,
                label = label)

    def plot_improvement_set(self,ax,w1,w2):

        par = self.par 

        ua = self.utility_A(par.w1A,par.w2A)
        ub = self.utility_B(1-par.w1A,1-par.w2A)

        x1AB = np.linspace(0.001,0.999,1000)
        x2A = self.x2A_indifference(ua,x1AB)
        x2B = self.x2B_indifference(ub,1-x1AB)

        I = x2A < 1-x2B # in between indifference curves
        I &= x2A > 0 # in box
        I &= x2A < 1.0

        x = x1AB[I]
        y1 = x2A[I]
        y2 = np.fmin(1-x2B[I],1.0)
        ax.fill_between(x,y1,y2,color='gray',alpha=0.5)

    def add_legend(self, ax_a, ax_b):
        handles_a, labels_a = ax_a.get_legend_handles_labels()
        handles_b, labels_b = ax_b.get_legend_handles_labels()
        handles_combined = handles_a + handles_b
        labels_combined = labels_a + labels_b

        ax_a.legend(
            handles_combined,
            labels_combined,
            loc='lower left',
            facecolor='white',
            framealpha=1.0
        )

    def solve_dictator_A(self):
        
        par = self.par

        obj = lambda x: -self.utility_A(x[0],x[1])


        # constraint that B must be as good or better, then initially
        ub_init = self.utility_B(1-par.w1A,1-par.w2A)    
        # ineq means that the function should be 0 or greater (meaning that utility of the allocation
        # should be the same as ub:init or greater)
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0],1-x[1]) - ub_init})

        # bounding the consumption of x1A and x2A
        bounds = [(0,1),(0,1)]

        #minimizing the objective function -> maximizing the ultility of A
        sol = optimize.minimize(obj,[par.w1A,par.w2A],constraints=constraints,
                                bounds=bounds,
                                method='SLSQP')
        
        return sol

    def solve_dictator_B(self):
        
        par = self.par 

        obj = lambda x: -self.utility_B(x[0],x[1])


        # constraint that A must be as good or better, then initially
        ua_init = self.utility_A(par.w1A,par.w2A)    
        # ineq means that the function should be 0 or greater (meaning that utility of the allocation
        # should be the same as ub:init or greater)
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_A(1-x[0],1-x[1]) - ua_init})

        # bounding the consumption of x1B and x2B
        bounds = [(0,1),(0,1)]

        #minimizing the objective function -> maximizing the utility of B
        sol = optimize.minimize(obj,[1-par.w1A,1-par.w2A],constraints=constraints,
                                bounds=bounds,
                                method='SLSQP')
        
        return sol