

import math

import numpy as np

from numpy import genfromtxt

import matplotlib.pyplot as plt

import pandas as pd



class Worldstat:




    def __init__(self, continent, country):

        self.continent = continent 

        self.country = country

    

    def print_all_stats(self):

        

        print("Continent name: {0}, Country name: {1}".format(self.continent, self.country))


class ODESolver:
    def __init__(self, f):
    #Wrap user’s f in a new function that always
    #converts list/tuple to array (or let array be array) 
        self.f = lambda u, t: np.asarray(f(u, t), float)
    
    def set_initial_condition(self, U0):
        if isinstance(U0, (float,int)): # scalar ODE
            self.neq = 1
            U0 = float(U0) 
        else:
            U0 = np.asarray(U0)
            self.neq = U0.size
        self.U0 = U0

    def solve(self, time_points):
        self.t = np.asarray(time_points) 
        N = len(self.t)
        if self.neq == 1: # scalar ODEs
            self.u = np.zeros(N)
        else: # systems of ODEs
            self.u = np.zeros((N,self.neq))
        # Assume that self.t[0] corresponds to self.U0
        self.u[0] = self.U0
        # Time loop
        for n in range(N-1): 
            self.n = n
            self.u[n+1] = self.advance() 
        return self.u, self.t

class ForwardEuler(ODESolver): 
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = t[n+1] - t[n]
        unew = u[n] + dt*f(u[n], t[n])
        return unew

class RungeKutta4(ODESolver): 
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = t[n+1] - t[n]
        dt2 = dt/2.0
        k1 = f(u[n], t)
        k2 = f(u[n] + dt2*k1, t[n] + dt2)
        k3 = f(u[n] + dt2*k2, t[n] + dt2)
        k4 = f(u[n] + dt*k3, t[n] + dt)
        unew = u[n] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4) 
        return unew

def main():

    print("\n***SIR COVID 19 model***\n")

    dataset1 = pd.read_csv('CONTINENTS.csv').to_numpy()  

    dataset2 = pd.read_csv('INFECTED.csv').to_numpy()  

    dataset3 = pd.read_csv('RVAL.csv').to_numpy()  

    

    for line in dataset1:

        print(line[0],'---',line[1])

    
    max_active = np.max(dataset2[1: ,2])

    max_index = np.where(dataset2 == max_active)[0][0]

    max_index2= dataset2[max_index][0]

    country_max_index= np.where(dataset1 == max_index2)[0][0]

    country_max = dataset1[country_max_index][1]
    

    min_active = np.min(dataset2[1: ,2])

    min_index = np.where(dataset2 == min_active)[0][0]

    min_index2= dataset2[min_index][0]

    country_min_index= np.where(dataset1 == min_index2)[0][0]

    country_min = dataset1[country_min_index][1]
    

    print('\nCovid 19 Statistics for 2021 Jan:')

    print('Maximum number of Covid19 Active Cases were', max_active, 'in', country_max) 
    
    print('Minimum number of Covid19 Active Cases were', min_active, 'in', country_min) 
   




    Worldstat = None

    while isinstance(Worldstat, type(None)):  

        continent_names = input("\nPlease enter the Continent: ")

        country_stats = dataset1  

        isInt = False

        isInt2 = False



        if continent_names.isalpha():

            isInt = True

        continent = ''

        name_matched = False

        while isinstance(Worldstat, type(None)): 

            if isInt:

                while not name_matched:

                    for continent_info in country_stats: # using a for-else loop

                        continent = continent_info[0]

                        if continent == continent_names:

                            

                            name_matched = True

                            break


                    else:

                        print('You must enter a valid Continent.')

                        continent_names = input("Please enter the correct Continent: ")

            break





        country_names = input('Please enter Country name: ')

        if country_names.isalpha():

            isInt2 = True

        
        

        name_matched = False

        while isinstance(Worldstat, type(None)):

            if isInt2:

                while not name_matched:

                    for continent_info in country_stats: # used for-else loop
                        # backslash below for continution of statement
                        if (continent_info[0] == continent_names) and \
                            (continent_info[1] == country_names):

                            name_matched = True

                            country_code = continent_info[2]

                            break

                    else:

                        print('You must enter a valid Country in the entered Continent.')

                        country_names = input("Please enter the correct Country: ")

            break

        

        break


        
    country_code = continent_info[2]
    
    index_N = np.where(dataset3 == country_code)

    country_index = (list(index_N[0]))[0] # get index of country code



    N = dataset3[country_index, 3] # import population of input country from dataset3(csv file)

    initial_infec = dataset2[country_index, 2] # import Number of people infected for input country from dataset2

    initial_removed = dataset2[country_index, 3] # import Total number of people removed from dataset 2

    initial_suscep = N - initial_infec - initial_removed

    R1= dataset3[country_index, 1] #import R1 from dataset3

    R2= dataset3[country_index, 2] #import R2 from dataset3

    Ro = (R1 + R2) / 2 #Average of R1 and R2 

    

    

    # A grid of time points (in days)

    t = np.linspace(0, 250, 250)

    def SIR_model(u,t): 
        gamma = 1/18
        beta = Ro * gamma
        S, I, R = u[0], u[1], u[2] 
        dS = -beta*S*I/N
        dI = beta*S*I/N - gamma *I
        dR = gamma *I
        return [dS,dI,dR]
    
    S0 = initial_suscep
    I0 = initial_infec
    R0 = initial_removed

    solver= RungeKutta4(SIR_model)
    solver.set_initial_condition([S0,I0,R0])
    time_points = np.linspace(0, 250, 250)
    u, t = solver.solve(time_points)
    S = u[:,0];  I = u[:,1]; R = u[:,2]

    print('\nSIR Graph Code\nSusceptible:1\nInfected:2\nRemoved:3\n')
    sir_graph = input('Please enter the code for the specific SIR Graph Required(Susceptible,Infected or Removed: ')

    if sir_graph == '1':
        plt.plot(t,S, c='blue')
        plt.xlabel('Number of Days')
        plt.ylabel('SIR population')
        plt.title('Susceptible S(t)')
        plt.show()
    elif sir_graph == '2':
        plt.plot(t,I, c='Orange')
        plt.xlabel('Number of Days')
        plt.ylabel('SIR population')
        plt.title('Infected I(t)')
        plt.show()
            
    elif sir_graph == '3':
        plt.plot(t,R, c='green')
        plt.xlabel('Number of Days')
        plt.ylabel('SIR population')
        plt.title('Removed R(t)')
        plt.show()
        

    plt.plot(t,S,t,I,t,R)
    plt.legend(('Susceptible S(t)', 'Infected I(t)', 'Removed R(t)'))
    plt.xlabel('Number of Days')
    plt.ylabel('SIR population')
    plt.title('SIR model for Covid-19 pandemic')
    plt.show()
        



if __name__ == "__main__":

    main()
