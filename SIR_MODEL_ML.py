


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.signal import savgol_filter

# Keep existing ODESolver and other classes
class ODESolver:
    def __init__(self, f):
        self.f = lambda u, t: np.asarray(f(u, t), float)
    
    def set_initial_condition(self, U0):
        if isinstance(U0, (float,int)):
            self.neq = 1
            U0 = float(U0)
        else:
            U0 = np.asarray(U0)
            self.neq = U0.size
        self.U0 = U0

    def solve(self, time_points):
        self.t = np.asarray(time_points)
        N = len(self.t)
        if self.neq == 1:
            self.u = np.zeros(N)
        else:
            self.u = np.zeros((N,self.neq))
        self.u[0] = self.U0
        for n in range(N-1):
            self.n = n
            self.u[n+1] = self.advance()
        return self.u, self.t

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

class EnhancedSIRPredictor:
    def __init__(self, country_code, initial_data, population, R0):
        self.country_code = country_code
        self.initial_data = initial_data
        self.population = population
        self.R0 = R0
        self.scaler = MinMaxScaler()
        
    def extract_features(self, S, I, R, t):
        features = []
        targets = []
        window_size = 14  # Two weeks of history
        

        S_diff = np.gradient(S)
        I_diff = np.gradient(I)
        R_diff = np.gradient(R)
        

        S_acc = np.gradient(S_diff)
        I_acc = np.gradient(I_diff)
        R_acc = np.gradient(R_diff)
        
        # Calculate moving averages
        def moving_average(data, window=7):
            return np.convolve(data, np.ones(window), 'valid') / window
        
        S_ma = moving_average(S)
        I_ma = moving_average(I)
        R_ma = moving_average(R)
        
   
        pad_length = len(S) - len(S_ma)
        S_ma = np.pad(S_ma, (pad_length, 0), 'edge')
        I_ma = np.pad(I_ma, (pad_length, 0), 'edge')
        R_ma = np.pad(R_ma, (pad_length, 0), 'edge')
        
        for i in range(window_size, len(t)):
            feature_vector = []
            
            # Add historical values
            for j in range(window_size):
                idx = i - window_size + j
                feature_vector.extend([
                    S[idx]/self.population,
                    I[idx]/self.population,
                    R[idx]/self.population,
                    S_diff[idx],
                    I_diff[idx],
                    R_diff[idx],
                    S_acc[idx],
                    I_acc[idx],
                    R_acc[idx],
                    S_ma[idx]/self.population,
                    I_ma[idx]/self.population,
                    R_ma[idx]/self.population
                ])
            
            # Add derived features
            feature_vector.extend([
                self.R0,  # Basic reproduction number
                np.sum(I[i-7:i])/7,  # Weekly average of infected
                np.max(I[i-14:i])/self.population,  # Max infections in last 2 weeks
                np.sum(R[i-7:i] - R[i-8:i-1])/7,  # Weekly recovery rate
                I[i]/S[i] if S[i] > 0 else 0,  # Current infection rate
            ])
            
            features.append(feature_vector)
            targets.append([S[i]/self.population, 
                          I[i]/self.population, 
                          R[i]/self.population])
            
        return np.array(features), np.array(targets)
    
    def train_ml_model(self, S, I, R, t):
        X, y = self.extract_features(S, I, R, t)
        X_scaled = self.scaler.fit_transform(X)
        
        # The GradientBoostingRegressor is used as the base model 
  
        base_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
    
    def predict_future(self, S, I, R, days=80):
        window_size = 14
        future_S = list(S)
        future_I = list(I)
        future_R = list(R)
        
        # Initialize feature arrays for future predictions
        S_diff = np.gradient(future_S)
        I_diff = np.gradient(future_I)
        R_diff = np.gradient(future_R)
        
        S_acc = np.gradient(S_diff)
        I_acc = np.gradient(I_diff)
        R_acc = np.gradient(R_diff)
        
        def moving_average(data, window=7):
            return np.convolve(data, np.ones(window), 'valid') / window
        
        for _ in range(days):
  
            S_ma = moving_average(future_S[-21:])[-1]
            I_ma = moving_average(future_I[-21:])[-1]
            R_ma = moving_average(future_R[-21:])[-1]
            
            feature_vector = []
            
            # Compile historical window features
            for j in range(window_size):
                idx = len(future_S) - window_size + j
                feature_vector.extend([
                    future_S[idx]/self.population,
                    future_I[idx]/self.population,
                    future_R[idx]/self.population,
                    S_diff[idx],
                    I_diff[idx],
                    R_diff[idx],
                    S_acc[idx],
                    I_acc[idx],
                    R_acc[idx],
                    S_ma/self.population,
                    I_ma/self.population,
                    R_ma/self.population
                ])
            
    
            feature_vector.extend([
                self.R0,
                np.sum(future_I[-7:])/7,
                np.max(future_I[-14:])/self.population,
                np.sum(np.diff(future_R[-8:]))/7,
                future_I[-1]/future_S[-1] if future_S[-1] > 0 else 0,
            ])
            
            # Make predictions using the ML model
            feature_scaled = self.scaler.transform([feature_vector])
            prediction = self.model.predict(feature_scaled)[0]
            
            # Convert predictions back to absolute numbers

            next_S = prediction[0] * self.population
            next_I = prediction[1] * self.population
            next_R = prediction[2] * self.population
            
            # Apply smoothing and constraints to the predictions
            total = next_S + next_I + next_R
            scaling_factor = self.population / total if total > 0 else 1
            
            next_S *= scaling_factor
            next_I *= scaling_factor
            next_R *= scaling_factor
            
      
            future_S.append(next_S)
            future_I.append(next_I)
            future_R.append(next_R)
            

            S_diff = np.gradient(future_S)
            I_diff = np.gradient(future_I)
            R_diff = np.gradient(future_R)
            
            S_acc = np.gradient(S_diff)
            I_acc = np.gradient(I_diff)
            R_acc = np.gradient(R_diff)
        
        # Apply Savitzky-Golay filter for final smoothing of the curves 
        window = 15 
        future_S_smooth = savgol_filter(future_S[-days:], window, 3)
        future_I_smooth = savgol_filter(future_I[-days:], window, 3)
        future_R_smooth = savgol_filter(future_R[-days:], window, 3)
        
        return future_S_smooth, future_I_smooth, future_R_smooth

def plot_sir_with_projection(t, S, I, R, future_S, future_I, future_R, country_name):
    plt.figure(figsize=(15, 10))
    
    # Plot historical data
    plt.plot(t, S, 'b-', label='Historical Susceptible', alpha=0.7)
    plt.plot(t, I, 'r-', label='Historical Infected', alpha=0.7)
    plt.plot(t, R, 'g-', label='Historical Recovered', alpha=0.7)
    
    # Plot projections
    future_t = np.arange(len(t), len(t) + len(future_S))
    plt.plot(future_t, future_S, 'b--', label='Projected Susceptible')
    plt.plot(future_t, future_I, 'r--', label='Projected Infected')
    plt.plot(future_t, future_R, 'g--', label='Projected Recovered')
    
    plt.axvline(x=len(t), color='gray', linestyle='--', alpha=0.5)
    plt.text(len(t), plt.ylim()[1], 'Projection Start', rotation=90, verticalalignment='top')
    
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title(f'SIR Model with Enhanced ML Projection - {country_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
 
    plt.fill_between(future_t, future_I*0.8, future_I*1.2, color='r', alpha=0.1)
    
    plt.show()

def main():
    # Load data
    dataset1 = pd.read_csv('CONTINENTS.csv')
    dataset2 = pd.read_csv('INFECTED.csv')
    dataset3 = pd.read_csv('RVAL.csv')
    
    def process_country(continent_names, country_names):
        country_info = dataset1[
            (dataset1['Continent'] == continent_names) & 
            (dataset1['Country'] == country_names)
        ].iloc[0]
        
        country_code = country_info['codes']
        country_data = dataset2[dataset2['Country code'] == country_code].iloc[0]
        country_params = dataset3[dataset3['Country Code'] == country_code].iloc[0]
        
        N = country_params['Population']
        initial_infec = country_data['Number of people infected  ']  # Note the extra spaces
        initial_removed = country_data['Total  number of people removed ']  # Note the extra spaces
        initial_suscep = N - initial_infec - initial_removed
        Ro = (country_params['Ro1'] + country_params['Ro2']) / 2
        
        # Calculate SIR model
        t = np.linspace(0, 250, 250)
        
        def SIR_model(u, t):
            gamma = 1/18
            beta = Ro * gamma
            S, I, R = u[0], u[1], u[2]
            dS = -beta*S*I/N
            dI = beta*S*I/N - gamma*I
            dR = gamma*I
            return [dS, dI, dR]
        
        solver = RungeKutta4(SIR_model)
        solver.set_initial_condition([initial_suscep, initial_infec, initial_removed])
        u, t = solver.solve(t)
        S, I, R = u[:,0], u[:,1], u[:,2]
        
        # Create and train ML predictor - Now passing Ro
        predictor = EnhancedSIRPredictor(country_code, u, N, Ro)
        predictor.train_ml_model(S, I, R, t)
        
        # Generate future predictions
        future_S, future_I, future_R = predictor.predict_future(S, I, R)
        
        # Plot results
        plot_sir_with_projection(t, S, I, R, future_S, future_I, future_R, country_names)
        
        return S, I, R, future_S, future_I, future_R

    print("\n***SIR COVID 19 model with ML Projections***\n")
    
    # Print available continents and countries
    for continent in dataset1['Continent'].unique():
        print(f"\n{continent}:")
        countries = dataset1[dataset1['Continent'] == continent]['Country'].tolist()
        print(", ".join(countries))
    
    while True:
        continent_names = input("\nPlease enter the Continent: ")
        if continent_names in dataset1['Continent'].unique():
            country_names = input('Please enter Country name: ')
            if country_names in dataset1[dataset1['Continent'] == continent_names]['Country'].values:
                process_country(continent_names, country_names)
                break
            else:
                print('Invalid country name. Please try again.')
        else:
            print('Invalid continent name. Please try again.')

if __name__ == "__main__":
    main()