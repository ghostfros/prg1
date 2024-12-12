import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data for years worked and cakes made
years_worked = np.array([1, 2, 3, 4, 5, 6])
cakes_made = np.array([6500, 7805, 10835, 11230, 15870, 16387])

# Create a scatter plot
plt.scatter(years_worked, cakes_made, color='blue', label='Cakes made')

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(years_worked, cakes_made)

# Line of best fit
line_of_best_fit = slope * years_worked + intercept

# Plot the line of best fit
plt.plot(years_worked, line_of_best_fit, color='red', label=f'Line of best fit')

# Add labels and title
plt.xlabel('Years worked')
plt.ylabel('Cakes made')
plt.title('Crabby Cakes made by Baker')
plt.legend()

# Show the plot
plt.show()

# Correlation coefficient (r)
correlation_coefficient = r_value

# Predict the number of cakes made after 10 years using the linear regression equation
predicted_cakes_after_10_years = slope * 10 + intercept

correlation_coefficient, predicted_cakes_after_10_years