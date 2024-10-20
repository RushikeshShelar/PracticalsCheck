import random
# # Input data
# x1 = list(map(float, input("Enter Income : ").split(" ")))
# x2 = list(map(float, input("Enter Saving: ").split(" ")))
# y = list(map(int, input("Loan Sanctioned (0/1): ").split(" ")))

# Initialize weights
b0, b1, b2 = 0, 0, 0
alpha = 0.3  # Learning rate
e = 2.7182818284  # Base of natural log (e)

# Sigmoid function with clipping to prevent overflow
def sigmoid(z):
    # Clip z to prevent overflow in exp function
    z = max(min(z, 500), -500)
    return 1 / (1 + e ** -z)

# Function to generate data for logistic regression
def generate_data(n):
    income = []
    savings = []
    loan_granted = []
    
    for _ in range(n):
        inc = random.randint(10000, 100000)  # Random income between 10,000 and 100,000
        sav = random.randint(1000, 50000)    # Random savings between 1,000 and 50,000
        
        # Apply the rule: If income > 2 * savings, loan is granted (1), otherwise rejected (0)
        if inc < 2 * sav:
            loan_granted.append(1)
        else:
            loan_granted.append(0)
        
        income.append(inc)
        savings.append(sav)
    
    # Store the data into x1, x2, and y respectively
    x1 = income
    x2 = savings
    y = loan_granted
    
    return x1, x2, y

x1, x2, y = generate_data(100)

# Gradient Descent Iterations
for epoch in range(1000):  # Run for 1000 iterations (you can increase it)
    for i in range(len(x1)):
        # Prediction using sigmoid function
        z = b0 + (b1 * x1[i]) + (b2 * x2[i])
        prediction = sigmoid(z)
        
        # Error = y[i] - prediction
        error = y[i] - prediction
        
        # Update the weights based on gradient descent
        b0 = b0 + alpha * error * prediction * (1 - prediction)
        b1 = b1 + alpha * error * prediction * (1 - prediction) * x1[i]
        b2 = b2 + alpha * error * prediction * (1 - prediction) * x2[i]

# Final weights after training
print("Final Weights: B0:", b0, "B1:", b1, "B2:", b2)

# Test input
test_X1 = float(input("Enter Income for testing: "))
test_X2 = float(input("Enter Savings for testing: "))

# Prediction for test input
test_z = b0 + (b1 * test_X1) + (b2 * test_X2)
test_prediction = sigmoid(test_z)

# Output the result
if test_prediction > 0.5:
    print("Loan Sanctioned: GRANT")
else:
    print("Loan Sanctioned: REJECT")
