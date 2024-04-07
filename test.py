from linear_model import *
import matplotlib.pyplot as plt
import pandas as pd 
import time 

dataset = pd.read_csv('student_score_generated_dataset.csv') #import the dataset
data_matrix = dataset[['Study Hour', 'Previous Exam Score', 'Attendance', 'Score']].values #convert the date to a matrix 
print(f"{data_matrix.shape[0]} students found")

training_data =data_matrix[:800, :]
testing_data =data_matrix[800:,:]
#print(training_data)
#print(testing_data)
t1=time.time() 
w0,w1,w2,b,cost_func_graph=gradient_descent(training_data,0.001,10000)
print("#####################################")
print(f"w0={w0}\nw0={w1}\nw0={w2}\nb={b}")
print("#####################################")
t2=time.time() 
print(f"gradint descent execution time ={np.round(t2-t1,2)} sec") 
print("#####################################")

#testing

iteration = cost_func_graph.keys()
cost = cost_func_graph.values()
plt.plot(iteration, cost, marker='.', linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function Value')
plt.title('Gradient Descent Convergence')
plt.grid(True)
plt.show()
#########################################################
predicted_values={}
error_value={}
for i in range(testing_data.shape[0]) :
    predicted=model(w0,w1,w2,b,testing_data[i,0],testing_data[i,1],testing_data[i,2])
    predicted_values[f"predicted value for student num {i+1}"]=predicted
    error_value[f"error value for student num{i+1}"]=abs(predicted-testing_data[i,3]) 
    print(f"error value for student num {i+1}=  {abs(predicted-testing_data[i,3])}")   
print("#####################################")
###################################################
predicted=predicted_values.values()
real_values=testing_data[:,3]
student_num=range(1,201)
plt.scatter(student_num, predicted, color='red', label='Predicted')
plt.scatter(student_num, real_values, color='blue', label='Real')
plt.title('Predicted vs Real Values for 200 Students')
plt.xlabel('Student Index')
plt.ylabel('Value')
plt.annotate('zoom in to spot the difference ', xy=(3, 5))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()