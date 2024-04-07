import pandas as pd
import numpy as np

nb_students = 1000

study_hours = np.round(np.abs(np.random.normal(loc=5, scale=3, size=nb_students)),2) 
first_exam_sccores = np.round(np.abs(np.random.normal(loc=5, scale=3, size=nb_students)),2)
attendance = np.round(np.random.choice([0, 1], size=nb_students, p=[0.2, 0.8]),2)

student_score =np.round(2 * study_hours + 50 * first_exam_sccores + 10 * attendance ,2) 
#the values predicted by the model should be around 2,50,10 and 12

data = pd.DataFrame({
    'Study Hour': study_hours,
    'Previous Exam Score': first_exam_sccores,
    'Attendance': attendance,
    'Score': student_score
})

print(data)

data.to_csv('student_score_generated_dataset.csv', index=False)
data.to_csv('student_score_generated_dataset.txt', index=False)

print("\n \n dataset generated ")