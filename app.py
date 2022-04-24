import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

#creating function for prediction
def diabetes_prediction(input_data):

    #changing the input_data to array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not Diabetic'
    else:
        return 'The person is Diabetic'

def main():

    #title for our web-page
    st.title('Diabetes Prediction Web App')

    #getting input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')    

    #code for prediction 
    diagnosis = ''

    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()