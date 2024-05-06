README.txt

This Repository is meant to hold files pertaining to the ECE 4424 - Machine Learning Final Project.

Our project involves:
- Continuously collecting data from the McComas Gym live occupancy website
- Storing the data along with date, time, and weather into csv files
- Creating a model/using an existing model to predict the occupancy of McComas Gym given weather, time, and a date/day of the week
- Uses binary classification to predict 'busy' and 'not busy' given the inputs

To run our project, just type 'python learner.py' in the terminal while in the correct directory. The code stops running when 
graphs are plotted out and continues after the user exits each plot. The last bit of the project allows the user to ask our 
best performing model a day of the week, time of day, and temperature in Farenheit, and the model will predict whether or not
the McComas Hall Gym will be busy.
