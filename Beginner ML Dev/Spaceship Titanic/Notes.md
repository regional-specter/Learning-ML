1. Import the libraries
2. Calculate the percentage of male and female survivors
3. Build the model

Calculating and predicting survival outcomes based off of only a single column such as gender can be very misleading and produce inaccurate results. By using more than one field you can produce a trustable conclusion and results

We use the random forest model and a selection of different columns to produce the model where it looks for patterns in four different columns ("Pclass", "Sex", "SibSp", and "Parch") of the data. 

It constructs the trees in the random forest model based on patterns in the `train.csv` file, before generating predictions for the passengers in `test.csv`