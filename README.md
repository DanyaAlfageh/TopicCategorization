# Topic Categorization of Documents

Topic Categorization is an implemention of logistic regression and naive bayes to determine the category in which a document would classify as.
## Running the program:
To run the project from the project root:
```
$ “python3 Main.py [ALGORITHM]”
```
Algorithm determines which of the two algorithms to run over the data:

`regression` (Linear Regression)  
`naive` (Naive Bayes)
`mi` (Mutual Information)
`beta` (A demonstration of Naive Bayes accuracy when using different beta values)

## Input File structure:

Provide two files to the program, which contain raw testing and training data:
```
./data/sparse/training.csv (your training set goes here)
./data/sparse/testing.csv (your testing file goes here)
```
or, if your data is dense:
```
./data/dense/training.csv (your training set goes here) 
./data/dense/testing.csv (your testing file goes here)
```
## Output File structure:
The program will output the following file:
```
./data/prediction.csv (the generated ML prediction)
```
With the internal structure:
```
Inside the ‘prediction.csv’:
ID | classification
```
