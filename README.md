# Topic Categorization of Documents

TODO: Pretty Description
## Running the program:
To run the project from the project root:
```
$ “python3 Main.py [ALGORITHM]”
```
Algorithm determines which of the two algorithms to run over the data:

`regression` (Linear Regression)  
`naive` (Naive Bayes)

## Input File structure: 

Provide two files to the program, which contain testing and training data:
```
./data/sparce/training.csv (your training set goes here) 
./data/sparce/testing.csv (your testing file goes here)
```
Inside the `training.csv` should be the following structure as a `.csv`:
```
TODO: Write this out
```
Inside the `testing.csv` should be the following structure as a `.csv`:
```
TODO: Write this out
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
