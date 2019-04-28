
Input Files = Input Data

ADF_test.py = Script for augmented Dickey–Fuller test (ADF),
              used to find the best window length for
              elaborating data.

              Input -> .edf file with data to analyze

              Output -> Prints out the result of the ADF test
           
edf.py = Script for elaborating input data.

         Input -> Input data from InputFiles folder
                  1. "InputFiles/chb05_12.edf"
                  2. "InputFiles/chb05_13s.edf"
                  3. "InputFiles/chb05_14.edf"
                  
         Output -> Elaborated data with added features
                   1. 'OutputFiles/Measure_12_df.csv'
                   2. 'OutputFiles/Measure_13_df.csv'
                   3. 'OutputFiles/Measure_14_df.csv'
           
kmeans.py = Script for applying kmeans algorithm to data.

            Input -> Elaborated data with added features
                     1. 'OutputFiles/Measure_12_df.csv'
                     2. 'OutputFiles/Measure_13_df.csv'
                     3. 'OutputFiles/Measure_14_df.csv'
                     
            Output -> Input data with added column for cluster
                      'kmeans3.csv'
                      
kmeansPlot.py = Script for plotting clusterization results.

                Input -> Elaborated data with cluster column 
                         'kmeans3.csv'
                         
                Output -> Plots three figures (one per file),
                          each showing all the electrods signals
                          colored by cluster number.
