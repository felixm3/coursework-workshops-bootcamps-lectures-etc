kNN Classification & Crossvalidation
================
2022-08-08

## Iris

### Load the Data

``` r
# import iris data
iris.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data", 
                      header = FALSE, 
                      col.names = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"))
```

### Look at the Data

``` r
head(iris.data)
```

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width     Species
    ## 1          5.1         3.5          1.4         0.2 Iris-setosa
    ## 2          4.9         3.0          1.4         0.2 Iris-setosa
    ## 3          4.7         3.2          1.3         0.2 Iris-setosa
    ## 4          4.6         3.1          1.5         0.2 Iris-setosa
    ## 5          5.0         3.6          1.4         0.2 Iris-setosa
    ## 6          5.4         3.9          1.7         0.4 Iris-setosa

``` r
iris.data[5] <- factor(iris.data[[5]])

str(iris.data)
```

    ## 'data.frame':    150 obs. of  5 variables:
    ##  $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
    ##  $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
    ##  $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
    ##  $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
    ##  $ Species     : Factor w/ 3 levels "Iris-setosa",..: 1 1 1 1 1 1 1 1 1 1 ...

``` r
summary(iris.data)
```

    ##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
    ##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
    ##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
    ##  Median :5.800   Median :3.000   Median :4.350   Median :1.300  
    ##  Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
    ##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
    ##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
    ##             Species  
    ##  Iris-setosa    :50  
    ##  Iris-versicolor:50  
    ##  Iris-virginica :50  
    ##                      
    ##                      
    ## 

``` r
pairs(~ ., data = iris.data[-5], 
      col = factor(iris.data[[5]])) # this would be clearer if used `Species` instead of `5` in the command
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-3-1.png" width="672" />

I used the `caret` package for model fitting and crossvalidation

``` r
# load caret package
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

I tested 5-fold, 10-fold, and leave-one-out crossvalidation (CV)

``` r
# 5-fold, 10-fold, and leave-one-out (LOO) CV
numFolds <- c(5, 10, 150)

# list to store the different fits
fitList <- vector(mode = "list", length = length(numFolds))
names(fitList) <- numFolds

for(i in 1:length(numFolds)){
  trControl <- trainControl(method = "cv",
                            number = numFolds[i])
  
  fitList[[i]] <- train(Species ~ .,
                        method = "knn",
                        tuneGrid = expand.grid(k = 1:50),
                        trControl = trControl,
                        metric = "Accuracy",
                        data = iris.data)
}
```

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo, :
    ## There were missing values in resampled performance measures.

``` r
fitList
```

    ## $`5`
    ## k-Nearest Neighbors 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 120, 120, 120, 120, 120 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa
    ##    1  0.9600000  0.94 
    ##    2  0.9533333  0.93 
    ##    3  0.9733333  0.96 
    ##    4  0.9800000  0.97 
    ##    5  0.9800000  0.97 
    ##    6  0.9733333  0.96 
    ##    7  0.9666667  0.95 
    ##    8  0.9533333  0.93 
    ##    9  0.9733333  0.96 
    ##   10  0.9600000  0.94 
    ##   11  0.9600000  0.94 
    ##   12  0.9666667  0.95 
    ##   13  0.9733333  0.96 
    ##   14  0.9733333  0.96 
    ##   15  0.9666667  0.95 
    ##   16  0.9666667  0.95 
    ##   17  0.9666667  0.95 
    ##   18  0.9600000  0.94 
    ##   19  0.9666667  0.95 
    ##   20  0.9600000  0.94 
    ##   21  0.9533333  0.93 
    ##   22  0.9533333  0.93 
    ##   23  0.9600000  0.94 
    ##   24  0.9533333  0.93 
    ##   25  0.9533333  0.93 
    ##   26  0.9533333  0.93 
    ##   27  0.9533333  0.93 
    ##   28  0.9400000  0.91 
    ##   29  0.9400000  0.91 
    ##   30  0.9533333  0.93 
    ##   31  0.9400000  0.91 
    ##   32  0.9466667  0.92 
    ##   33  0.9400000  0.91 
    ##   34  0.9400000  0.91 
    ##   35  0.9400000  0.91 
    ##   36  0.9466667  0.92 
    ##   37  0.9466667  0.92 
    ##   38  0.9466667  0.92 
    ##   39  0.9466667  0.92 
    ##   40  0.9466667  0.92 
    ##   41  0.9400000  0.91 
    ##   42  0.9466667  0.92 
    ##   43  0.9333333  0.90 
    ##   44  0.9466667  0.92 
    ##   45  0.9400000  0.91 
    ##   46  0.9333333  0.90 
    ##   47  0.9333333  0.90 
    ##   48  0.9333333  0.90 
    ##   49  0.9333333  0.90 
    ##   50  0.9200000  0.88 
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 5.
    ## 
    ## $`10`
    ## k-Nearest Neighbors 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 135, 135, 135, 135, 135, 135, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa
    ##    1  0.9600000  0.94 
    ##    2  0.9666667  0.95 
    ##    3  0.9600000  0.94 
    ##    4  0.9666667  0.95 
    ##    5  0.9600000  0.94 
    ##    6  0.9733333  0.96 
    ##    7  0.9800000  0.97 
    ##    8  0.9733333  0.96 
    ##    9  0.9800000  0.97 
    ##   10  0.9666667  0.95 
    ##   11  0.9733333  0.96 
    ##   12  0.9733333  0.96 
    ##   13  0.9666667  0.95 
    ##   14  0.9800000  0.97 
    ##   15  0.9733333  0.96 
    ##   16  0.9733333  0.96 
    ##   17  0.9733333  0.96 
    ##   18  0.9733333  0.96 
    ##   19  0.9733333  0.96 
    ##   20  0.9800000  0.97 
    ##   21  0.9733333  0.96 
    ##   22  0.9666667  0.95 
    ##   23  0.9666667  0.95 
    ##   24  0.9800000  0.97 
    ##   25  0.9666667  0.95 
    ##   26  0.9600000  0.94 
    ##   27  0.9533333  0.93 
    ##   28  0.9466667  0.92 
    ##   29  0.9466667  0.92 
    ##   30  0.9466667  0.92 
    ##   31  0.9466667  0.92 
    ##   32  0.9533333  0.93 
    ##   33  0.9466667  0.92 
    ##   34  0.9333333  0.90 
    ##   35  0.9400000  0.91 
    ##   36  0.9400000  0.91 
    ##   37  0.9400000  0.91 
    ##   38  0.9466667  0.92 
    ##   39  0.9466667  0.92 
    ##   40  0.9466667  0.92 
    ##   41  0.9533333  0.93 
    ##   42  0.9533333  0.93 
    ##   43  0.9533333  0.93 
    ##   44  0.9400000  0.91 
    ##   45  0.9400000  0.91 
    ##   46  0.9400000  0.91 
    ##   47  0.9400000  0.91 
    ##   48  0.9466667  0.92 
    ##   49  0.9466667  0.92 
    ##   50  0.9266667  0.89 
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 24.
    ## 
    ## $`150`
    ## k-Nearest Neighbors 
    ## 
    ## 150 samples
    ##   4 predictor
    ##   3 classes: 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (150 fold) 
    ## Summary of sample sizes: 149, 149, 149, 149, 149, 149, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa
    ##    1  0.9600000  0    
    ##    2  0.9533333  0    
    ##    3  0.9600000  0    
    ##    4  0.9600000  0    
    ##    5  0.9666667  0    
    ##    6  0.9666667  0    
    ##    7  0.9733333  0    
    ##    8  0.9733333  0    
    ##    9  0.9666667  0    
    ##   10  0.9600000  0    
    ##   11  0.9733333  0    
    ##   12  0.9533333  0    
    ##   13  0.9666667  0    
    ##   14  0.9800000  0    
    ##   15  0.9733333  0    
    ##   16  0.9733333  0    
    ##   17  0.9733333  0    
    ##   18  0.9733333  0    
    ##   19  0.9800000  0    
    ##   20  0.9600000  0    
    ##   21  0.9733333  0    
    ##   22  0.9600000  0    
    ##   23  0.9666667  0    
    ##   24  0.9600000  0    
    ##   25  0.9666667  0    
    ##   26  0.9666667  0    
    ##   27  0.9666667  0    
    ##   28  0.9533333  0    
    ##   29  0.9533333  0    
    ##   30  0.9466667  0    
    ##   31  0.9466667  0    
    ##   32  0.9466667  0    
    ##   33  0.9533333  0    
    ##   34  0.9533333  0    
    ##   35  0.9533333  0    
    ##   36  0.9533333  0    
    ##   37  0.9466667  0    
    ##   38  0.9333333  0    
    ##   39  0.9466667  0    
    ##   40  0.9466667  0    
    ##   41  0.9400000  0    
    ##   42  0.9466667  0    
    ##   43  0.9400000  0    
    ##   44  0.9466667  0    
    ##   45  0.9400000  0    
    ##   46  0.9400000  0    
    ##   47  0.9533333  0    
    ##   48  0.9533333  0    
    ##   49  0.9400000  0    
    ##   50  0.9466667  0    
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 19.

Plot the results showing the comparison in performance

``` r
plot(1:50, 1-fitList[[1]]$results$Accuracy,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", ylab = "Misclassification Error Rate")
points(1:50, 1-fitList[[2]]$results$Accuracy, type = "o", col = "blue")
points(1:50, 1-fitList[[3]]$results$Accuracy, type = "o", col = "black")
legend("topleft", legend = c("5-fold CV", "10-fold CV", "LOOCV"),
       col = c("red", "blue", "black"), lty = 1)
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-6-1.png" width="672" />

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/pressure-1.png" width="672" />

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
