*K*-Nearest Neighbors (*k*NN) Classification & Cross-Validation
================
2022-08-08

## Iris

PROBLEM STATEMENT: for the *k*NN classifier, compare the 5-fold,
10-fold, and leave-one-out cross-validation error rates for *k* = 1, …,
50 on the classic `Iris` dataset

### Load the Data

``` r
# import iris data
iris.data <- read.csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data", 
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
      col = factor(iris.data[[5]])) 
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-3-1.png" width="672" />

``` r
# this would be clearer if used `Species` instead of `5` in the command
```

### Fit and Validate the *k*NN Model

I used the `caret` package for model fitting and cross-validation

``` r
# load caret package
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

I tested 5-fold, 10-fold, and leave-one-out cross-validation (CV)

``` r
# 5-fold, 10-fold, and leave-one-out (LOO) CV
numFolds <- c(5, 10, 150)

# list to store the different fits
fitList <- vector(mode = "list", length = length(numFolds))
names(fitList) <- numFolds

for(i in 1:length(numFolds)){
  trControl <- trainControl(method = "cv",
                            number = numFolds[i])

# vary the number of nearest neighbors k from 1 to 50
set.seed(1250)
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
    ##    2  0.9600000  0.94 
    ##    3  0.9600000  0.94 
    ##    4  0.9600000  0.94 
    ##    5  0.9533333  0.93 
    ##    6  0.9400000  0.91 
    ##    7  0.9400000  0.91 
    ##    8  0.9600000  0.94 
    ##    9  0.9600000  0.94 
    ##   10  0.9600000  0.94 
    ##   11  0.9600000  0.94 
    ##   12  0.9733333  0.96 
    ##   13  0.9666667  0.95 
    ##   14  0.9600000  0.94 
    ##   15  0.9733333  0.96 
    ##   16  0.9666667  0.95 
    ##   17  0.9666667  0.95 
    ##   18  0.9600000  0.94 
    ##   19  0.9600000  0.94 
    ##   20  0.9600000  0.94 
    ##   21  0.9666667  0.95 
    ##   22  0.9600000  0.94 
    ##   23  0.9400000  0.91 
    ##   24  0.9600000  0.94 
    ##   25  0.9400000  0.91 
    ##   26  0.9333333  0.90 
    ##   27  0.9333333  0.90 
    ##   28  0.9333333  0.90 
    ##   29  0.9333333  0.90 
    ##   30  0.9333333  0.90 
    ##   31  0.9400000  0.91 
    ##   32  0.9400000  0.91 
    ##   33  0.9400000  0.91 
    ##   34  0.9400000  0.91 
    ##   35  0.9400000  0.91 
    ##   36  0.9400000  0.91 
    ##   37  0.9400000  0.91 
    ##   38  0.9466667  0.92 
    ##   39  0.9400000  0.91 
    ##   40  0.9200000  0.88 
    ##   41  0.9333333  0.90 
    ##   42  0.9400000  0.91 
    ##   43  0.9466667  0.92 
    ##   44  0.9400000  0.91 
    ##   45  0.9266667  0.89 
    ##   46  0.9200000  0.88 
    ##   47  0.9066667  0.86 
    ##   48  0.9200000  0.88 
    ##   49  0.9133333  0.87 
    ##   50  0.9266667  0.89 
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 15.
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
    ##    2  0.9600000  0.94 
    ##    3  0.9600000  0.94 
    ##    4  0.9600000  0.94 
    ##    5  0.9600000  0.94 
    ##    6  0.9533333  0.93 
    ##    7  0.9533333  0.93 
    ##    8  0.9666667  0.95 
    ##    9  0.9733333  0.96 
    ##   10  0.9733333  0.96 
    ##   11  0.9600000  0.94 
    ##   12  0.9600000  0.94 
    ##   13  0.9733333  0.96 
    ##   14  0.9733333  0.96 
    ##   15  0.9666667  0.95 
    ##   16  0.9733333  0.96 
    ##   17  0.9733333  0.96 
    ##   18  0.9733333  0.96 
    ##   19  0.9666667  0.95 
    ##   20  0.9666667  0.95 
    ##   21  0.9733333  0.96 
    ##   22  0.9666667  0.95 
    ##   23  0.9666667  0.95 
    ##   24  0.9666667  0.95 
    ##   25  0.9600000  0.94 
    ##   26  0.9600000  0.94 
    ##   27  0.9533333  0.93 
    ##   28  0.9600000  0.94 
    ##   29  0.9466667  0.92 
    ##   30  0.9400000  0.91 
    ##   31  0.9333333  0.90 
    ##   32  0.9400000  0.91 
    ##   33  0.9400000  0.91 
    ##   34  0.9466667  0.92 
    ##   35  0.9533333  0.93 
    ##   36  0.9533333  0.93 
    ##   37  0.9466667  0.92 
    ##   38  0.9466667  0.92 
    ##   39  0.9466667  0.92 
    ##   40  0.9466667  0.92 
    ##   41  0.9466667  0.92 
    ##   42  0.9533333  0.93 
    ##   43  0.9466667  0.92 
    ##   44  0.9466667  0.92 
    ##   45  0.9466667  0.92 
    ##   46  0.9466667  0.92 
    ##   47  0.9466667  0.92 
    ##   48  0.9466667  0.92 
    ##   49  0.9400000  0.91 
    ##   50  0.9400000  0.91 
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 21.
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
    ##    2  0.9600000  0    
    ##    3  0.9600000  0    
    ##    4  0.9600000  0    
    ##    5  0.9666667  0    
    ##    6  0.9666667  0    
    ##    7  0.9666667  0    
    ##    8  0.9666667  0    
    ##    9  0.9666667  0    
    ##   10  0.9733333  0    
    ##   11  0.9800000  0    
    ##   12  0.9600000  0    
    ##   13  0.9666667  0    
    ##   14  0.9800000  0    
    ##   15  0.9733333  0    
    ##   16  0.9733333  0    
    ##   17  0.9733333  0    
    ##   18  0.9800000  0    
    ##   19  0.9800000  0    
    ##   20  0.9800000  0    
    ##   21  0.9666667  0    
    ##   22  0.9600000  0    
    ##   23  0.9666667  0    
    ##   24  0.9666667  0    
    ##   25  0.9666667  0    
    ##   26  0.9533333  0    
    ##   27  0.9533333  0    
    ##   28  0.9666667  0    
    ##   29  0.9533333  0    
    ##   30  0.9533333  0    
    ##   31  0.9466667  0    
    ##   32  0.9533333  0    
    ##   33  0.9533333  0    
    ##   34  0.9466667  0    
    ##   35  0.9533333  0    
    ##   36  0.9466667  0    
    ##   37  0.9466667  0    
    ##   38  0.9400000  0    
    ##   39  0.9466667  0    
    ##   40  0.9400000  0    
    ##   41  0.9466667  0    
    ##   42  0.9466667  0    
    ##   43  0.9400000  0    
    ##   44  0.9466667  0    
    ##   45  0.9400000  0    
    ##   46  0.9466667  0    
    ##   47  0.9466667  0    
    ##   48  0.9466667  0    
    ##   49  0.9266667  0    
    ##   50  0.9200000  0    
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 20.

### Analyze Performance Metrics

Plot the results showing the comparison in performance

``` r
plot(1:50, 1-fitList[[1]]$results$Accuracy,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", 
     ylab = "Cross-Validation Error Rate", 
     ylim = c(0.01, 0.091))
points(1:50, 1-fitList[[2]]$results$Accuracy, type = "o", col = "blue")
points(1:50, 1-fitList[[3]]$results$Accuracy, type = "o", col = "black")
legend("topleft", legend = c("5-fold CV", "10-fold CV", "LOOCV"),
       col = c("red", "blue", "black"), lty = 1)
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-6-1.png" width="672" />

## USPS digits

PROBLEM STATEMENT: for the *k*NN classifier, compare the test error
rates for three different distance metrics (Euclidean, Manhattan,
cosine) on the [USPS handwritten zip code digits
dataset](https://hastie.su.domains/ElemStatLearn/data.html)

Load the data

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✔ tibble  3.1.7     ✔ dplyr   1.0.9
    ## ✔ tidyr   1.2.0     ✔ stringr 1.4.0
    ## ✔ readr   2.1.2     ✔ forcats 0.5.1
    ## ✔ purrr   0.3.4

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ✖ purrr::lift()   masks caret::lift()

``` r
usps_train <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz", 
  delim = " ", col_names = FALSE)
```

    ## Rows: 7291 Columns: 258

    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: " "
    ## dbl (257): X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,...
    ## lgl   (1): X258
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

Look at the data.

[Each row
is](https://hastie.su.domains/ElemStatLearn/datasets/zip.info.txt) the
256 normalized grayscale values of the 16 x 16 pixel image of a digit.

The first column contains the identity (label) of the digit. The last
column is all NA.

``` r
# str(usps_train)

(usps_train_labels <- usps_train[, 1])
```

    ## # A tibble: 7,291 × 1
    ##       X1
    ##    <dbl>
    ##  1     6
    ##  2     5
    ##  3     4
    ##  4     7
    ##  5     3
    ##  6     6
    ##  7     3
    ##  8     1
    ##  9     0
    ## 10     1
    ## # … with 7,281 more rows

``` r
summary(usps_train[, 258])
```

    ##    X258        
    ##  Mode:logical  
    ##  NA's:7291

``` r
head(usps_train)
```

    ## # A tibble: 6 × 258
    ##      X1    X2    X3    X4     X5     X6     X7     X8     X9    X10    X11
    ##   <dbl> <dbl> <dbl> <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ## 1     6    -1    -1    -1 -1     -1     -1     -1     -0.631  0.862 -0.167
    ## 2     5    -1    -1    -1 -0.813 -0.671 -0.809 -0.887 -0.671 -0.853 -1    
    ## 3     4    -1    -1    -1 -1     -1     -1     -1     -1     -1     -0.996
    ## 4     7    -1    -1    -1 -1     -1     -0.273  0.684  0.96   0.45  -0.067
    ## 5     3    -1    -1    -1 -1     -1     -0.928 -0.204  0.751  0.466  0.234
    ## 6     6    -1    -1    -1 -1     -1     -0.397  0.983 -0.535 -1     -1    
    ## # … with 247 more variables: X12 <dbl>, X13 <dbl>, X14 <dbl>, X15 <dbl>,
    ## #   X16 <dbl>, X17 <dbl>, X18 <dbl>, X19 <dbl>, X20 <dbl>, X21 <dbl>,
    ## #   X22 <dbl>, X23 <dbl>, X24 <dbl>, X25 <dbl>, X26 <dbl>, X27 <dbl>,
    ## #   X28 <dbl>, X29 <dbl>, X30 <dbl>, X31 <dbl>, X32 <dbl>, X33 <dbl>,
    ## #   X34 <dbl>, X35 <dbl>, X36 <dbl>, X37 <dbl>, X38 <dbl>, X39 <dbl>,
    ## #   X40 <dbl>, X41 <dbl>, X42 <dbl>, X43 <dbl>, X44 <dbl>, X45 <dbl>,
    ## #   X46 <dbl>, X47 <dbl>, X48 <dbl>, X49 <dbl>, X50 <dbl>, X51 <dbl>, …

``` r
ls()
```

    ## [1] "fitList"           "i"                 "iris.data"        
    ## [4] "numFolds"          "trControl"         "usps_train"       
    ## [7] "usps_train_labels"

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
