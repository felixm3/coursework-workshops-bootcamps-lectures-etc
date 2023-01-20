*K*-Nearest Neighbors (*k*NN) Classification & Cross-Validation
================
2022-08-08

## Intro to *k*NN Classification & Cross-Validation: Iris

**PROBLEM STATEMENT**: for the *k*NN classifier, compare the 5-fold
cross-validation, 10-fold cross-validation, and leave-one-out
cross-validation (LOOCV) error rates for *k* = 1, …, 50 on the classic
`Iris` dataset

### Load the Data

Let’s download the data from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/index.php)

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
# encode 5th column (Species) as categorical variable
iris.data[5] <- factor(iris.data[[5]])

str(iris.data)
```

    ## 'data.frame':    150 obs. of  5 variables:
    ##  $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
    ##  $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
    ##  $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
    ##  $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
    ##  $ Species     : Factor w/ 3 levels "Iris-setosa",..: 1 1 1 1 1 1 1 1 1 1 ...

-   the dataset has 150 observations (rows) and 5 variables (columns)
-   four of the columns are numeric (continuous) variables
-   the `Species` column is categorical with 3 levels

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

-   the observations are equally divided among the three `Species`
-   the variables are in different ranges and should therefore
    (probably) be scaled before *k*NN

``` r
# confirm no missing values (NA)
sum(is.na(iris.data))
```

    ## [1] 0

-   the dataset has no missing values (NAs) - confirmed by
    `sum(is.na(iris.data))` returning 0

Let’s plot the data

``` r
# pairs plot of the data coloring by Species column
pairs(~ ., data = iris.data[-5], 
      col = factor(iris.data[[5]]), 
      oma = c(3, 3, 3, 14)) 

par(xpd = TRUE)
legend("bottomright", 
       fill = unique(iris$Species), 
       legend = c(levels(iris$Species)))
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-5-1.png" width="672" />

### Fit and Validate the *k*NN Model

Let’s use the `caret` package for model-fitting and for
cross-validation, testing 5-fold cross-validation, 10-fold
cross-validation, and leave-one-out cross-validation (LOOCV)

``` r
# load caret package
library(caret)

# 5-fold, 10-fold, and leave-one-out (LOO) CV
numFolds <- c(5, 10, 150)

# create list to store the different fits
fitList <- vector(mode = "list", length = length(numFolds))
names(fitList) <- numFolds

# loop through the three different CV types
for(i in 1:length(numFolds)){
  trControl <- trainControl(method = "cv",
                            number = numFolds[i])

  # vary the number of nearest neighbors k from 1 to 50
  set.seed(1250)
  fitList[[i]] <- train(Species ~ .,
                          method = "knn",
                          tuneGrid = expand.grid(k = 1:50), 
                          preProcess = c("center", "scale"), 
                          trControl = trControl,
                          metric = "Accuracy",
                          data = iris.data)
}
```

### Analyze Performance Metrics

Plot the results showing the comparison in performance

``` r
plot(1:50, 1-fitList[[1]]$results$Accuracy,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", 
     ylab = "Cross-Validation Error Rate", 
     main = "kNN Classification of Iris Dataset", 
     ylim = c(0, 0.14))

points(1:50, 1-fitList[[2]]$results$Accuracy, 
       type = "o", col = "blue")

points(1:50, 1-fitList[[3]]$results$Accuracy, 
       type = "o", col = "black")

legend("topleft", legend = c("5-fold CV", "10-fold CV", "LOOCV"),
       col = c("red", "blue", "black"), lty = 1)
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-7-1.png" width="672" />

-   CV error rate at first decreases with increasing number of nearest
    neighbors used
-   it then levels out before increasing, forming the familiar
    *bias-variance tradeoff* U-curve
-   Lowest CV error rate seen with \~7-18 nearest neighbors.
-   CV error rate generally increases: 5-fold CV \> 10-fold CV \> LOOCV

------------------------------------------------------------------------

## *k*NN classifer with different distance metrics: USPS digits

**PROBLEM STATEMENT**: for the *k*NN classifier, compare the test error
rates for three different distance metrics (Euclidean, Manhattan,
cosine) on the [USPS handwritten zip code digits
dataset](https://hastie.su.domains/ElemStatLearn/data.html)

### Load the data

``` r
library(tidyverse)
usps_train <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)
usps_test <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)
```

### Look at the data

[Each row
is](https://hastie.su.domains/ElemStatLearn/datasets/zip.info.txt) the
256 normalized grayscale values of the 16 x 16 pixel image of a
handwritten digit.

The first column contains the identity (label) of the digit. The last
column of `usps_train` is all NA.

``` r
usps_train <- as.matrix(usps_train)
usps_test <- as.matrix(usps_test)

# first column contains labels
usps_train_labels <- usps_train[, 1]
usps_test_labels <- usps_test[, 1]

# drop first column of usps_train and usps_test
usps_train <- usps_train[, -1]
usps_test <- usps_test[, -1]

# drop last column (all NA) of usps_train
usps_train <- usps_train[, -257]

str(usps_train)
```

    ##  num [1:7291, 1:256] -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
    ##  - attr(*, "dimnames")=List of 2
    ##   ..$ : NULL
    ##   ..$ : chr [1:256] "X2" "X3" "X4" "X5" ...

``` r
str(usps_test)
```

    ##  num [1:2007, 1:256] -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
    ##  - attr(*, "dimnames")=List of 2
    ##   ..$ : NULL
    ##   ..$ : chr [1:256] "X2" "X3" "X4" "X5" ...

-   the training set and the test set have 7,291 and 2,007 observations
    respectively

Encode labels as categorical variables (factors)

``` r
usps_train_labels <- factor(usps_train_labels)
usps_test_labels <- factor(usps_test_labels)

str(usps_train_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 7 6 5 8 4 7 4 2 1 2 ...

``` r
str(usps_test_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 10 7 4 7 7 1 1 1 7 10 ...

### Calculate Distance Matrices for the Three Distances (Euclidean, Manhattan, cosine)

**Define functions that calculate distance (Euclidean, Manhattan,
cosine) between two vectors.**

These will be used to calculate the `n x m` distances between each of
the `n` test points and all the `m` train points.

``` r
euclidean_dist <- function(x1, x2){
  # calculates Euclidean distance between vectors x1 and x2
  return(sqrt(sum((x1 - x2) ^ 2)))
}
manhattan_dist <- function(x1, x2){
  # calculates Manhattan distance between vectors x1 and x2
  return(sum(abs(x1 - x2)))
}
cosine_dist <- function(x1, x2){
  # calculates Cosine 'distance' between vectors x1 and x2
  return(1 - ((x1 %*% x2)/(sqrt(x1 %*% x1) * sqrt(x2 %*% x2))))
}
```

**Calculate distance matrices**

For the three ‘distances’ (Euclidean, Manhattan, cosine), we create a
matrix to store distances between each test point and all the train
points. Each row of the distance matrix is a test point, each column is
a train point, and the element `(i, j)` is the distance between test
point `i` and train point `j`. This matrix will be used to figure out
each test point’s nearest neighbors.

``` r
distances_Euclidean <- matrix(0, nrow = nrow(usps_test), ncol = nrow(usps_train))
distances_Manhattan <- matrix(0, nrow = nrow(usps_test), ncol = nrow(usps_train))
distances_Cosine <- matrix(0, nrow = nrow(usps_test), ncol = nrow(usps_train))
```

**Populate the distance matrices**

*k*NN usually requires scaling before calculating distances to ensure
all features are on the same footing. But since this data has already
been scaled (all values are between -1 and 1), this step was skipped.

``` r
for(i in 1:nrow(usps_test)){
  for(j in 1:nrow(usps_train)){
    distances_Euclidean[i, j] <- euclidean_dist(usps_test[i, ], usps_train[j, ])
    distances_Manhattan[i, j] <- manhattan_dist(usps_test[i, ], usps_train[j, ])
    distances_Cosine[i, j] <- cosine_dist(usps_test[i, ], usps_train[j, ])
  }
} # ~563s
```

### Implement *k*NN Classification for the Three Distances

``` r
# create matrix with k cols & nrow(usps_test) rows 
# to store preds for every test point using 1 to k neighbors
usps_test_pred_Euclidean <- matrix(0, nrow = nrow(usps_test), ncol = 10)

# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train set, train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train, usps_train_labels, distances_Euclidean[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Euclidean[i, ]), ] # sort by distance
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store counts of ten classes
  z_labels <- levels(usps_train_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:10){ # count how many times each class appears in the k neighbors
      z[m] <- sum(trSetLabDist[1:k, "usps_train_labels"] == z_labels[m])
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Euclidean[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
} # ~75s
```

Now do for Manhattan and cosine distance

``` r
# create matrix with k cols & nrow(usps_test) rows 
# to store preds for every test point using 1 to k neighbors
usps_test_pred_Manhattan <- matrix(0, nrow = nrow(usps_test), ncol = 10)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train set, train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train, usps_train_labels, distances_Manhattan[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Manhattan[i, ]), ] # sort by distance
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store counts of ten classes
  z_labels <- levels(usps_train_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:10){ # count how many times each class appears in the k neighbors
      z[m] <- sum(trSetLabDist[1:k, "usps_train_labels"] == z_labels[m])
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Manhattan[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
}


# create matrix with k cols & nrow(usps_test) rows 
# to store preds for every test point using 1 to k neighbors
usps_test_pred_Cosine <- matrix(0, nrow = nrow(usps_test), ncol = 10)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train set, train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train, usps_train_labels, distances_Cosine[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Cosine[i, ]), ] # sort by distance
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store counts of ten classes
  z_labels <- levels(usps_train_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:10){ # count how many times each class appears in the k neighbors
      z[m] <- sum(trSetLabDist[1:k, "usps_train_labels"] == z_labels[m])
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Cosine[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
}
```

Calculate error rates for the three distances and plot them

``` r
library(caret) # for function confusionMatrix

errorRate_Euclidean <- numeric(10)
for(i in 1:10){
  errorRate_Euclidean[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Euclidean[, i]), 
                                                usps_test_labels)$overall["Accuracy"]
}

errorRate_Manhattan <- numeric(10)
for(i in 1:10){
  errorRate_Manhattan[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Manhattan[, i]), 
                                                usps_test_labels)$overall["Accuracy"]
}

errorRate_Cosine <- numeric(10)
for(i in 1:10){
  errorRate_Cosine[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Cosine[, i]), 
                                             usps_test_labels)$overall["Accuracy"]
}

# Plot the error rates

plot(1:10, errorRate_Euclidean,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", ylab = "Misclassification Error Rate",
     ylim = c(0.05, 0.08))
points(1:10, errorRate_Manhattan, 
       type = "o", col = "blue")
points(1:10, errorRate_Cosine, 
       type = "o", col = "black")
legend("top", legend = c("Euclidean", "Manhattan", "Cosine"),
       col = c("red", "blue", "black"), lty = 1)
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-14-1.png" width="672" />

``` r
cbind(errorRate_Cosine, errorRate_Euclidean, errorRate_Manhattan)
```

    ##       errorRate_Cosine errorRate_Euclidean errorRate_Manhattan
    ##  [1,]       0.05729945          0.05630294          0.06228201
    ##  [2,]       0.06028899          0.06826109          0.06776283
    ##  [3,]       0.05580468          0.05630294          0.05929248
    ##  [4,]       0.05580468          0.05480817          0.06128550
    ##  [5,]       0.05630294          0.05729945          0.06228201
    ##  [6,]       0.05879422          0.06078724          0.06377678
    ##  [7,]       0.06128550          0.05879422          0.06576981
    ##  [8,]       0.06028899          0.05829596          0.07125062
    ##  [9,]       0.06327853          0.06128550          0.07224714
    ## [10,]       0.06477329          0.06228201          0.07673144

-   using Manhattan distance resulted in highest error rates
-   Euclidean and cosine distances were similar to each other in
    performance

------------------------------------------------------------------------

## weighted *k*NN: USPS digits

**PROBLEM STATEMENT**: apply the weighted *k*NN classifier (Euclidean
distance, weights: inverse, squared inverse, and linear) on the [USPS
handwritten zip code digits
dataset](https://hastie.su.domains/ElemStatLearn/data.html) and compare
test errors.

<details>
<summary>
Load the data as above
</summary>

``` r
library(tidyverse)
usps_train <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)
usps_test <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)

# str(usps_train)

usps_train <- as.matrix(usps_train)
usps_test <- as.matrix(usps_test)

# first column contains labels
usps_train_labels <- usps_train[, 1]
usps_test_labels <- usps_test[, 1]

# drop first column of usps_train and usps_test
usps_train <- usps_train[, -1]
usps_test <- usps_test[, -1]

# drop last column (all NA) of usps_train
usps_train <- usps_train[, -257]

# convert labels to factors
usps_train_labels <- factor(usps_train_labels)
usps_test_labels <- factor(usps_test_labels)

str(usps_train_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 7 6 5 8 4 7 4 2 1 2 ...

``` r
str(usps_test_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 10 7 4 7 7 1 1 1 7 10 ...

</details>
<details>
<summary>
Is it faster to apply a function to a matrix directly `foo(M)` or to use
`apply` to do it `apply(M, c(1, 2), foo)`?
</summary>

``` r
library(microbenchmark)

inv <- function(x){1/x}

M <- matrix(1:1000000, nrow = 1000)

microbenchmark(inv(M), 
               apply(M, 1, inv))
```

    ## Unit: milliseconds
    ##              expr      min        lq      mean    median        uq      max
    ##            inv(M)  1.36041  1.706876  3.289155  2.017697  4.071511 28.75429
    ##  apply(M, 1, inv) 14.12604 15.196100 21.351325 17.632669 20.825277 79.07359
    ##  neval cld
    ##    100  a 
    ##    100   b

Based on the median time, using the function directly on the matrix is
\~9X faster than using `apply`

</details>

Define functions for the three weighting methods

``` r
inv <- function(x, k = 1){k/x}

inv_sqr <- function(x, k = 1){k/x^2}

lin <- function(x, k = 1){k - x}
```

Create Euclidean distance matrix to store Euclidean distances between
each test point and all the train points. Each row of the distance
matrix is a test point, each column is a train point, and the matrix
element `(i, j)` is the distance between test point `i` and train point
`j`. This matrix will be used to figure out each test point’s nearest
neighbors.

``` r
euclidean_dist <- function(x1, x2){
  # calculates Euclidean distance between vectors x1 and x2
  return(sqrt(sum((x1 - x2) ^ 2)))
}

# initialize matrix
distances_Euclidean <- matrix(0, nrow = nrow(usps_test), ncol = nrow(usps_train))

# populate matrix
ptm <- proc.time()
for(i in 1:nrow(usps_test)){
  for(j in 1:nrow(usps_train)){
    distances_Euclidean[i, j] <- euclidean_dist(usps_test[i, ], usps_train[j, ])
  }
}
proc.time() - ptm
```

    ##    user  system elapsed 
    ## 187.439  29.136 216.764

Run *k*NN with inverse weights

``` r
# create matrix with k cols & nrow(usps_test) rows to 
# store preds for every test point using 1 to k neighbors
# element [i, k] is the prediction of test point i using k neighbors
usps_test_pred_Euclidean_inv <- matrix(0, nrow = nrow(usps_test), ncol = 10)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train_labels, distances_Euclidean[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Euclidean[i, ]), ] # sort by distance
  max_dist <- max(trSetLabDist[, 2])
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store *WEIGHTED* counts of ten classes
  z_labels <- levels(usps_test_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:k){ # loop through the k neighbors and add the weight to z
      z[which(z_labels == trSetLabDist[m, 1])] <- 
        z[which(z_labels == trSetLabDist[m, 1])] + inv(trSetLabDist[m, 2], k = max_dist)
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Euclidean_inv[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
}
```

Run *k*NN with squared inverse weight

``` r
# create matrix with k cols & nrow(usps_test) rows to 
# store preds for every test point using 1 to k neighbors
# element [i, k] is the prediction of test point i using k neighbors
usps_test_pred_Euclidean_inv_sqr <- matrix(0, nrow = nrow(usps_test), ncol = 10)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train_labels, distances_Euclidean[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Euclidean[i, ]), ] # sort by distance
  max_dist <- max(trSetLabDist[, 2])
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store *WEIGHTED* counts of ten classes
  z_labels <- levels(usps_test_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:k){ # loop through the k neighbors and add the weight to z
      z[which(z_labels == trSetLabDist[m, 1])] <- 
        z[which(z_labels == trSetLabDist[m, 1])] + inv_sqr(trSetLabDist[m, 2], k = max_dist)
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Euclidean_inv_sqr[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
}
```

Run *k*NN with linear weight

``` r
# create matrix with k cols & nrow(usps_test) rows to 
# store preds for every test point using 1 to k neighbors
# element [i, k] is the prediction of test point i using k neighbors
usps_test_pred_Euclidean_lin <- matrix(0, nrow = nrow(usps_test), ncol = 10)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train_labels, distances_Euclidean[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Euclidean[i, ]), ] # sort by distance
  max_dist <- max(trSetLabDist[, 2])
  # make prediction for test point i for k = 1 to k = 10 (10 predictions per test point)
  z <- integer(10) # vector to store *WEIGHTED* counts of ten classes
  z_labels <- levels(usps_test_labels)
  for(k in 1:10){ # predictions for k = 1 to k = 10
    for(m in 1:k){ # loop through the k neighbors and add the weight to z
      z[which(z_labels == trSetLabDist[m, 1])] <- 
        z[which(z_labels == trSetLabDist[m, 1])] + lin(trSetLabDist[m, 2], k = max_dist)
    }
    # assign to class with highest z; in case of tie, pick randomly
    usps_test_pred_Euclidean_lin[i, k] <- sample(z_labels[which(z == max(z))], size = 1) 
  }
} 
```

Calculate error rates for the three weighting schemes

``` r
library(caret) # for function confusionMatrix

errorRate_Euclidean_inv <- numeric(10)
for(i in 1:10){
  errorRate_Euclidean_inv[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Euclidean_inv[, i]), 
                                                    usps_test_labels)$overall["Accuracy"]
}

errorRate_Euclidean_inv_sqr <- numeric(10)
for(i in 1:10){
errorRate_Euclidean_inv_sqr[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Euclidean_inv_sqr[, i]), 
                                                        usps_test_labels)$overall["Accuracy"]
}

errorRate_Euclidean_lin <- numeric(10)
for(i in 1:10){
  errorRate_Euclidean_lin[i] <- 1 - confusionMatrix(as.factor(usps_test_pred_Euclidean_lin[, i]), 
                                                    usps_test_labels)$overall["Accuracy"]
}
```

Plot the error rates

``` r
plot(1:10, errorRate_Euclidean_inv,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", ylab = "Misclassification Error Rate",
     ylim = c(0.052, 0.057))
points(1:10, errorRate_Euclidean_inv_sqr, type = "o", col = "blue")
points(1:10, errorRate_Euclidean_lin, type = "o", col = "black")
legend("top", legend = c("inverse", "squared inverse", "linear"),
       col = c("red", "blue", "black"), lty = 1)
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-23-1.png" width="672" />

``` r
cbind(errorRate_Euclidean_inv, errorRate_Euclidean_inv_sqr, errorRate_Euclidean_lin)
```

    ##       errorRate_Euclidean_inv errorRate_Euclidean_inv_sqr
    ##  [1,]              0.05630294                  0.05630294
    ##  [2,]              0.05630294                  0.05630294
    ##  [3,]              0.05630294                  0.05630294
    ##  [4,]              0.05430992                  0.05381166
    ##  [5,]              0.05331340                  0.05281515
    ##  [6,]              0.05480817                  0.05281515
    ##  [7,]              0.05331340                  0.05281515
    ##  [8,]              0.05331340                  0.05181863
    ##  [9,]              0.05331340                  0.05231689
    ## [10,]              0.05331340                  0.05331340
    ##       errorRate_Euclidean_lin
    ##  [1,]              0.05630294
    ##  [2,]              0.05630294
    ##  [3,]              0.05630294
    ##  [4,]              0.05480817
    ##  [5,]              0.05381166
    ##  [6,]              0.05480817
    ##  [7,]              0.05331340
    ##  [8,]              0.05331340
    ##  [9,]              0.05381166
    ## [10,]              0.05580468

-   using the squared inverse weighting scheme results in lower test
    error than using linear or inverse weighting schemes.

------------------------------------------------------------------------

## *k*NN nearest local centroid: USPS digits

**PROBLEM STATEMENT**: Apply the nearest local centroid classifier to
the USPS digits data with different values of *k*. Plot the test error
curve and interpret your results. What is the confusion matrix
corresponding to the optimal *k*?

Strategy

-   set initial centroid for all labels to max_dist
-   update centroids for labels within *k* neighbors
-   assign label as `which.min(centroids)`

<details>
<summary>
Load the data as above
</summary>

``` r
rm(list = ls())
gc()
```

    ##           used  (Mb) gc trigger  (Mb) max used  (Mb)
    ## Ncells 2677102 143.0    4916968 262.6  4916968 262.6
    ## Vcells 4562182  34.9   77520657 591.5 96900821 739.3

``` r
library(tidyverse)
usps_train <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)
usps_test <- read_delim(
  "https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz", 
  delim = " ", col_names = FALSE, show_col_types = FALSE)

# str(usps_train)

usps_train <- as.matrix(usps_train)
usps_test <- as.matrix(usps_test)

# first column contains labels
usps_train_labels <- usps_train[, 1]
usps_test_labels <- usps_test[, 1]

# drop first column of usps_train and usps_test
usps_train <- usps_train[, -1]
usps_test <- usps_test[, -1]

# drop last column (all NA) of usps_train
usps_train <- usps_train[, -257]

# convert labels to factors
usps_train_labels <- factor(usps_train_labels)
usps_test_labels <- factor(usps_test_labels)

str(usps_train_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 7 6 5 8 4 7 4 2 1 2 ...

``` r
str(usps_test_labels)
```

    ##  Factor w/ 10 levels "0","1","2","3",..: 10 7 4 7 7 1 1 1 7 10 ...

</details>

Create Euclidean distance matrix to store Euclidean distances between
each test point and all the train points.

Each row of the distance matrix is a test point, each column is a train
point, and the matrix element (i, j) is the distance between test point
i and train point j. This matrix will be used to figure out each test
point’s nearest neighbors.

``` r
euclidean_dist <- function(x1, x2){
  # calculates Euclidean distance between vectors x1 and x2
  return(sqrt(sum((x1 - x2) ^ 2)))
}

# initialize matrix
distances_Euclidean <- matrix(0, nrow = nrow(usps_test), ncol = nrow(usps_train))

# populate matrix
ptm <- proc.time()
for(i in 1:nrow(usps_test)){
  for(j in 1:nrow(usps_train)){
    distances_Euclidean[i, j] <- euclidean_dist(usps_test[i, ], usps_train[j, ])
  }
}
proc.time() - ptm # ~3.5 minutes
```

    ##    user  system elapsed 
    ## 184.455  24.907 209.568

calculate local centroids and assign test point to nearest local
centroid

``` r
# create matrix with k cols & nrow(usps_test) rows to 
# store preds for every test point using 1 to k neighbors
# element [i, k] is the prediction of test point i using k neighbors

system.time({
  

max_k <- 40
usps_test_pred_Euclidean_cent <- matrix(0, nrow = nrow(usps_test), ncol = max_k)
# for each test point i
for(i in 1:nrow(usps_test)){
  # cbind train labels and dist[i, ] and sort by distance
  trSetLabDist <- data.frame(usps_train_labels, distances_Euclidean[i, ])
  trSetLabDist <- trSetLabDist[order(distances_Euclidean[i, ]), ] # sort by distance
  max_dist <- max(trSetLabDist[, 2])
  # make prediction for test point i for k = 1 to k = max_k (max_k predictions per test point)
  z <- rep(max_dist, 10) # vector to store local centroids of ten classes
  z_labels <- levels(usps_test_labels)
  for(k in 1:max_k){ # predictions for k = 1 to k = max_k
    # loop through the k neighbors calculating centroids & dist to test point i

    neighbors <- data.frame(usps_train_labels, 
                              usps_train)[as.integer(rownames(trSetLabDist[1:k, ])), ]
    
    for (lbl in unique(neighbors[, 'usps_train_labels'])){
      
      # group neighbors by label
      lbl_neighbors <- neighbors[neighbors$usps_train_labels == lbl, -1]
      
      # calculate local centroid for each label
      lbl_centroid <- colSums(lbl_neighbors)/nrow(lbl_neighbors)
      
      # calculate distance from i to each centroid
      z[which(z_labels == lbl)] <- euclidean_dist(usps_test[i, ], lbl_centroid)
    }
    # assign to class with smallest z i.e. nearest local centroid
    # in case of tie, pick randomly
    usps_test_pred_Euclidean_cent[i, k] <- sample(z_labels[which(z == min(z))], size = 1) 
  }
}

}) # end system.time
```

    ##     user   system  elapsed 
    ## 1246.761  453.292 1704.106

Calculate and plot error rates for the `max_k` k values

``` r
library(caret) # for function confusionMatrix

errorRate_Euclidean_cent <- numeric(max_k)
for(i in 1:max_k){
  errorRate_Euclidean_cent [i] <- 
    1 - confusionMatrix(as.factor(usps_test_pred_Euclidean_cent [, i]), 
                        usps_test_labels)$overall["Accuracy"]
}

plot(1:max_k, errorRate_Euclidean_cent,
     type = "o", col = "red",
     xlab = "k (number of nearest neighbors used)", ylab = "Misclassification Error Rate",
     ylim = c(0.04, 0.057))
```

<img src="01-kNN-and-Crossvalidation_files/figure-gfm/unnamed-chunk-27-1.png" width="672" />

-   error rate at first decreases implying overfitting for small k
-   optimal k (lowest error rate) is 10
-   for k \> 10 test error rate is flat until k \~ 36 at which point it
    starts to rise
-   this rise might indicate underfitting

``` r
# confusion matrix for k = 10
confusionMatrix(as.factor(usps_test_pred_Euclidean_cent [, 10]), 
                        usps_test_labels)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1   2   3   4   5   6   7   8   9
    ##          0 355   0   7   0   0   1   1   0   2   0
    ##          1   0 255   0   0   2   0   0   1   0   0
    ##          2   2   0 185   2   0   1   2   1   2   0
    ##          3   0   0   0 154   0   1   0   1   3   0
    ##          4   0   5   1   0 188   0   1   2   0   2
    ##          5   0   0   0   7   2 154   1   0   2   0
    ##          6   0   3   0   0   1   0 165   0   1   0
    ##          7   1   1   2   1   2   0   0 141   0   3
    ##          8   0   0   3   0   0   1   0   1 153   1
    ##          9   1   0   0   2   5   2   0   0   3 171
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9571          
    ##                  95% CI : (0.9473, 0.9656)
    ##     No Information Rate : 0.1789          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9519          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity            0.9889   0.9659  0.93434  0.92771  0.94000  0.96250
    ## Specificity            0.9933   0.9983  0.99447  0.99728  0.99391  0.99350
    ## Pos Pred Value         0.9699   0.9884  0.94872  0.96855  0.94472  0.92771
    ## Neg Pred Value         0.9976   0.9949  0.99283  0.99351  0.99336  0.99674
    ## Prevalence             0.1789   0.1315  0.09865  0.08271  0.09965  0.07972
    ## Detection Rate         0.1769   0.1271  0.09218  0.07673  0.09367  0.07673
    ## Detection Prevalence   0.1824   0.1286  0.09716  0.07922  0.09915  0.08271
    ## Balanced Accuracy      0.9911   0.9821  0.96441  0.96250  0.96696  0.97800
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.97059  0.95918  0.92169  0.96610
    ## Specificity           0.99728  0.99462  0.99674  0.99290
    ## Pos Pred Value        0.97059  0.93377  0.96226  0.92935
    ## Neg Pred Value        0.99728  0.99677  0.99297  0.99671
    ## Prevalence            0.08470  0.07324  0.08271  0.08819
    ## Detection Rate        0.08221  0.07025  0.07623  0.08520
    ## Detection Prevalence  0.08470  0.07524  0.07922  0.09168
    ## Balanced Accuracy     0.98393  0.97690  0.95921  0.97950

------------------------------------------------------------------------

``` r
sessionInfo()
```

    ## R version 4.1.3 (2022-03-10)
    ## Platform: x86_64-apple-darwin13.4.0 (64-bit)
    ## Running under: macOS Big Sur/Monterey 10.16
    ## 
    ## Matrix products: default
    ## BLAS/LAPACK: /Users/felix.mbuga/opt/anaconda3/envs/rstud/lib/libopenblasp-r0.3.20.dylib
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] microbenchmark_1.4.9 forcats_0.5.2        stringr_1.4.1       
    ##  [4] dplyr_1.0.10         purrr_0.3.5          readr_2.1.3         
    ##  [7] tidyr_1.2.1          tibble_3.1.8         tidyverse_1.3.2     
    ## [10] caret_6.0-93         lattice_0.20-45      ggplot2_3.4.0       
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] nlme_3.1-159         fs_1.5.2             bit64_4.0.5         
    ##  [4] lubridate_1.8.0      httr_1.4.4           tools_4.1.3         
    ##  [7] backports_1.4.1      utf8_1.2.2           R6_2.5.1            
    ## [10] rpart_4.1.16         DBI_1.1.3            colorspace_2.0-3    
    ## [13] nnet_7.3-17          withr_2.5.0          tidyselect_1.2.0    
    ## [16] curl_4.3.3           bit_4.0.4            compiler_4.1.3      
    ## [19] cli_3.4.1            rvest_1.0.3          xml2_1.3.3          
    ## [22] sandwich_3.0-2       scales_1.2.1         mvtnorm_1.1-3       
    ## [25] proxy_0.4-27         digest_0.6.30        rmarkdown_2.16      
    ## [28] pkgconfig_2.0.3      htmltools_0.5.3      parallelly_1.32.1   
    ## [31] dbplyr_2.2.1         fastmap_1.1.0        highr_0.9           
    ## [34] rlang_1.0.6          readxl_1.4.1         rstudioapi_0.14     
    ## [37] generics_0.1.3       zoo_1.8-11           jsonlite_1.8.3      
    ## [40] vroom_1.6.0          ModelMetrics_1.2.2.2 googlesheets4_1.0.1 
    ## [43] magrittr_2.0.3       Matrix_1.5-1         Rcpp_1.0.9          
    ## [46] munsell_0.5.0        fansi_1.0.3          lifecycle_1.0.3     
    ## [49] multcomp_1.4-20      stringi_1.7.8        pROC_1.18.0         
    ## [52] yaml_2.3.6           MASS_7.3-58.1        plyr_1.8.8          
    ## [55] recipes_1.0.1        grid_4.1.3           parallel_4.1.3      
    ## [58] listenv_0.8.0        crayon_1.5.2         haven_2.5.1         
    ## [61] splines_4.1.3        hms_1.1.2            knitr_1.40          
    ## [64] pillar_1.8.1         future.apply_1.10.0  reshape2_1.4.4      
    ## [67] codetools_0.2-18     stats4_4.1.3         reprex_2.0.2        
    ## [70] glue_1.6.2           evaluate_0.16        data.table_1.14.6   
    ## [73] modelr_0.1.9         vctrs_0.5.1          tzdb_0.3.0          
    ## [76] foreach_1.5.2        cellranger_1.1.0     gtable_0.3.1        
    ## [79] future_1.29.0        assertthat_0.2.1     xfun_0.33           
    ## [82] gower_1.0.0          prodlim_2019.11.13   broom_1.0.1         
    ## [85] e1071_1.7-11         class_7.3-20         survival_3.4-0      
    ## [88] googledrive_2.0.0    gargle_1.2.1         timeDate_4021.104   
    ## [91] iterators_1.0.14     hardhat_1.2.0        lava_1.6.10         
    ## [94] globals_0.16.1       TH.data_1.1-1        ellipsis_0.3.2      
    ## [97] ipred_0.9-13
