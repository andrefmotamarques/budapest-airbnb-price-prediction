# Installs the R packages used by airbnb_price_models.R (only the ones missing)
pkgs <- c("caret", "ggplot2", "dplyr", "corrplot", "randomForest", "gbm",
          "e1071", "ipred", "tidyr", "doParallel", "tidyverse")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
