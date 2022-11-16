#### 1. Libraries ----
library(tidyverse)              # Core tidyverse functions
library(tidymodels)             # Tidymodels 
library(usemodels)              # Template code suggestions for ML models
library(corrr)                  # Correlation
library(ggplot2)                # Plots
library(parsnip)                # ML Engines (random forest, KNN, xgBoost)
library(bonsai)                 # ML Engines (LightGBM)
library(workflows)              # Modeling workflows
library(recipes)                # Preprocessing recipies
library(rsample)                # Resampling
library(tune)                   # Parameter tuning
library(dials)                  # Adjustments
library(yardstick)              # Evaluation metrics
library(vip)                    # Variable importance
library(MASS)                   # Sample data

#### 2. Load data ----
data <- Boston

#### 3. Exploratory data analysis ----
#-- Plotting crime vs. median value
data %>%
    ggplot(aes(crim, medv))+
    geom_point(alpha = 0.4)+
    scale_x_log10()+
    scale_y_log10()+
    geom_point(color = 'midnightblue') +
    geom_smooth(method = 'lm', color = "tomato", se = T)+
    ggtitle("Median Value vs Crime") +
    xlab("Crime") +
    ylab("Median Value")

#-- Variable correlation    
data_corr <- data %>%
    correlate()%>%
    rearrange() %>%
    shave() %>%
    print()

#-- Wrangle data & feature engineering
model_df <- data %>%
    mutate(medv = log10(medv),
           crim = log10(crim))

#-- Select variables for data frame
# As required

#### 4. Build Models ----
#-- Partition data
set.seed(123)
df_split <- initial_split(model_df, strata = crim)
model_train <- training(df_split)
model_test <- testing(df_split)

#-- Bootstrap resamples
set.seed(234)
data_resamples <- bootstraps(model_train, strata = crim)
data_resamples

#-- Template model parameter suggestions from library(usemodels)
use_xgboost(crim ~., data = model_train)
#use_ranger(crim ~., data = model_train)
#use_kknn(crim ~., data = model_train)
#use_glmnet(crim ~., data = model_train)

#-- xgBoost recipe
xgboost_recipe <- 
    recipe(formula = crim ~ ., data = model_train) %>% 
    step_zv(all_predictors()) 
#   step_normalize() %>%
#   step_pca()

xgboost_spec <- 
    boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune(), 
               loss_reduction = tune(), sample_size = tune()) %>% 
    set_mode("regression") %>% 
    set_engine("xgboost") 

xgboost_workflow <- 
    workflow() %>%
    add_recipe(xgboost_recipe) %>% 
    add_model(xgboost_spec)

set.seed(94987)
xgboost_tune <-
    tune_grid(xgboost_workflow, resamples = data_resamples, grid = 11)

#### 5. Explore trained models ----
#-- Show best results
show_best(xgboost_tune, metric = "rsq")
show_best(xgboost_tune, metric = "rmse")
autoplot(xgboost_tune)

#-- Finalize workflow from best model
final_workflow_xgb <- xgboost_workflow %>%
    finalize_workflow(select_best(xgboost_tune))
final_workflow_xgb

#-- Fit finalized workflow from best model to entire training set & evaluate on test set
lastFit_xgb <- last_fit(final_workflow_xgb, df_split)

#-- Test set evaluation
collect_metrics(lastFit_xgb)
collect_predictions(lastFit_xgb)

#-- Plot test set evaluation
collect_predictions(lastFit_xgb) %>%
    ggplot(aes(crim, .pred))+
    geom_abline(lty = 2, color = "gray50")+
    geom_point(alpha = 0.6, color = "midnightblue")+
    coord_fixed()

#### 6. Predictions on new data ----
#-- Prediction with best model
predict(lastFit_xgb$.workflow[[1]],("ADD IN PREDICTION DATA HERE"))

#-- Model feature importance with library(vip)
varImp_spec <- xgboost_spec %>%
    finalize_model(select_best(xgboost_tune)) %>%
    set_engine("xgboost") 

workflow() %>%
    add_recipe(xgboost_recipe) %>%
    add_model(varImp_spec) %>%
    fit(model_train) %>%
    pull_workflow_fit() %>%
    vip(aesthetics = list(alpha = 0.8, fill = "midnightblue"))
    