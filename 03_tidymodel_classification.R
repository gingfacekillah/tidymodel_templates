# Tidymodel Tempate: Classification, Multiple Models, Quick n' Dirty
# In this template multiple classification models (xgBoost, Random Forest, LightGBM, Decision Tree, Bayesian Logistic Regression, KNN) are
# quickly fitted on training data with minimal tuning. After finding a decent model, tune the grid to your requirements.

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
    geom_smooth(method = 'lm', color = "tomato", se = T) +
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
           crim = ifelse(crim>0.25651,1,0))

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

##-- Model Specifications
#-- Bayesian logistic regression
glm_stan_spec <- logistic_reg() %>%
    set_engine("stan") %>%
    set_mode("classification")

#-- Decision tree
decision_tree_spec <- decision_tree() %>%
    set_engine("rpart") %>%
    set_mode("classification")

#-- Random forest
rf_spec <- rand_forest() %>%
    set_engine("ranger", importance = "permutation") %>%
    set_mode("classification")

#-- XGBoost
xgb_spec <- boost_tree() %>%
    set_engine("xgboost") %>%
    set_mode("classification")

#-- KNN
knn_spec <- nearest_neighbor() %>%
    set_engine("kknn") %>% 
    set_mode("classification")
    
#-- LightGBM
lgbm_spec <- boost_tree() %>%
    set_engine("lightgbm") %>%
    set_mode("classification") 

# PCA recipe - normalize and perform PCA
pca_recipe <-
    recipe(crim ~ ., data = model_train) %>%
    step_normalize() %>%
    step_pca()

# Model workflow
model_workflow <- workflow() %>%
    add_recipe(pca_recipe)

##-- Model training outputs
#-- Bayesian logistic regression
glm_stan_results <- model_workflow %>%
    add_model(glm_stan_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#-- Decision tree
decision_tree_results <- model_workflow %>%
    add_model(decision_tree_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#-- Random forest
rf_results <- model_workflow %>%
    add_model(rf_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#-- XGBoost
xgb_results <- model_workflow %>%
    add_model(xgb_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#-- KNN
knn_results <- model_workflow %>%
    add_model(knn_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#-- LightGBM
lgbm_results <- model_workflow %>%
    add_model(lgbm_spec) %>%
    fit_resamples(
        resamples = data_resamples,
        control = control_resamples(save_pred = TRUE, verbose = TRUE))

#### 5. Explore trained models ----
#-- Evaluation metrics for all models trained
collect_metrics(glm_stan_results)
collect_metrics(decision_tree_results)
collect_metrics(rf_results)
collect_metrics(xgb_results)
collect_metrics(knn_results)
collect_metrics(lgbm_results)

#-- Confusion matrix
rf_results %>%
  conf_mat_resampled()
  
#-- ROC
rf_results %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(crim, .pred_0) %>%
  autoplot()

#-- Choose best model
# As required

#-- Tune selected model
