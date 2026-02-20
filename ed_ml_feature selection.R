#############################################
# ER Peak Wait Time Analysis using XGBoost
# Exploring Feature Selection
#############################################

# load packages
# -------------

if (!require(pacman)) install.packages("pacman")

pacman::p_load(
  caret, # configures ML model
  dplyr, # manipulates data
  janitor, # cleans data
  lubridate, # formats dates
  tableone, # formats descriptive stats tables
  xgboost, # implements xgboost
  pROC, # implements ROC curves
  readr, # reads csv
  ggplot2, # visualizes data
  SHAPforxgboost, # implements SHAP for ML
  tibble, # optimizes data frames
  tidyr, # use with dplyr and ggplot
  forcats, # use for factors
  vip # use for permutation importance
)

# import data
# -----------

# set work directory
setwd("C:/Users/C/OneDrive/Desktop/Learning/ED feature engineering")

# read csv
df <- read.csv("ER Wait Time Dataset.csv", 
               header = TRUE, 
               stringsAsFactors = FALSE
               )

# clean variable names (lower case, underscore only)
df <- clean_names(df)

# data pre-processing
# ------------------

# create new temporal factors and binary target variable
df <- df %>%
  mutate(
    peak = ifelse(total_wait_time_min >= (3*60), 1, 0) # create binary target
  ) %>%
  select(-visit_date) # drop visit date

# check
table(df$peak)
prop.table(table(df$peak)) * 100

# convert character type to factor type
charvars <- c(
  "hospital_name", "region", "day_of_week", "season","time_of_day", "urgency_level"
)
df <- df %>% 
  mutate(
    across(all_of(charvars), factor)
  )

# gather factor variables
factorvars <- names(df)[sapply(df, is.factor)]
print(factorvars)

# gather numeric predictors
numericvars <- names(df)[sapply(df, is.numeric)]
print(numericvars)

# exclude target and other numeric variables not used as predictors
numericvars <- setdiff(numericvars, c("peak", 
                                      "total_wait_time_min", 
                                      "patient_satisfaction",
                                      "time_to_registration_min",
                                      "time_to_triage_min",
                                      "time_to_medical_professional_min"
                                      ))

# split train and test data
# ---------------------------

# for reproducibility
set.seed(123)

# do 80% train 20% test split
trainindex <- createDataPartition(df$peak, p = 0.8, list = FALSE)

# separate train and test data
train <- df[trainindex, ]
test  <- df[-trainindex, ]

# prepare matrices for xgboost
# -----------------------------

# identify predictors
predictors <- c("hospital_name", "region", "day_of_week", "season",
                "time_of_day",
                "urgency_level", "nurse_to_patient_ratio", 
                "specialist_availability",
                "facility_size_beds")

# train matrix
train_matrix <- model.matrix(~ . - 1, data = train[, predictors])
train_label  <- train$peak
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

# test matrix
test_matrix  <- model.matrix(~ . - 1, data = test[, predictors])
test_label   <- test$peak
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# train full xgboost model
params <- list(
  objective = "binary:logistic"
  # all other parameters set to default
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  evals = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# evaluate feature selection methods
# ----------------------------------

# 1. xgboost built in gain importance
# ..............................
xgb_importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model) %>%
  arrange(desc(Gain)) %>%
  slice(1:10) # select top features

top_features_xgb <- xgb_importance$Feature

# 2. permutation importance
# .........................
set.seed(123)

# Convert train labels to factor
train_label_factor <- factor(train_label, levels = c(0, 1))  # 1 is positive

perm_importance <- vi_permute(
  object = xgb_model,
  feature_names = colnames(train_matrix),
  train = as.data.frame(train_matrix),  # raw training data
  target = train_label_factor,
  metric = "roc_auc",
  pred_wrapper = function(object, newdata) {
    predict(object, newdata = xgb.DMatrix(as.matrix(newdata)))
  },
  event_level = "second" # "1" is the positive class
)

# select top features
perm_importance <- perm_importance %>%
  arrange(desc(Importance)) %>%
  slice(1:10) 

top_features_perm <- perm_importance$Variable

# model evaluation
# -----------------

# create function
train_eval_xgb <- function(feature_subset) {
  
  train_mat <- xgb.DMatrix(data = train_matrix[, feature_subset], label = train_label)
  test_mat  <- xgb.DMatrix(data = test_matrix[, feature_subset], label = test_label)
  
  model <- xgb.train(
    params = params,
    data = train_mat,
    nrounds = 100,
    evals = list(train = train_mat, test = test_mat),
    early_stopping_rounds = 10,
    print_every_n = 10
  )
  
  pred_prob  <- predict(model, test_mat)
  pred_class <- ifelse(pred_prob > 0.3, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred_class, levels = c(0,1)),
    factor(test_label, levels = c(0,1)),
    positive = "1"
  )
  
  FNR <- 1 - cm$byClass['Sensitivity']
  roc_obj <- roc(test_label, pred_prob)
  auroc <- auc(roc_obj)
  
  metrics <- data.frame(
    Metric = c("Accuracy","Recall (Sensitivity)","Precision","F1 Score","AUROC","False Negative Rate"),
    Value = c(
      cm$overall['Accuracy'],
      cm$byClass['Sensitivity'],
      cm$byClass['Precision'],
      cm$byClass['F1'],
      as.numeric(auroc),
      FNR
    )
  )
  
  return(list(model = model, metrics = metrics))
}

# train & evaluate model using top xgboost gain features
res_xgb <- train_eval_xgb(top_features_xgb)
print(res_xgb$metrics)

# train & evaluate model using top permutation features
res_perm <- train_eval_xgb(top_features_perm)
print(res_perm$metrics)

# compare metrics side by side
comparison <- res_xgb$metrics %>%
  rename(XGB_Importance = Value) %>%
  left_join(res_perm$metrics %>% rename(Perm_Importance = Value), by = "Metric")

print(comparison)
