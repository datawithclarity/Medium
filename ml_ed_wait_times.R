#############################################
# ER Peak Wait Time Analysis using XGBoost
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
  forcats # use for factors
)

# import data
# -----------

# set work directory
setwd("C:/Users/C/OneDrive/Desktop/Learning/")

# read csv
df <- read.csv("ER Wait Time Dataset.csv", 
               header = TRUE, 
               stringsAsFactors = FALSE
               )

# clean variable names (lower case, underscore only)
df <- clean_names(df)

# data preprocessing
# ------------------

# create new temporal factors and binary target variable
df <- df %>%
  mutate(
    visitdatetime = ymd_hms(visit_date), # reformat to POSIXct
    visithour = hour(visitdatetime), # extract hour
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

# generate descriptive statistics
# -------------------------------

# categorical predictors
# ......................

# gather factor variables
factorvars <- names(df)[sapply(df, is.factor)]
print(factorvars)

# calculate peak % per predictor level
cat_data <- df %>%
  select(all_of(factorvars), peak) %>%
  pivot_longer(cols = all_of(factorvars), 
               names_to = "Variable", 
               values_to = "Level") %>%
  group_by(Variable, Level) %>%
  summarise(
    total = n(),
    peakcount = sum(peak == 1),
    peakpercent = (peakcount / total) * 100,
    .groups = "drop"
    )

# generate box plot for peak % x predictor
catplot <- ggplot(cat_data, aes(x = Level, y = peakpercent, fill = Level)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(
    title = "Percentage of Peak by Categorical Predictors",
    x = "Level",
    y = "Peak (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.text.x = element_text(angle = 45, hjust = 1)
        )

# export figure
ggsave(
  "ed_catplot.png",
  plot = catplot,
  width = 6,
  height = 4,
  units = "in",
  dpi = 300
)

# numeric predictors
# ...................

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

# create subset of numerical predictors only, pivot for ggplot
num_data <- df %>%
  select(all_of(numericvars), peak) %>%
  pivot_longer(cols = all_of(numericvars), 
               names_to = "Variable", 
               values_to = "Value")

# generate box plots
numplot <- ggplot(num_data, aes(x = factor(peak), y = Value, fill = factor(peak))) +
  geom_jitter(width = 0.2, alpha = 0.3, color = "steelblue") +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  facet_wrap(~ Variable, scales = "free_y") +
  labs(
    title = "Distribution of Numeric Predictors by Peak (0/1)",
    x = "Peak (0 = No, 1 = Yes)",
    y = "Value",
    fill = "Peak"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  "ed_numplot.png",
  plot = numplot,
  width = 6,
  height = 4,
  units = "in",
  dpi = 300
)
# create frequency and mean sd table
table1 <- CreateTableOne(
  vars = c(numericvars, factorvars),
  factorVars = factorvars,
  data = df,
  includeNA = TRUE
)
print(table1, showAllLevels = TRUE)

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

# train xgboost model
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

# model evaluation
# -----------------
pred_prob <- predict(xgb_model, dtest)
pred_class <- ifelse(pred_prob > 0.3, 1, 0) # tune threshold to account for rare peaks

# confusion matrix and metrics
test_label_factor <- factor(test_label, levels = c(0,1))
pred_class_factor <- factor(pred_class, levels = c(0,1))
cm <- confusionMatrix(pred_class_factor, test_label_factor, positive = "1")

# evaluation metrics
accuracy  <- cm$overall['Accuracy']
recall    <- cm$byClass['Sensitivity']
precision <- cm$byClass['Precision']
f1_score  <- cm$byClass['F1']
FNR <- 1 - recall
roc_obj <- roc(test_label, pred_prob)
auroc <- auc(roc_obj)

metrics <- data.frame(
  Metric = c("Accuracy","Recall (Sensitivity)","Precision","F1 Score","AUROC","False Negative Rate"),
  Value = c(accuracy, recall, precision, f1_score, as.numeric(auroc), FNR)
)
print(metrics)

# shap, feature importance
# ------------------------
shap_train <- shap.values(xgb_model = xgb_model, X_train = train_matrix)

shap_score_train <- shap_train$shap_score

# generate mean shap score
shap_importance <- shap_score_train %>%
  as.data.frame() %>%
  summarise(across(everything(), mean)) %>%
  t() %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  rename(Mean_SHAP = V1) %>%
  arrange(Mean_SHAP)

# plot top 15 features
shaplot <- ggplot(shap_importance, aes(x = reorder(Feature, -Mean_SHAP), y = Mean_SHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top Features Impacting Peak",
       x = "Feature", y = "Mean |SHAP|") +
  theme_minimal()

ggsave(
  "ed_shaplot.png",
  plot = shaplot,
  width = 6,
  height = 4,
  units = "in",
  dpi = 300
)

# shap beeswarm plot
shap_long <- shap.prep(shap_contrib = shap_score_train, X_train = train_matrix)
beesplot <- shap.plot.summary(shap_long)

ggsave(
  "ed_beesplot.png",
  plot = beesplot,
  width = 6,
  height = 4,
  units = "in",
  dpi = 300
)

