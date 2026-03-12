# ER time series forcasting
# comparison of arima vs. xgboost

# load packages
# --------------

if (!require(pacman)) install.packages("pacman")

pacman::p_load(
  forecast,
  dplyr,
  janitor,
  lubridate,
  tseries,
  data.table,
  xgboost,
  caret,
  ggplot2,
  tidyr,
  stringr,
  zoo,
  timeDate,
  SHAPforxgboost,
  tibble
)

# import data
# -----------
setwd("C:/Users/C/OneDrive/Desktop/Learning/ED time series")

df <- read.csv(
  "Supplementaryfile1.csv",
  skip = 1,
  header = TRUE,
  stringsAsFactors = FALSE
)

df <- clean_names(df)

df <- df %>%
  slice(4:(n()-1)) %>%
  select(where(~ any(!is.na(.) & . != ""))) %>%
  select(-matches("total"))

# create yearly subsets
# ----------------------
ed14 <- df %>% slice(2:25)
ed15 <- df %>% slice(27:50)
ed16 <- df %>% slice(52:75)
ed17 <- df %>% slice(77:n())

# wrangle from wide to long format
# --------------------------------
pivot_dt <- function(df, year){
  
  df %>%
    pivot_longer(
      cols = -x,
      names_to = "day_col",
      values_to = "count"
    ) %>%
    mutate(
      
      count = as.numeric(count),
      
      hour_num = hour(parse_date_time(x, orders = "I p")),
      
      day_col = str_remove(day_col, "^x"),
      
      day = as.integer(str_extract(day_col, "^[0-9]+")),
      
      month = str_to_title(str_extract(day_col, "[a-z]+$")),
      
      month_num = match(month, month.abb),
      
      datetime = make_datetime(
        year = year,
        month = month_num,
        day = day,
        hour = hour_num
      )
      
    ) %>%
    select(datetime, count)
  
}

# apply pivot
ed14_dt <- pivot_dt(ed14, 2014)
ed15_dt <- pivot_dt(ed15, 2015)
ed16_dt <- pivot_dt(ed16, 2016)
ed17_dt <- pivot_dt(ed17, 2017)

# combine
ed <- bind_rows(ed14_dt, ed15_dt, ed16_dt, ed17_dt) %>%
  filter(!is.na(count)) %>%
  arrange(datetime)

# create daily ED counts
# ----------------------
ed_daily <- ed %>%
  mutate(date = as.Date(datetime)) %>%
  group_by(date) %>%
  summarise(
    daily_count = sum(count, na.rm = TRUE)
  ) %>%
  arrange(date)

# fill missing dates
# -------------------
ed_daily_complete <- ed_daily %>%
  complete(
    date = seq.Date(
      from = min(date),
      to = max(date),
      by = "day"
    ),
    fill = list(daily_count = NA)
  )

# interpolate missing values
# ---------------------------
ed_daily_imp <- ed_daily_complete
ed_daily_imp$daily_count <- na.approx(ed_daily_complete$daily_count)

# create holiday features
# -----------------------
years <- unique(year(ed_daily_imp$date))

can_holidays <- as.Date(
  unique(
    unlist(
      lapply(years, function(y) as.Date(timeDate::holidayTSX(y)))
    )
  )
)

pre_holidays  <- can_holidays - 1
post_holidays <- can_holidays + 1

# feature engineering
# -------------------

tsed_features <- ed_daily_imp %>%
  arrange(date) %>%
  mutate(
    dayofweek = wday(date),
    quarter = quarter(date),
    month = month(date),
    year = year(date),
    dayofyear = yday(date),
    dayofmonth = mday(date),
    weekofyear = isoweek(date),
    
    holiday = ifelse(date %in% can_holidays, 1, 0),
    pre_holiday = ifelse(date %in% pre_holidays, 1, 0),
    post_holiday = ifelse(date %in% post_holidays, 1, 0),
    
    lag_1 = lag(daily_count, 1),
    lag_7 = lag(daily_count, 7),
    
    roll7 = rollmean(
      daily_count,
      7,
      fill = NA,
      align = "right"
    )
  ) 

# train/test split (80/20)
# -------------------------

split_ts <- function(y, train = 0.8){
  
  n <- length(y)
  train_end <- floor(train * n)
  
  list(
    train = y[1:train_end],
    test  = y[(train_end+1):n]
  )
  
}

# evaluation metrics
rmse <- function(actual, pred){
  sqrt(mean((actual - pred)^2))
}

mae <- function(actual, pred){
  mean(abs(actual - pred))
}

mape <- function(actual, pred){
  
  idx <- actual != 0
  
  mean(
    abs((actual[idx] - pred[idx]) / actual[idx])
  ) * 100
  
}

y <- tsed_features$daily_count

X <- tsed_features %>%
  select(-daily_count, -date)

y_split <- split_ts(y)

train_idx <- 1:length(y_split$train)
test_idx  <- (length(y_split$train)+1):length(y)

X_train <- as.matrix(X[train_idx, ])
X_test  <- as.matrix(X[test_idx, ])

y_train <- y_split$train
y_test  <- y_split$test

# xgboost
# --------

dtrain <- xgb.DMatrix(X_train, label = y_train)

xgb_model <- xgb.train(
  params = list(objective = "reg:squarederror"),
  data = dtrain,
  nrounds = 500,
  verbose = 1
)

pred_test <- predict(xgb_model, X_test)

cat(
  "XGBoost Test RMSE:", rmse(y_test, pred_test),
  "MAE:", mae(y_test, pred_test),
  "MAPE:", mape(y_test, pred_test),
  "\n"
)

# arima
# ------

X_train_arima <- X_train
X_test_arima  <- X_test

arimax_model <- auto.arima(
  y_train,
  xreg = X_train_arima
)

pred_test_arima <- forecast(
  arimax_model,
  xreg = X_test_arima
)$mean

cat(
  "ARIMAX Test RMSE:", rmse(y_test, pred_test_arima),
  "MAE:", mae(y_test, pred_test_arima),
  "MAPE:", mape(y_test, pred_test_arima),
  "\n"
)

# plot prediction
# ---------------

fc_arimax <- forecast(arimax_model, xreg = X_test_arima, h = length(y_test))

# create new df for plotting
plot_df <- data.frame(
  date = tsed_features$date[test_idx],
  Actual = y_test,
  XGBoost = pred_test,
  ARIMAX = as.numeric(fc_arimax$mean),
  ARIMAX_lo = fc_arimax$lower[,2],  # 95% CI lower
  ARIMAX_hi = fc_arimax$upper[,2]   # 95% CI upper
)

# plot arima
plot_arimax <- ggplot(plot_df, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = ARIMAX, color = "ARIMAX"), linewidth = 1) +
  geom_ribbon(aes(ymin = ARIMAX_lo, ymax = ARIMAX_hi),
              fill = "red", alpha = 0.2) +
  labs(
    title = "ED Forecast: Actual vs ARIMAX",
    y = "Daily ED Visits",
    x = "Date"
  ) +
  scale_color_manual(values = c("Actual" = "black", "ARIMAX" = "red")) +
  theme_minimal()

plot_arimax

# plot xgboost
plot_xgb <- ggplot(plot_df, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = XGBoost, color = "XGBoost"), linewidth = 1) +
  labs(
    title = "ED Forecast: Actual vs XGBoost",
    y = "Daily ED Visits",
    x = "Date"
  ) +
  scale_color_manual(values = c("Actual" = "black", "XGBoost" = "blue")) +
  theme_minimal()

plot_xgb

# feature importance
# -------------------

dev.off() # turn off margin settings

# xgboost gain metric
# -------------------

importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)

# shap beeswarm plot
# ------------------
shap_train <- shap.values(xgb_model = xgb_model, X_train = X_train)
shap_score_train <- shap_train$shap_score
shap_long <- shap.prep(shap_contrib = shap_score_train, X_train = X_train)

mean_shap <- shap_long %>%
  group_by(variable) %>%
  summarise(mean_abs_shap = mean(abs(value), na.rm = TRUE)) %>%
  arrange(desc(mean_abs_shap))

# select top features
top_features <- mean_shap$variable[1:15]

# subset shap_long to only top features
shap_long_top <- shap_long %>%
  filter(variable %in% top_features)

# plot shap beeswarm for top features
shap.plot.summary(shap_long_top)

# plot to show train, test split
# ------------------------------

split_df <- data.frame(
  date = tsed_features$date,
  count = y,
  set = ifelse(1:length(y) %in% train_idx, "Train", "Test")
)

split_point <- tsed_features$date[max(train_idx)]

ggplot(split_df, aes(x = date, y = count, color = set)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = split_point,
    linetype = "dashed",
    color = "black",
    linewidth = 1
  ) +
  labs(
    title = "Train/Test Split (80/20)",
    subtitle = "Emergency Department Daily Visits",
    x = "Date",
    y = "Daily ED Visits",
    color = "Dataset"
  ) +
  scale_color_manual(
    values = c(
      "Train" = "steelblue",
      "Test" = "firebrick"
    )
  ) +
  theme_minimal()
