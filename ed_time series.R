# ED time series 

# load packages
# -------------

if (!require(pacman)) install.packages("pacman")

pacman::p_load(
  forecast, # arima, forecasting
  dplyr, # manipulates data
  janitor, # formats column names
  lubridate, # formats dates
  tseries # time series test
)

# data preprocessing
# ------------------

# import
# set work directory
setwd("C:/Users/C/OneDrive/Desktop/Learning/ED time series")

# read csv
df <- read.csv("Supplementaryfile1.csv", 
               skip = 1,
               header = TRUE, 
               stringsAsFactors = FALSE)

# clean variable names (lower case, underscore only)
df <- clean_names(df)

df <- df %>% 
  # remove first 3 rows and last row
  slice(4:(n()-1)) %>%
# remove columns with all blank
  select(where(~ any(!is.na(.) & . != ""))) %>%
# remove columns that include "total"
  select(-matches("total"))

# create subsets per year
ed14 <- df %>% slice(2:25) 
ed15 <- df %>% slice(27:50)
ed16 <- df %>% slice(52:75)
ed17 <- df %>% slice(77:n())

# function to pivot and convert to datetime
pivot_dt <- function(df, year) {
  df %>%
    pivot_longer(
      cols = -x,
      names_to = "day_col",
      values_to = "count"
    ) %>%
    mutate(
      # convert count to numeric
      count = as.numeric(count),
      # convert Hour text to numeric 24-hour format
      hour_num = hour(parse_date_time(x, orders = "I p")),
      # clean day_col
      day_col = str_remove(day_col, "^x"),
      # extract day number
      day = as.integer(str_extract(day_col, "^[0-9]+")),
      # extract month abbreviation
      month = str_to_title(str_extract(day_col, "[a-z]+$")),
      # convert month abbreviation to numeric
      month_num = match(month, month.abb),
      # create datetime
      datetime = make_datetime(year = year, 
                               month = month_num, day = day, 
                               hour = hour_num, 
                               min = 0, 
                               sec = 0)
    ) %>%
    select(datetime, count)
}

# apply function to your data frames
ed14_dt <- pivot_dt(ed14, 2014)
ed15_dt <- pivot_dt(ed15, 2015)
ed16_dt <- pivot_dt(ed16, 2016)
ed17_dt <- pivot_dt(ed17, 2017)

# combine rows
ed <- bind_rows(ed14_dt, ed15_dt, ed16_dt, ed17_dt) %>%
  # remove rows where count is blank
  filter(!is.na(count)) %>%
  # arrange by date time
  arrange(datetime)

# create daily count
ed_daily <- ed %>%
  mutate(date = as.Date(datetime)) %>%  # extract date
  group_by(date) %>%
  summarise(daily_count = sum(count, na.rm = TRUE)) %>%
  arrange(date)

# fill in missing dates
ed_daily_complete <- ed_daily %>%
  # ensure date column is Date class
  mutate(date = as.Date(date)) %>%
  # fill in all dates in the year
  complete(
    date = seq.Date(from = as.Date(paste0(min(date), "-01-01")), 
                    to = as.Date(paste0(max(date), "-12-31")), 
                    by = "day"),
    fill = list(daily_count = NA)  
  ) %>%
  arrange(date)

# identify % of missing days
mean(is.na(ed_daily_complete$daily_count)) * 100

# arima
# -----

# adjust the aspect of the plot
options(repr.plot.width = 22, repr.plot.height = 10) 

# plot
ggplot(ed_daily_complete, aes(x = date, y = daily_count)) +
  geom_line(color = "steelblue") +
  labs(title = "Time Series of ED Patient Visits",
       x = "Date and Time",
       y = "ED Patient Visits") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 10), # Increase size of both axis titles
    axis.text = element_text(size = 10)   # Increase size of both axis labels
  )

# perform the augmented dickey-fuller test
adf.test(na.omit(ed_daily_complete$daily_count))

# create a time series object with start date and frequency
# with weekly frequency. Start on day 1 of 2014
ed_ts <- ts(ed_daily_complete$daily_count, 
                    start = c(2014, 1), 
                    frequency = 7)

# impute values through interpolation
ed_ts_imp <- na.interp(ed_ts)

# fit the best model using the time series with differencing
ts_model_auto <- auto.arima(ed_ts_imp, d = 1)

# print the model summary 
summary(ts_model_auto)

# perdict future values 
futurval <- forecast(ts_model_auto,h=7, level=c(90)) #confidence level 90%
plot(forecast(futurval))
futurval$mean
