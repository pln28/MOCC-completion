library(tidyverse)
data <- read.csv("online_course_engagement_data.csv")
# Inspect the dataset
str(data)
summary(data)
# Check for missing values
missing_values <- sum(is.na(data))
cat("Total Missing Values in Dataset:", missing_values, "\n")