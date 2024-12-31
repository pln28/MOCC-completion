library(tidyverse)
data <- read.csv("online_course_engagement_data.csv")
# Inspect the dataset
str(data)
summary(data)
# Check for missing values
missing_values <- sum(is.na(data))
cat("Total Missing Values in Dataset:", missing_values, "\n")

# ----- Descriptive Statistics -----
cat("\nDescriptive Statistics for Numerical Features:\n")
numerical_features <- data %>%
select(TimeSpentOnCourse, NumberOfVideosWatched, NumberOfQuizzesTaken, QuizScores, CompletionRate)
summary(numerical_features)

#Distribution of Course Categories
print(table(data$CourseCategory))

#Distribution of Device Types
print(table(data$DeviceType))

#Distribution of Course Completion
print(table(data$CourseCompletion))

# ----- Visualizations -----
# Histogram: Time Spent on Course
ggplot(data, aes(x = TimeSpentOnCourse)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Time Spent on Courses",
       x = "Time Spent on Course (hours)",
       y = "Frequency")

# Boxplot: Quiz Scores by Course Completion
ggplot(data, aes(x = as.factor(CourseCompletion), y = QuizScores, fill = as.factor(CourseCompletion))) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Quiz Scores by Course Completion",
       x = "Course Completion (0 = Not Completed, 1 = Completed)",
       y = "Quiz Scores (%)") +
  scale_fill_discrete(name = "Course Completion")

# Bar Chart: Course Completion Distribution
ggplot(data, aes(x = as.factor(CourseCompletion), fill = as.factor(CourseCompletion))) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Course Completion Distribution",
       x = "Course Completion (0 = Not Completed, 1 = Completed)",
       y = "Count") +
  scale_fill_discrete(name = "Course Completion")

# Stacked Bar Chart: Device Type vs. Course Completion
ggplot(data, aes(x = as.factor(DeviceType), fill = as.factor(CourseCompletion))) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Device Type vs. Course Completion",
       x = "Device Type (0 = Desktop, 1 = Mobile)",
       y = "Proportion") +
  scale_fill_discrete(name = "Course Completion")

