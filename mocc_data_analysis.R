usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1])) 
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

usePackage("tidyverse")
usePackage("ggplot2")
usePackage("caret")
usePackage("pROC")
usePackage("randomForest")
usePackage("xgboost")
usePackage("Matrix")
usePackage("shiny")
usePackage("shinydashboard")
usePackage("ggcorrplot")

data <- read.csv("online_course_engagement_data.csv")
#Inspect the dataset
str(data)
summary(data)
#Check for missing values
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
#Histogram: Time Spent on Course
ggplot(data, aes(x = TimeSpentOnCourse)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Time Spent on Courses",
       x = "Time Spent on Course (hours)",
       y = "Frequency")

#Boxplot: Quiz Scores by Course Completion
ggplot(data, aes(x = as.factor(CourseCompletion), y = QuizScores, fill = as.factor(CourseCompletion))) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Quiz Scores by Course Completion",
       x = "Course Completion (0 = Not Completed, 1 = Completed)",
       y = "Quiz Scores (%)") +
  scale_fill_discrete(name = "Course Completion")

#Bar Chart: Course Completion Distribution
ggplot(data, aes(x = as.factor(CourseCompletion), fill = as.factor(CourseCompletion))) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Course Completion Distribution",
       x = "Course Completion (0 = Not Completed, 1 = Completed)",
       y = "Count") +
  scale_fill_discrete(name = "Course Completion")

#Stacked Bar Chart: Device Type vs. Course Completion
ggplot(data, aes(x = as.factor(DeviceType), fill = as.factor(CourseCompletion))) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Device Type vs. Course Completion",
       x = "Device Type (0 = Desktop, 1 = Mobile)",
       y = "Proportion") +
  scale_fill_discrete(name = "Course Completion")

# ----- Correlation Analysis -----

#Calculate the correlation matrix
numerical_data <- data %>%
  select(TimeSpentOnCourse, NumberOfVideosWatched, NumberOfQuizzesTaken, QuizScores, CompletionRate)

correlation_matrix <- cor(numerical_data, use = "complete.obs")
print(correlation_matrix)

#Correlation matrix heatmap
library(ggcorrplot)
ggcorrplot(correlation_matrix, 
           lab = TRUE, 
           title = "Correlation Heatmap for Numerical Features",
           colors = c("red", "white", "blue"))
# Scatterplot: Time Spent on Course vs Completion Rate
ggplot(data, aes(x = TimeSpentOnCourse, y = CompletionRate, color = as.factor(CourseCompletion))) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Time Spent on Course vs Completion Rate",
       x = "Time Spent on Course (hours)",
       y = "Completion Rate (%)",
       color = "Course Completion")

# ----- Logistic Regression -----

#Split data into training and testing sets
set.seed(123) 
train_index <- createDataPartition(data$CourseCompletion, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#Build logistic regression model
logistic_model <- glm(CourseCompletion ~ TimeSpentOnCourse + NumberOfVideosWatched + 
                        NumberOfQuizzesTaken + QuizScores + CompletionRate + DeviceType, 
                      data = train_data, 
                      family = binomial)
summary(logistic_model)

#Predict on test data
test_data$predicted_prob <- predict(logistic_model, newdata = test_data, type = "response")
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.5, 1, 0)

#Confusion matrix
confusion <- confusionMatrix(as.factor(test_data$predicted_class), as.factor(test_data$CourseCompletion))
print(confusion)

#Calculate and plot ROC curve
roc_curve <- roc(test_data$CourseCompletion, test_data$predicted_prob)
plot(roc_curve, col = "blue", main = "ROC Curve for Logistic Regression")
cat("\nAUC (Area Under Curve):", auc(roc_curve), "\n")

#Model accuracy
accuracy <- confusion$overall["Accuracy"]
cat("\nModel Accuracy:", accuracy, "\n")

#Extract coefficients from logistic regression model
coefficients <- summary(logistic_model)$coefficients
coefficients <- as.data.frame(coefficients)
coefficients <- coefficients %>% 
  mutate(Feature = rownames(coefficients)) %>% 
  rename(Coefficient = Estimate, StdError = `Std. Error`, ZValue = `z value`, PValue = `Pr(>|z|)`)

#Filter out non-significant features (p-value > 0.05)
significant_features <- coefficients %>% filter(PValue < 0.05)
print(significant_features)

# Barplot: Feature Importance (Coefficients)
ggplot(significant_features, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Feature Importance (Logistic Regression Coefficients)",
       x = "Feature",
       y = "Coefficient")
# ----- XGBoost Model -----

#Convert data to a sparse matrix format for XGBoost
train_matrix <- sparse.model.matrix(CourseCompletion ~ TimeSpentOnCourse + NumberOfVideosWatched +
                                      NumberOfQuizzesTaken + QuizScores + CompletionRate + DeviceType, 
                                    data = train_data)[, -1]
test_matrix <- sparse.model.matrix(CourseCompletion ~ TimeSpentOnCourse + NumberOfVideosWatched +
                                     NumberOfQuizzesTaken + QuizScores + CompletionRate + DeviceType, 
                                   data = test_data)[, -1]

#Convert labels to numeric
train_labels <- as.numeric(train_data$CourseCompletion)
test_labels <- as.numeric(test_data$CourseCompletion)

#Train XGBoost model
set.seed(123)
xgb_model <- xgboost(data = train_matrix, label = train_labels, 
                     objective = "binary:logistic", nrounds = 100, 
                     max_depth = 6, eta = 0.1, verbose = 0)
xgb_predictions <- predict(xgb_model, newdata = test_matrix)
xgb_predicted_class <- ifelse(xgb_predictions > 0.5, 1, 0)

#Confusion Matrix
xgb_confusion <- confusionMatrix(as.factor(xgb_predicted_class), as.factor(test_labels))
print(xgb_confusion)

#ROC Curve
xgb_roc <- roc(test_labels, xgb_predictions)
plot(xgb_roc, col = "red", main = "ROC Curve: XGBoost")
cat("\nXGBoost AUC:", auc(xgb_roc), "\n")

#Accuracy
xgb_accuracy <- xgb_confusion$overall["Accuracy"]
cat("\nXGBoost Accuracy:", xgb_accuracy, "\n")

#Feature importance
xgb_importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)

#Print feature importance table
print(xgb_importance)

#Plot feature importance
xgb.plot.importance(xgb_importance, top_n = 10, 
                    main = "Feature Importance: XGBoost")

# ----- Random Forest -----

#Train Random Forest Model
set.seed(123)  
rf_model <- randomForest(as.factor(CourseCompletion) ~ TimeSpentOnCourse + NumberOfVideosWatched + 
                           NumberOfQuizzesTaken + QuizScores + CompletionRate + DeviceType, 
                         data = train_data, 
                         ntree = 100, 
                         mtry = 3, 
                         importance = TRUE)
print(rf_model)

#Predict on Test Data
test_data$rf_predicted <- predict(rf_model, newdata = test_data)

#Confusion Matrix
rf_confusion <- confusionMatrix(as.factor(test_data$rf_predicted), as.factor(test_data$CourseCompletion))
print(rf_confusion)

#ROC Curve for Random Forest
rf_roc <- roc(test_data$CourseCompletion, as.numeric(test_data$rf_predicted))
plot(rf_roc, col = "darkgreen", main = "ROC Curve: Random Forest")
cat("\nRandom Forest AUC:", auc(rf_roc), "\n")

#Extract feature importance from Random Forest model
importance <- as.data.frame(importance(rf_model))
importance$Feature <- c("TimeSpentOnCourse", "NumberOfVideosWatched", 
                        "NumberOfQuizzesTaken", "QuizScores", "CompletionRate", "DeviceType")

#Barplot: Feature Importance by Mean Decrease Gini
ggplot(importance, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Feature Importance: Random Forest",
       x = "Feature",
       y = "Mean Decrease Gini")

#Choose Random Forest as the final model
#Visualize top features by course completion
top_features <- c("CompletionRate", "NumberOfQuizzesTaken", "QuizScores")

for (feature in top_features) {
  plot <- ggplot(data, aes(x = .data[[feature]], fill = as.factor(CourseCompletion))) +
    geom_histogram(position = "dodge", bins = 30) +
    theme_minimal() +
    labs(title = paste("Distribution of", feature, "by Course Completion"),
         x = feature,
         y = "Count",
         fill = "Course Completion") +
    scale_fill_discrete(name = "Course Completion")
  
  print(plot) #print each plot
}

#Save feature importance plot
ggsave("feature_importance_random_forest.png", plot = last_plot())

# Save feature-specific distributions
for (feature in top_features) {
  plot <- ggplot(data, aes(x = .data[[feature]], fill = as.factor(CourseCompletion))) +
    geom_histogram(position = "dodge", bins = 30) +
    theme_minimal() +
    labs(title = paste("Distribution of", feature, "by Course Completion"),
         x = feature,
         y = "Count",
         fill = "Course Completion") +
    scale_fill_discrete(name = "Course Completion")
  
  ggsave(paste0(feature, "_distribution.png"), plot = plot)
}

# ----- Cross-Validation for Random Forest -----
control <- trainControl(method = "cv", number = 5)

#Train Random Forest model with cross-validation
set.seed(123)
rf_cv_model <- train(as.factor(CourseCompletion) ~ TimeSpentOnCourse + NumberOfVideosWatched + 
                       NumberOfQuizzesTaken + QuizScores + CompletionRate + DeviceType,
                     data = train_data,
                     method = "rf",
                     trControl = control,
                     tuneLength = 3)
print(rf_cv_model)

#Performance on test data
rf_cv_predictions <- predict(rf_cv_model, newdata = test_data)
rf_cv_confusion <- confusionMatrix(rf_cv_predictions, as.factor(test_data$CourseCompletion))
print(rf_cv_confusion)


# ----- Dashboard -----
data <- read.csv("online_course_engagement_data.csv")

#UI
ui <- dashboardPage(
  dashboardHeader(title = "Course Completion Analysis"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Feature Importance", tabName = "importance", icon = icon("bar-chart")),
      menuItem("Feature Distributions", tabName = "distributions", icon = icon("area-chart"))
    )
  ),
  dashboardBody(
    tabItems(
      #Tab for feature importance
      tabItem(tabName = "importance",
              fluidRow(
                box(title = "Feature Importance (Random Forest)", status = "primary", solidHeader = TRUE,
                    plotOutput("importancePlot", height = "400px"))
              )),
      
      #Tab for feature distributions
      tabItem(tabName = "distributions",
              fluidRow(
                box(title = "Distribution of Features by Course Completion", status = "primary", solidHeader = TRUE,
                    selectInput("feature", "Select a Feature:", choices = c("CompletionRate", "NumberOfQuizzesTaken", "QuizScores")),
                    plotOutput("distributionPlot", height = "400px"))
              ))
    )
  )
)

server <- function(input, output) {
  
  #Render feature importance plot
  output$importancePlot <- renderPlot({
    ggplot(importance, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_bar(stat = "identity", fill = "forestgreen") +
      theme_minimal() +
      coord_flip() +
      labs(title = "Feature Importance: Random Forest",
           x = "Feature",
           y = "Mean Decrease Gini")
  })
  
  #Render feature distribution plot
  output$distributionPlot <- renderPlot({
    ggplot(data, aes_string(x = input$feature, fill = "as.factor(CourseCompletion)")) +
      geom_histogram(position = "dodge", bins = 30) +
      theme_minimal() +
      labs(title = paste("Distribution of", input$feature, "by Course Completion"),
           x = input$feature,
           y = "Count",
           fill = "Course Completion")
  })
}

#Run
shinyApp(ui = ui, server = server)

