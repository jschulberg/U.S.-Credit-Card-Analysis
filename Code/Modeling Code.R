#######################################################################
#######################################################################

#####                  U.S. Credit Card Analysis                  #####

####################################################################### 
#######################################################################
# In this script, I will analyze the U.S. Credit Card dataset included in 
# the data folder of this repository. I will use a variety of exploratory
# and modeling techniques to do so, including, but not limited to:

#   - k-Nearest Neighbor (kNN)  
#   - K-Means Clustering  
#   - Support Vector Machine (SVM)    
#   - Other methods 


#######################################################################
# Set Up --------------------------------------------------------------
#######################################################################
# Bring in packages
suppressMessages(library("dplyr")) # Used for data cleaning
suppressMessages(library("tidyr")) # Used for data cleaning
suppressMessages(library("ggplot2")) # Used for visualizations
suppressMessages(library("kernlab")) # Used for Support Vector Machine
suppressMessages(library("kknn")) # Used for k-Nearest Neighbors

# bring in the data, delimited by a tab ("\t")
data_cc <- read.delim(here::here("Data/credit_card_data-headers.txt"), header = T, sep = "\t")

# convert to a tibble
data_cc <- as_tibble(data_cc)

# Let's take a peek under the hood
head(data_cc)
summary(data_cc)

# Let's clean up the data based on the data description provided.
# A1:	b, a.
# A2:	continuous.
# A3:	continuous.
# A8:	continuous.
# A9:	t, f.
# A10:	t, f.
# A11:	continuous.
# A12:	t, f.
# A14:	continuous.
# A15:	continuous.
# R1: +,-         (class attribute)
# data$R1 <- as.factor(data$R1)

data_cc$R1 <- as.factor(data_cc$R1)

# What does our response variable look like?
table(data_cc$R1)
round(prop.table(table(data_cc$R1)), 2) # 55% of the observations are 0, and 45% are 1


########################################################################
# Support Vector Machine -----------------------------------------------
######################################################################## 
# In the first section, we'll use a simple SVM model to predict on our 
# credit card data.

# Set the seed to ensure proper randomization/reproducibility
set.seed(123)

# Let's try our first Support Vector Machine model.
model1 <- ksvm(as.matrix(data_cc[, 1:10]), # independent variables
               data_cc[, 11], # dependent variables
               type = "C-svc", 
               kernel = "vanilladot",
               C = 100, # also known as our lambda value (used for soft-classifiers)
               scaled = TRUE # used to scale the data
               )

# calculate a1.am
a <- colSums(model1@xmatrix[[1]] * model1@coef[[1]])
a

# calculate a0
a0 <- -(model1@b)
a0

# see what the model1 predicts
pred <- predict(model1, data_cc[, 1:10])
pred
table(pred)

# see what fraction of the model1's predictions match the actual classification
mean(pred == data_cc$R1) # Our accuracy is 86.39%

# Let's try out the Radial Basis kernel "Gaussian"
model_rbfdot <- ksvm(as.matrix(data_cc[,1:10]), data_cc[,11], type = "C-svc", kernel = "rbfdot",
                     C = 100, scaled = TRUE)
pred_rbfdot <- predict(model_rbfdot, data_cc[, 1:10])
mean(pred_rbfdot == data_cc$R1) # Our accuracy is 95.26%

# Now let's try this approach for all of the kernels available to see which works best
# First define a list of the different kernels available. We'll include vanilladot
# to ensure that the result matches what we originally got
kernels <- c("vanilladot", "rbfdot", "polydot", "tanhdot",
             "laplacedot", "besseldot", "anovadot", "splinedot")

# We'll initialize an empty vector that we can later fill with the names of
# the kernels as well as the accompanying accuracy of each kernel
kernel_accuracy <- c()

# Run a for loop through all the kernels above
for (i in kernels) {
  cat("                                                  ", sep = "\n")
  cat("==================================================", sep = "\n")
  
  # Set the kernel in the ksvm to 'i'
  kernel_model <- ksvm(as.matrix(data_cc[,1:10]), data_cc[,11], type = "C-svc", kernel = i,
                       C = 100, scaled = TRUE)
  kernel_pred <- predict(kernel_model, data_cc[, 1:10])
  # Find the accuracy of our predictions
  accuracy <- mean(kernel_pred == data_cc$R1)
  
  # Let's also produce the end formula for each kernel
  a <- colSums(kernel_model@xmatrix[[1]] * kernel_model@coef[[1]])
  a0 <- -(kernel_model@b)
  
  # Let's print out our results
  cat("The kernel ", i, " has an accuracy of ", accuracy, ".", sep = "")
  
  # Let's also save our results into a new list
  kernel_accuracy[i] <- accuracy
  cat("                                                  ", sep = "\n")
  cat("==================================================", sep = "\n")
}

# Let's now bring in the names of the kernels to our accuracy vector
kernel_accuracy_df <- kernel_accuracy %>%
  cbind(kernels) %>%
  # convert it into a data frame
  as.data.frame()

# rename the first column to be accuracy
names(kernel_accuracy_df)[1] <- "accuracy"


### Visualization
# order by accuracy in a descending fashion
kernel_accuracy_ordered <- kernel_accuracy_df %>%
  arrange(desc(accuracy))

# let's also round this column to make it more visually appealing
kernel_accuracy_ordered$accuracy <- as.numeric(as.character(kernel_accuracy_ordered$accuracy))
kernel_accuracy_ordered$accuracy <- 100*round(kernel_accuracy_ordered$accuracy, 2)

# Let's plot our results to better visualize everything
ggplot(kernel_accuracy_ordered,
       # order by accuracy
       aes(x = reorder(kernels, accuracy), y = accuracy)) +
  # Let's make it a bar graph and change the color
  geom_col(fill = "slateblue2", color = "black") +
  # Add the text labels in for p-values so it's easier to read
  geom_label(label = kernel_accuracy_ordered$accuracy) +
  # Change the theme to minimal
  theme_minimal() +
  # Let's change the names of the axes and title
  xlab("Kernel") +
  ylab("Accuracy (%)") +
  ggtitle("Accuracy of Different Kernels \n in SVM Method") +
  # Let's flip the axes
  coord_flip(ylim = c(50, 100)) +
  # center the title
  theme(plot.title = element_text(hjust = 0.5))
