library(Matrix)

# X
csv2_matrix <- as.matrix(csv2)
csv_sparse <- Matrix(csv2_matrix, sparse=TRUE)
writeMM(csv_sparse, file="testX2_MatrixMarket.mtx")

# Y
y <- read.csv('testY.csv', header=FALSE)
y <- y[2]
y_sparse = Matrix(as.matrix(y), sparse=TRUE)
writeMM(y_sparse, file="testY_MatrixMarket.mtx")
