library("circlize")

# mat = matrix(1:9, 3)
# rownames(mat) = letters[1:3]
# colnames(mat) = LETTERS[1:3]
# mat
# 
# df = data.frame(from = letters[1:3], to = LETTERS[1:3], value = 1:3)
# df

set.seed(999)
mat = matrix(sample(18, 18), 3, 6) 
rownames(mat) = paste0("E", 1:3)
colnames(mat) = paste0("E", 1:6)
mat

df = data.frame(from = rep(rownames(mat), times = ncol(mat)),
                to = rep(colnames(mat), each = nrow(mat)),
                value = as.vector(mat),
                stringsAsFactors = FALSE)
df

chordDiagram(mat)

data <- read.csv(file= 'results/chord/C2F_FG_50.csv')
data$from <- as.character(data$from)
data$to <- as.character(data$to)
data <- subset(data, select = c('from', 'to', 'value'))

chordDiagram(data)

# circos.clear()