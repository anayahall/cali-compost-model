# Farm Operations and Expenses by County

fo <- read_csv("data/CA_farmoperations_raw.csv") %>% 
  filter(Domain == "TOTAL") %>% 
  filter(`Data Item` == "FARM OPERATIONS - ACRES OPERATED" | `Data Item` == "FARM OPERATIONS - NUMBER OF OPERATIONS")


fe <- read_csv("data/CA_fertilizerexpenses_raw.csv")

farm_ops <- rbind(fo,fe)
