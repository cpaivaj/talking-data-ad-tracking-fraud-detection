## Projeto da Formação Cientista de Dados da Data Science Academy
## Prever se um clique em anuncio eh fraudulendo ou nao
## Entende-se como fraudulento quando clica, mas nao faz o download (is_attributed == 0)

## Dataset usado: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data?select=train_sample.csv

## CARREGANDO O DATASET
dataset <- read.csv("train_sample.csv")

# Target - is_attributed (classificacao)

View(dataset)
str(dataset)

# Observa-se que a coluna attributed_time so eh preenchida quando
#   quando o campo is_attributed == 1
# Logo, vou remover essa coluna do dataset pois ja tenho a click_time
#   que pode ser usada em conjunto com is_attributed

## FEATURE SELECTION e FEATURE ENGINEERING
# Removendo coluna
# Crio uma copia do dataset para nao mexer nos dados originais
dataset.v1 <- dataset
dataset.v1$attributed_time <- NULL
View(dataset.v1)

# Agora quero dividir a click_time em duas colunas
# click_date e click_time
# click_date vai armazenar apenas data
# click_hour vai armazenar apenas hora

# Extraindo datas e convertendo para POSIXct
dates <- as.POSIXct(dataset.v1$click_time)
class(dates)

# Pegando datas
dataset.v1$click_date <- format(dates, format = "%Y/%m/%d")

# Pegando o dia da semana com base na data
dataset.v1$click_weekday <- weekdays(as.Date(dataset.v1$click_date))

# Pegando horas (apenas hora, estou ignorando os minutos e segundos)
dataset.v1$click_hour <- as.numeric(format(dates, format = "%H"))

str(dataset.v1)

# Vou remover o campo click_time pois ja tenho os dados que preciso
#   e nao faz sentido manter dados duplicados
dataset.v1$click_time <- NULL

str(dataset.v1)
head(dataset.v1)

# Tranformando variaveis em fator
dataset.v1$click_date <- as.factor(dataset.v1$click_date)

# Funcao para agrupar as horas de acordo com meus parametros de partes do dis
# Manha - De 5 ate 12
# Tarde - De 12 ate 19
# Noite - De 19 ate 5
group_day_part <- function(x){
  if(x>5 && x<=12){
    return("Manha")
  }else if(x>12 && x<=19){
    return("Tarde")
  }else{
    return("Noite")
  }
}

# Armazenado as partes do dia em uma variavel nova
# Preciso usar o unlist() senao nao vou conseguir converter para fator, pois apos o lapply
#   a variavel fica do tipo list
dataset.v1$day_part <- unlist(lapply(dataset.v1$click_hour, group_day_part))
dataset.v1$day_part <- as.factor(dataset.v1$day_part)
dataset.v1$click_weekday <- as.factor(dataset.v1$click_weekday)
dataset.v1$is_attributed <- as.factor(dataset.v1$is_attributed)

# Decidi remover as horas pois ja tenho a informacao que queria (parte do dia)
dataset.v1$click_hour <- NULL

str(dataset.v1)

## ANALISE EXPLORATORIA
table(dataset.v1$is_attributed)
# Como ja era de se esperar, existem mais downloads nao efetuados

table(dataset.v1$click_weekday)
# Ja nessa outra tabela, observa-se que existem menos ocorrencias na segunda-feira
# E maior numero de ocorrencias na quarta
# Indicando que com base nesses dados, e nas datas analisadas, o dia da semana que
#   os anuncios sao mais clicados sao na quarta-feira

#install.packages("ggplot2")
library(ggplot2)
ggplot(dataset.v1[dataset.v1$is_attributed == 0,], aes(x = click_weekday)) +
  geom_bar() +
  ggtitle("CLIQUES SEM DOWNLOAD por DIA DA SEMANA")

ggplot(dataset.v1[dataset.v1$is_attributed == 0,], aes(x = day_part)) +
  geom_bar() +
ggtitle("CLIQUES SEM DOWNLOAD por PARTE DO DIA")

ggplot(dataset.v1[dataset.v1$is_attributed == 0,], aes(x = day_part, fill = click_weekday)) +
  labs(fill = "Dia da Semana") +
  geom_bar() +
  facet_grid(. ~ click_weekday) +
  ylab("Cliques") +
  xlab("Parte do Dia") +
  ggtitle("Downloads por parte do dia e dia da semana")

# Como podemos ver, na terca e quarta a noite ocorrem a maioria dos cliques
# Sendo esse o periodo com maior chance de ocorrer alguma fraude (clique sem download)

## CRIACAO DO MODELO
table(dataset.v1$is_attributed)

# Os dados estao desbalanceados
# Balanceando os dados atraves de undersamplig
#   (deixar os dados com maior qtd em uma quantidade igual ao menor, random)

# Separando as duas categorias de is_attributed (0 e 1)
dataset.v1.0 <- dataset.v1[dataset.v1$is_attributed == 0,]
dataset.v1.1 <- dataset.v1[dataset.v1$is_attributed == 1,]

dataset.v1.0 <- dataset.v1.0[sample(1:nrow(dataset.v1.1)),]

str(dataset.v1.0)
str(dataset.v1.1)

# Unindo os dois datasets
dataset.v2 <- merge(dataset.v1.0, dataset.v1.1, all = T)

str(dataset.v2)
table(dataset.v2$is_attributed)

# Sao poucos dados mas agora nao vai tender mais pra um dos lados

# Normalizacao
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando as variáveis
numeric.vars <- c('ip', 'app', 'device', 'os', 'channel')
dataset.v2 <- scale.features(dataset.v2, numeric.vars)

# Separando dados em treino (70%) e teste (30%)
train_data <- dataset.v2[1:round(nrow(dataset.v2) * 0.7),]
test_data <- dataset.v2[(round(nrow(dataset.v2) * 0.7)+1):(nrow(dataset.v2)),]

# Treinando o modelo
#install.packages("randomForest")
library(randomForest)

model.rf.v1 <- randomForest(is_attributed ~ .,
                         data = train_data,
                         ntree = 100,
                         nodesize = 10)

# Imprimindo resultado do treinamento v1
print(model.rf.v1)

## SCORE do modelo
predict.rf <- data.frame(observado = test_data$is_attributed,
                         previsto = predict(model.rf.v1, newdata = test_data))

# Visualizando resultado
View(predict.rf)

# Criando uma confusion matrix
library(caret)
confusionMatrix(predict.rf$observado, predict.rf$previsto)

## Otimizando o modelo

# Alterando variaveis usada pelo modelo
# Selecionando variaveis com randomForest
variables.rf <- randomForest(is_attributed ~ .,
                             data = dataset.v2,
                             ntree = 100,
                             nodesize = 10,
                             importance = TRUE)
# Visualizando resultado
varImpPlot(variables.rf)

# Recriando modelo com as variaveis mais importantes de acordo com o resultado acima
model.rf.v2 <- randomForest(is_attributed ~ app
                            + ip
                            + channel
                            + device
                            + os,
                            data = train_data,
                            ntree = 100,
                            nodesize = 10)

# Imprimindo resultado do treinamento v2
print(model.rf.v2)

## SCORE do modelo v2
predict.rf.v2 <- data.frame(observado = test_data$is_attributed,
                         previsto = predict(model.rf.v2, newdata = test_data))

# Visualizando resultado
View(predict.rf.v2)

# Criando uma confusion matrix
library(caret)
confusionMatrix(predict.rf.v2$observado, predict.rf.v2$previsto)

# Como observado na confusion matrix acima, obtivemos uma melhora na
#   acuracia, apenas alterando as variaveis usadas para o treinamento






