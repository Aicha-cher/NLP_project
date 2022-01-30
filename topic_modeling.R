library(tidytext)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(ggplot2)
library(stringr)
library(spacyr)
library(widyr)
library(keras)
library(tensorflow)
path <- paste(getwd(), "/data/txt_Files_of_Climate_Change_Testimonies_by_Scientists_1985_2013",sep="")
setwd(path)
testimonies <- list.files(getwd(),pattern="*.txt")  
# remove duplicated files
testimonies <- testimonies[testimonies %in% c("2007_H_CC_Hegerl_SCI_P (2).txt", "2007_H_CC_Hegerl_SCI_O (2).txt") == FALSE]

# create a data frame of documents and text
raw_text <- tibble(testimonies) %>%
  mutate(text = map(testimonies, readr::read_file)) %>%
  unnest(cols = c(text))


raw_string <- raw_text

raw_string$text <- paste(raw_text$text,collapse = "\n")
####################Cleaning #######################
#remove special characters 
cleaned_text <- raw_text
cleaned_text$text  <- sub("\\([^()]*\\)", "", cleaned_text$text )
cleaned_text$text  <- gsub("(\\d-*\\.*\\d*)", "", cleaned_text$text )
cleaned_text$text  <- gsub("( \\s+|\\*+ )", "", cleaned_text$text )
cleaned_text$text  <- gsub("\r?\n|\r", " ", cleaned_text$text )
cleaned_text$text  <- gsub("[[:punct:]]", "", cleaned_text$text )
# Tokenization, lemmatization and  stop words removal
# Normalization using lemmatization 
spacy_initialize(model = "en_core_web_sm")
spacy_text = spacy_parse(structure(cleaned_text$text, names = cleaned_text$testimonies),lemma = TRUE, pos = TRUE, entity = TRUE)

#Remove english stop words 
spacy_text = unnest_tokens(spacy_text,word, lemma) %>%
  dplyr::anti_join(stop_words)

#word count
words_by_testimony<- spacy_text %>%
  count(doc_id, word, sort = TRUE) %>%
  ungroup()
#tf-idf
tf_idf <- words_by_testimony %>%
  bind_tf_idf(word, doc_id, n) %>%
  arrange(desc(tf_idf))
#plotting 2010 testimonies
tf_idf %>%
  filter(str_detect(doc_id, "^2010\\_")) %>%
  group_by(doc_id) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(tf_idf, word, fill =doc_id)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ doc_id, scales = "free") +
  labs(x = "tf-idf", y = NULL)
# calculating words correlation : which testimonies are similar to each other in text content
testemonies_cors <- words_by_testimony %>%
  pairwise_cor(doc_id, word, n, sort = TRUE)

# include only words that occur at least 50 times
spacy_text_50 <- spacy_text %>%
    group_by(word) %>%
  mutate(word_total = n()) %>%
  ungroup() %>%
  filter(word_total > 50)
# convert into a document-term matrix

df_dtm <- spacy_text_50 %>%
  unite(document, doc_id) %>%
  count(document, word) %>%
  cast_dtm(document, word, n)
library(topicmodels)
df_lda <- LDA(df_dtm, k = 5, control = list(seed = 2))
# plotting word count by cluster
df_lda %>%
  tidy() %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()+
theme(plot.title = element_text(size = 12), axis.text.x =element_text(size=8),axis.text.y =element_text(size=10), legend.title = element_text(size=9), legend.text = element_text(size=9), 
      panel.background = element_blank(), plot.background = element_blank())+
  labs(x = "Beta", y = "Topic terms", title="word by Topic")
# plotting document compistion by topic
df_lda %>%
  tidy(matrix = "gamma") %>%
  filter(str_detect(document, "^2010")) %>%
  mutate(doc_id = reorder(document, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ document) +
    theme(plot.title = element_text(size = 12), legend.title = element_text(size=9), legend.text = element_text(size=9), axis.line.y = element_line(),
        panel.background = element_blank(), plot.background = element_blank(), panel.grid.major=element_line(colour = 'gray', size = 0.1))+
  labs(x = "Topic",
       y = "Document topic decomposition")




spacy_finalize()

