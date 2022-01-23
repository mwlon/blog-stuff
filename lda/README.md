AUTHOR: Martin Loncaric
---------------------

HOW TO USE:
-----------
  *  change get_fnames() to find the documents you want to run on
    * if these are not plain text files, change read_doc(fname)
  * /usr/share/dict/words is standard for the location of a Unix system's
        english dictionary file, but if your dictionary has a different location
        or format, you will need to change that global variable
  * in main(), set iters and n_topics to your preferences

ADDITIONAL OPTIONS:
-------------------
  *  adjust the number of topics and iterations
  *  if you want to ignore additional words or punctuation, change stopwords.py
  *  increase/decrease alpha if you think your typical document has more/fewer
        topics represented
  *  increase/decrease beta if you think your typical topic has more/less
        diversity in vocabulary
  *  change display_topics and the final portion of get_topics to give return
        the topic information as you would like it processed/output# SimpleLDA
