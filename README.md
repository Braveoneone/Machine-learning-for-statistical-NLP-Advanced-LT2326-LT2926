# Machine-learning-for-statistical-NLP-Advanced-LT2326-LT2926
Machine learning for statistical NLP: Advanced LT2326/LT2926

## Just run:
python3 trecTest.py

It's using the trec06p dataset right now.
For change the dataset, in the code, just cancel the comment #:
'''
Change to uci SMS Spam Collection
'''
total_email = pd.read_table('./uci_spam/SMSSpamCollection.txt', sep='\t', names=['label', 'mem'])


# Dataset source website:
## trec06p: https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06
DOWNLOAD
The TREC 2006 Public Corpus -- 75MB (trec06p.tgz) [select save as; do not copy link]

## sms Spam: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
