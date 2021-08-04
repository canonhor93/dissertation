import datetime
import re
from nltk.stem import WordNetLemmatizer


def process_date(date):
    convert_month = date
    convert_symbol = convert_month.replace("/", " ")
    convert_symbol = convert_symbol.replace(".", " ")
    convert_symbol = convert_symbol.replace(",", " ")
    convert_symbol = convert_symbol.replace("-", " ")
    convert_symbol = convert_symbol.split(" ")
    date_preprocessed = ''
    i = 0
    for date1 in convert_symbol:
        if i == 0:
            if len(date1) < 2:
                date_preprocessed = date_preprocessed + "0" + date1 + "-"
            else:
                date_preprocessed = date_preprocessed + date1 + "-"
        if i == 1:
            if "JAN" in date1:
                date1 = "01"
            if "FEB" in date1:
                date1 = "02"
            if "MAR" in date1:
                date1 = "03"
            if "APR" in date1:
                date1 = "04"
            if "MAY" in date1:
                date1 = "05"
            if "JUN" in date1:
                date1 = "06"
            if "JUL" in date1:
                date1 = "07"
            if "AUG" in date1:
                date1 = "08"
            if "SEP" in date1:
                date1 = "09"
            if "OCT" in date1:
                date1 = "10"
            if "NOV" in date1:
                date1 = "11"
            if "DEC" in date1:
                date1 = "12"
            if len(date1) < 2:
                date_preprocessed = date_preprocessed + "0" + date1 + "-"
            else:
                date_preprocessed = date_preprocessed + date1 + "-"
        if i == 2:
            if len(date1) == 4:
                date_preprocessed = date_preprocessed + date1
            else:
                date_preprocessed = date_preprocessed + "20" + date1
        i += 1
    return date_preprocessed


def process_total(total):
    try:
        total_preprocessed = re.search('\d+(\.\d{2})', total).group()
    except:
        total_preprocessed = "0.00"
    return total_preprocessed


def process_keyword(keyword):
    stemmer = WordNetLemmatizer()
    documents = []
    # Remove all the number
    document = re.sub(r'\d', ' ', keyword)
    # Remove all unwanted word
    document = document.replace(' RM ', " ")
    document = document.replace(' RM', " ")
    document = document.replace('RM ', " ")
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)
    return document
