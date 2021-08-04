# dissertation
MASTER OF COMPUTER SCIENCE (APPLIED COMPUTING)

This is a Master Dissertation done by Hor Zhang Neng from University Malaya.

# Introduction of Program

1. main_program
    - Main program to execute Pre-Receipt Parsing, Receipt Parsing and Post-Receipt Parsing.
2. convert_jpeg
    - Standarized test receipt to Jpeg format
3. image_preprocess
    - Image preprocessing to increase performance of OCR
4. image_ocr
    - Recognize text from receipt
5. key_info_extract
    - Extract key information from receipt
6. data_processing
    - Preprocess data before write to file
7. receipt_classification
    - Categorized receipt into category

# Introduction of Dataset

1. raw
    - Folder which included 100 raw receipts 
2. proceeded
    - Folder which included result of program executed
3. Expected Result
    - Folder which included actual text in the receipt
4. stopwords
    - Stopwords to remove before receipt classification, which included common unuseful words in receipt, to increase accuracy of classification
5. training
    - Training dataset to train classifier, which is a dictionary to store common keyword in the receipt to identify characteristic of receipt
