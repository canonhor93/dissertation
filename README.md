# Understanding Scanned Receipts with Optical Character Recognition and Machine Learning Techniques for Automated Reimbursement Management
MASTER OF COMPUTER SCIENCE (APPLIED COMPUTING) - UNIVERSITY MALAYA <br/>
HOR ZHANG NENG

# Introduction of Program

1. main_program
    - Main program to execute image pre-processing, text extraction, key information extraction and receipt classification.
2. convert_jpeg
    - Standardized test receipt to Jpeg format
3. image_preprocess
    - Image preprocessing to increase performance of OCR
4. image_ocr
    - Recognize text from receipt
5. write_csv
    - Write information into csv file
6. key_info_extract
    - Extract key information from receipt
7. data_processing
    - Preprocess data before write to file
8. receipt_classification
    - Categorized receipt into category

# Introduction of Dataset

1. raw
    - Folder which included 100 raw receipts 
    - The origin of receipt is determined by file name which started with:
        - I - ICDAR Database
        - P - PDF Receipt
        - R - Physical Receipt by Daily Collection
        - S - Screenshot Receipt
    - The category of receipt is determined by file name which ended with:
        - A - Accommodation
        - F - Fares (transport)
        - G - Groceries
        - M - Meals
        - P - ICDAR Database
        - T - Telecommunication
2. proceeded
    - Folder which included result of program executed
3. Expected Result
    - Folder which included actual text in the receipt
4. stopwords
    - Stopwords to remove before receipt classification, which included common unuseful words in receipt, to increase accuracy of classification
5. training
    - Training dataset to train classifier, which is a dictionary to store common keyword in the receipt to identify characteristic of receipt
