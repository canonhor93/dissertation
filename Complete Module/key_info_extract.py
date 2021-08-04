import os
import re
from data_processing import process_date, process_total, process_keyword

def key_info_extract(x):
    x = x.splitlines()

    # Remove Space from List
    x = [x.strip(' ') for x in x]
    x = [x.upper() for x in x]
    print(x)

    # Remove empty List from List using filter
    res = list(filter(None, x))

    # Date Regular Expression
    date_re1 = r"(([0-2][0-9])|([3][0-1]))/(JAN.*|FEB.*|MAR.*|APR.*|MAY.*|JUN.*|JUL.*|AUG.*|SEP.*|OCT.*|NOV.*|DEC.*|([0-1][0-9]))/((20[0-2][0-9])|([0-2][0-9]))"
    date_re2 = r"(([0-2][0-9])|([3][0-1]))-(JAN.*|FEB.*|MAR.*|APR.*|MAY.*|JUN.*|JUL.*|AUG.*|SEP.*|OCT.*|NOV.*|DEC.*|([0-1][0-9]))-((20[0-2][0-9])|([0-2][0-9]))"
    date_re3 = r"(([0-2][0-9])|([3][0-1]))\.(JAN.*|FEB.*|MAR.*|APR.*|MAY.*|JUN.*|JUL.*|AUG.*|SEP.*|OCT.*|NOV.*|DEC.*|([0-1][0-9]))\.((20[0-2][0-9])|([0-2][0-9]))"
    date_re4 = r"(([0-2][0-9])|([3][0-1])) (JAN.*|FEB.*|MAR.*|APR.*|MAY.*|JUN.*|JUL.*|AUG.*|SEP.*|OCT.*|NOV.*|DEC.*|([0-1][0-9])) ((20[0-2][0-9])|([0-2][0-9]))"

    # Price Regular Expression
    price_re = r"(\d{1,3},?)*\d{1,3}\.\d{2}"
    # Total Price Pattern
    price_keyword_1 = r"(TOTAL|NETT)"
    price_keyword_2 = r"(CURRENT|CUMENE|CARRERE|JUMIAH|JUMIAB|JUMALAH|JUMLAH)"
    price_keyword_0 = r"(CHARGE|ANOUNT|ROUNDING|SEVICE|SERVICE|CHANGE|CASH|MYDEBIT|SAVINGS|ROUNDING|DISCOUNT|CASH|PREVIOUS|BALANCE|REDEEM|" \
                      r"DISCOUNIS|TOTAT|ADJUSTMENTS)"

    # Keywords
    keyword = r"^(VEHICLE|VEHICLES|CAR|CARS|GRAB|FOOD)$"

    issuer = ' '
    total_price = ''
    issue_date_list = []
    price_list = []
    keyword_list = []
    issuer_checklist = r"(PAGE|INVOICE|PM|BILL)"
    for i in res:
        # Get Issuer
        if not re.search(issuer_checklist, i) and issuer == " " and len(i)>3:
            issuer = i
        match = False
        # Get Date
        if re.search(date_re1, i):
            match = re.search(date_re1, i)
            issue_date_list.append(match.group(0))
        if not match:
            if re.search(date_re2, i):
                match = re.search(date_re2, i)
                issue_date_list.append(match.group(0))
        if not match:
            if re.search(date_re3, i):
                match = re.search(date_re3, i)
                issue_date_list.append(match.group(0))
        if not match:
            if re.search(date_re4, i):
                match = re.search(date_re4, i)
                issue_date_list.append(match.group(0))
        # Get Price
        if re.search(price_re, i):
            price_list.append(i)
        # Get Keyword
        if re.search(keyword, i):
            keyword_list.append(i)

    # Set Undefined if no date found
    if len(issue_date_list) == 0:
        issue_date_list.append('undefined')

    # Identify Total Price in price list
    for i in price_list:
        match = re.search(price_keyword_1, i)
        if match:
            total_price = i
    if total_price == "":
        for i in price_list:
            match = re.search(price_keyword_2, i)
            if match:
                total_price = i
    # Get Highest Price as Total if pattern not match
    if total_price == "":
        price = 0.00
        for i in price_list:
            total_price1 = process_total(i)
            if float(total_price1) > float(price):
                price = total_price1
            break
        total_price = price

    # Clean Price List
    item_to_remove = []
    item_list = price_list
    for i in item_list:
        match = re.search(price_keyword_1, i)
        if match:
            item_to_remove.append(i)
        match = re.search(price_keyword_2, i)
        if match:
            item_to_remove.append(i)
        match = re.search(price_keyword_0, i)
        if match:
            item_to_remove.append(i)

    for i in item_to_remove:
        try:
            item_list.remove(i)
        except:
            print("Data has been removed")

    # Standardized data before post to CSV
    issuer = re.sub('[^a-zA-Z0-9 \n\.]', '', issuer)
    issue_date = process_date(issue_date_list[0])
    total_price1 = process_total(total_price)
    keyword_list.append(issuer)
    for i in item_list:
        keyword_list.append(i)
    keyword_raw = ""
    for i in keyword_list:
        keyword_raw = keyword_raw + " " +i
    keyword_raw = process_keyword(keyword_raw)
    if issuer == "":
        issuer = " "
    if keyword_raw == "":
        keyword_raw = " "
    return issuer, issue_date, price_list, item_list, keyword_raw, total_price1

