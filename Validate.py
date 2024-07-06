import configparser

import pymysql
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import datetime


def validate_lottery_numbers(prediction):
    # 检查主号码和bonus是否有重复
    main_numbers = prediction[:6]
    bonus_number = prediction[6]
    powerball_number = prediction[7]

    if len(set(main_numbers + [bonus_number])) != len(main_numbers) + 1:
        return False, "Duplicate numbers found among main numbers and bonus."

    # 检查主号码和bonus是否在有效范围内
    for num in main_numbers:
        if not (1 <= num <= 40):
            return False, "Main numbers should be between 1 and 40."

    if not (1 <= bonus_number <= 40):
        return False, "Bonus number should be between 1 and 40."

    if not (1 <= powerball_number <= 10):
        return False, "Powerball number should be between 1 and 10."

    # 检查是否存在连续四个或五个整数
    main_numbers_sorted = sorted(main_numbers)
    for i in range(len(main_numbers_sorted) - 4):
        if main_numbers_sorted[i+4] - main_numbers_sorted[i] == 4:
            return False, "Avoiding consecutive number patterns of 5 integers."
    for i in range(len(main_numbers_sorted) - 3):
        if main_numbers_sorted[i+3] - main_numbers_sorted[i] == 3:
            return False, "Avoiding consecutive number patterns of 4 integers."

    return True, "Validation successful."

# Example usage:
predictions = [[1, 20, 23, 24, 25, 26, 31, 10],
               [10, 12, 13, 14, 25, 26, 17, 8],
               [1, 12, 15, 17, 25, 26, 27, 8]]
for prediction in predictions:
    valid, message = validate_lottery_numbers(prediction)
    print(valid, message)
