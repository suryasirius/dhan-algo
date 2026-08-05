"""
Module: oi_utils.py
Description: Utility functions, time zone managers, and mathematical helpers for OI engine.
"""

from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

def get_current_ist_time():
    return datetime.now(IST)

def safe_divide(numerator, denominator, fallback=0.0):
    try:
        if denominator == 0 or denominator is None:
            return fallback
        return float(numerator) / float(denominator)
    except ZeroDivisionError:
        return fallback
