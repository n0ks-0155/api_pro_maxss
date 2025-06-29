# app/utils.py
import re

def extract_objects(text: str):
    # Разделяем по союзам и знакам препинания
    parts = re.split(r'[и,.\-]', text.lower())
    objects = [p.strip() for p in parts if len(p.strip()) > 3]
    return objects
