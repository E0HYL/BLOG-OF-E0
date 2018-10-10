#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import os

with open("vocab.json", 'rb+') as filehandle:
    filehandle.seek(-1, os.SEEK_END)
    filehandle.truncate()

while True:
    vocabulary = raw_input("vocabulary:")
    if vocabulary:
        soundmark = raw_input("soundmark:")
        explaination = raw_input("explaination:")
        instance = raw_input("instance:")
        dicitem = {
            "vocabulary": vocabulary,
            "soundmark": soundmark,
            "explaination": explaination,
            "instance": instance
        }
        with open("vocab.json", "a") as f:
            f.write(",")
            json.dump(dicitem, f)
        print '----------------------------------'
    else:
        break

with open("vocab.json", "a") as f:
    f.write("]")
