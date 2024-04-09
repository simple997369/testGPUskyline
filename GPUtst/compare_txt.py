#!/usr/bin/python3
import filecmp
import os, sys
sys.path.append(os.path.abspath(os.pardir))
here = os.path.dirname(os.path.abspath(__file__))

# 檢查 file1.txt 與 file2.txt 是否相同
if filecmp.cmp(here + "/demo.txt", here + "/demo2.txt"):
    print("檔案相同")
else:
    print("檔案不同")

