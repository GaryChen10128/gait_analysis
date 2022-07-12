# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:29:27 2022

@author: 11005080
"""

#encoding=utf-8
import jieba

sentence = "1)	一、病史：病人無手術、輸血經驗。二、評估：病人於2021/04/28 15：08經由門診步行入院。主要因【今年1月因左側左乳房腫脹疼痛，吃藥無效，門診醫師建議入院行清創手術。】，醫師診斷為【乳腺炎】，收【65】-【101】入院治療，4/28：SR、CXR：1.雙LUNG MARKING上升2.T-L SPINE DJD，4/28採under IVG行Debridement (5-10cm)，4/28收AER & ANAER CULTURE：normal，住院過程中依病人病情狀況予建立急性疼痛(通用)、皮膚完整性受損/通用之照顧計劃。"
print("Input：", sentence)
words = jieba.cut(sentence, cut_all=False)
print("Output 精確模式 Full Mode：")
out=[]
for word in words:
    out.append(word)
print(out)

# sentence = "独立音乐需要大家一起来推广，欢迎加入我们的行列！"
# print("Input：", sentence)
# words = jieba.cut(sentence, cut_all=False)
# print("Output 精確模式 Full Mode：")
# for word in words:
#     print(word)