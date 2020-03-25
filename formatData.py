# -*- encoding= utf-8 -*-
"""
将评分转换为评价值
100分为满分，记为0
90-99记为1
80-89记为2
70-79记为3
60-69记为4
59及以下记为5
"""
def f():
    with open('scores.txt','r',encoding='utf-8')as sc, open('format_scores.txt','w',encoding='utf-8')as scw:
        outscore=100
        scw.truncate()
        for lines in sc.readlines():
            score = int(lines)
            if score == 100:
                outscore = 0
            elif score >= 90:
                outscore = 1
            elif score >= 80:
                outscore = 2
            elif score >= 70:
                outscore = 3
            elif score >=60:
                outscore = 4
            else:
                outscore = 5
            scw.write(str(outscore))
            scw.write('\n')