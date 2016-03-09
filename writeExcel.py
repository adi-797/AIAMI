import xlsxwriter
from stats import run_stats_analysis

__author__ = 'andres'

means, stand, instrument_count = run_stats_analysis(5000, True)

workbook = xlsxwriter.Workbook('Analysis-NewTrumpet-2.xlsx')
worksheet = workbook.add_worksheet('stats')

row = 0
col = 0

for key in means.keys():
    worksheet.write(row, col, key)
    worksheet.write(row, col + 1, means[key])
    worksheet.write(row, col + 2, stand[key])
    worksheet.write(row, col + 3, instrument_count[key])
    row += 1
workbook.close()
