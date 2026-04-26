* Encoding: UTF-8.
BEGIN PROGRAM Python.

import spss, spssaux
import os
import re

# Directory
folder = r"C:\Users\ASUS\Desktop\Multimotion_application\survey_data\SurveyDataJuly24\exp-2025-11-02-with-lopo"

# List xlsx files
files = [f for f in os.listdir(folder) if f.startswith("distance_matrix_no_") and f.endswith(".xlsx")]

for file in files:
    full_path = os.path.join(folder, file)
    code = re.search(r'distance_matrix_no_([A-Za-z0-9]+)', file).group(1)
    output_file = os.path.join(folder, "lopo", "{}.txt".format(code))

    # Import file
    spss.Submit("""
GET DATA
  /TYPE=XLSX
  /FILE="{}"
  /SHEET=name 'Combined_Sheet'
  /CELLRANGE=full
  /READNAMES=on.
EXECUTE.
""".format(full_path))

 # Redirect OMS output in a TXT file
    spss.Submit("""
OMS
  /DESTINATION FORMAT=TEXT OUTFILE='{}'.
""".format(output_file))

    # INDSCAL
    spss.Submit("""
ALSCAL
  /VARIABLES=HN_1	HN_2_H	HN_2_L	HN_3_H	HN_3_L	HN_4	HN_5	HN_6	HN_7	HN_8	HP_1_H	HP_1_L	HP_2	HP_3_H	HP_3_L	HP_4	HP_5	HP_6	HP_7_H	HP_7_L	HP_8	LN_1	LN_2	LN_3	LN_4	LN_5	LN_6	LN_7_N	LN_7_P	LN_8	LP_1	LP_2	LP_3	LP_4	LP_5	LP_6	LP_7	LP_8
  /SHAPE=SYMMETRIC
  /LEVEL=INTERVAL
  /CONDITION=MATRIX
  /MODEL=INDSCAL
  /CRITERIA=CONVERGE(.0000001) STRESSMIN(.0000001) ITER(1000) CUTOFF(0) DIMENS(2)
  /PLOT=default all
  /PRINT=HEADER.
""")

    # Close OMS redirect
    spss.Submit("OMSEND.")

END PROGRAM.
