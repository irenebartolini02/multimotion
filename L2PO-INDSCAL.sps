* Encoding: UTF-8.
BEGIN PROGRAM Python.

import spss, spssaux
import os
import re

# Directory
folder = r"C:\Users\ASUS\Desktop\Multimotion_application\survey_data\SurveyDataJuly24\exp-2025-11-02-with-lopo"

# List xlsx files
files = [f for f in os.listdir(folder) if f.startswith("distance_matrix_no_") and f.endswith(".xlsx")]

print(files)

for file in files:
    full_path = os.path.join(folder, file)
    match = re.search(r'distance_matrix_no_([A-Za-z0-9]+)_no_([A-Za-z0-9]+)', file)
    
    # 2. Check if a match was found (prevents errors)
    if match:
        # group(1) is the first code (e.g., '5KB3V')
        code1 = match.group(1) 
        # group(2) is the second code (e.g., 'F5tXL')
        code2 = match.group(2)
        
        # 3. Combine the codes with an underscore
        combined_code = "{}_{}".format(code1, code2)
        
        # 4. Create the new output file path
        output_file = os.path.join(folder, "l2po", "{}.txt".format(combined_code))
        
        # Now you can use 'output_file', for example:
        print("Processing {}".format(file))

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

