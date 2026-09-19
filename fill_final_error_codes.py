import pandas as pd
import os
import shutil

INPUT = r".\outputs\error_analysis\manual_error_annotation_final.csv"
BACKUP = r".\outputs\error_analysis\manual_error_annotation_final_backup.csv"

# Backup first
if not os.path.exists(BACKUP):
    shutil.copy2(INPUT, BACKUP)

df = pd.read_csv(INPUT)

codes = {
"HingGPT": {
1:"E3,E4,E5,E6,E8,E10",2:"E1,E3,E4,E5,E6,E10",3:"E5,E6,E10",4:"E5,E6,E10",
5:"E5,E6,E10",6:"E1,E6,E10",7:"E1,E3,E4,E6,E8,E10",8:"E1,E3,E5,E6,E10",
9:"E3,E4,E6,E8,E10",10:"E1,E3,E4,E5,E6,E10",11:"E1,E3,E4,E5,E6,E10",
12:"E1,E3,E4,E5,E6,E7,E10",13:"E1,E3,E4,E5,E6,E10",14:"E1,E3,E4,E5,E6,E10",
15:"E1,E3,E4,E5,E6,E10",16:"E5,E6,E7,E10",17:"E1,E3,E4,E5,E6,E10",
18:"E1,E3,E4,E6,E10",19:"E1,E3,E4,E6,E8,E10",20:"E1,E3,E4,E5,E6,E10",
21:"E1,E3,E4,E5,E6,E10",22:"E1,E3,E4,E5,E6,E10",23:"E5,E6,E7,E10",
24:"E1,E3,E4,E5,E6,E8,E10",25:"E1,E3,E4,E6,E8,E10",
26:"E1,E3,E4,E5,E6,E8,E10",27:"E1,E3,E4,E5,E6,E10",28:"E1,E3,E4,E5,E6,E10",
29:"E1,E3,E4,E5,E6,E10",30:"E1,E3,E4,E5,E6,E10",31:"E1,E3,E4,E6,E10",
32:"E1,E3,E4,E6,E10",33:"E1,E3,E4,E5,E6,E10",34:"E1,E3,E4,E6,E10"
},

"Phi-3.5-mini": {
1:"E2,E3,E4,E6,E8,E10",2:"E2,E3,E4,E6,E7,E10",3:"E2,E3,E4,E6,E7,E10",
4:"E1,E3,E4,E6,E10",5:"E2,E3,E4,E6,E7,E10",6:"E1,E6,E10",
7:"E1,E3,E4,E6,E10",8:"E1,E3,E4,E6,E9,E10",9:"E1,E6,E10",
10:"E1,E3,E4,E6,E10",11:"E1,E3,E4,E6,E10",12:"E2,E3,E4,E6,E7,E10",
13:"E1,E3,E4,E6,E10",14:"E1,E3,E4,E6,E10",15:"E1,E6,E7,E10",
16:"E2,E3,E4,E6,E10",17:"E1,E6,E10",18:"E1,E3,E4,E6,E10",
19:"E1,E3,E4,E6,E10",20:"E2,E3,E4,E6,E10",21:"E1,E3,E4,E6,E10",
22:"E1,E3,E4,E6,E10",23:"E1,E6,E10",24:"E1,E6,E10",
25:"E1,E3,E4,E6,E10",26:"E1,E3,E4,E6,E10",27:"E1,E6,E10",
28:"E1,E3,E4,E6,E10",29:"E1,E3,E4,E6,E10",30:"E1,E6,E7,E10",
31:"E1,E3,E4,E6,E10",32:"E1,E3,E4,E6,E10",33:"E1,E6,E7,E10",
34:"E1,E3,E6"
},

"Qwen2.5-3B": {
1:"E1,E5,E6,E10",2:"E5",3:"E5,E7",4:"E7",5:"E2,E3,E4,E5,E6,E10",
6:"E1,E5,E6,E10",7:"E1,E6,E10",8:"E1,E5",9:"E1,E5,E6,E10",
10:"E5,E7,E6,E10",11:"E1,E6",12:"E1,E5,E6",13:"E5,E6,E10",
14:"E1,E6,E10",15:"E2,E3,E4,E6,E10",16:"E1,E6,E7,E10",
17:"E1,E3,E4,E6,E10",18:"E1,E6,E7,E10",19:"E1,E6,E10",
20:"E5,E7,E6,E10",21:"E2,E3,E4,E5,E6,E10",22:"E1,E6",
23:"E5,E7,E6,E10",24:"E1,E5,E6,E10",25:"E1,E6",26:"E1,E6,E10",
27:"E5,E7,E6,E10",28:"E1,E5,E6,E10",29:"E2,E3,E4,E5,E6,E7,E10",
30:"E2,E3,E4,E5,E6,E10",31:"E1,E5,E6,E9,E10",32:"E1,E6,E10",
33:"E1,E3,E4,E5",34:"E5,E7,E6,E10"
},

"Qwen2.5-7B": {
1:"E1,E3,E4,E5,E6,E10",2:"E5,E7,E6,E10",3:"E2,E3,E4,E5,E6,E10",
4:"E5,E7,E6,E10",5:"E2,E3,E4,E5,E6",6:"E1,E6,E10",
7:"E1,E6,E10",8:"E5,E7",9:"E1,E5,E6,E10",10:"E3,E4,E6",
11:"E5,E7,E6,E10",12:"E1,E5,E6,E10",13:"E5,E7,E6,E10",
14:"E1,E6,E10",15:"E1,E6,E10",16:"E1,E3,E5,E6,E7,E10",
17:"E1,E6,E10",18:"E2,E3,E4,E5,E6,E10",19:"E5,E7,E6,E10",
20:"E1,E5,E7,E6,E10",21:"E1,E6,E10",22:"E1,E3,E6,E7",
23:"E1,E6",24:"E1,E6,E10",25:"E7",26:"E2,E3,E4,E5,E6,E10",
27:"E2,E3,E4,E5,E6,E10",28:"E1,E5,E6,E10",29:"E1,E3,E4,E5,E6,E10",
30:"E1,E3,E4,E6,E7",31:"E2,E3,E4,E6,E10",32:"E1,E3,E4,E6,E10",
33:"E5,E7,E6,E10",34:"E1,E5,E6,E7"
}
}

# Fill final categories and indicator columns
for i, row in df.iterrows():
    model = row["model"]
    sample = int(row["sample"])

    final_code = codes[model][sample]
    df.at[i, "Final_Error_Category"] = final_code

    error_codes = set(final_code.split(","))

    for n in range(1, 11):
        df.at[i, f"E{n}_" + {
            1:"English_dominant",
            2:"Hindi_dominant",
            3:"Unnatural_code_switching",
            4:"Grammatical_error",
            5:"Repetition",
            6:"Prompt_misunderstanding",
            7:"Incomplete_response",
            8:"Spelling_transliteration_error",
            9:"Hallucination_factual_error",
            10:"Irrelevant_response"
        }[n]] = 1 if f"E{n}" in error_codes else 0

df.to_csv(INPUT, index=False)

print("DONE ✅")
print("Total responses:", len(df))
print("Output:", INPUT)
print("\nError category counts:")
print(df["Final_Error_Category"].value_counts())
print("\nReview status:")
print(df["Review_Status"].value_counts())