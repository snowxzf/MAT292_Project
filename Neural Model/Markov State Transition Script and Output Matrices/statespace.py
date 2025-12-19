#!/usr/bin/env python3

import pandas as pd
import numpy as np

fulldata = "/mnt/c/Users/jessi/Desktop/MAT_Project/LUMIERE_ExpertRating_Train.csv"
methylationinfo = pd.read_csv("/mnt/c/Users/jessi/Desktop/MAT_Project/LUMIERE_Demographics_Pathology_Train.csv")

overall = pd.read_csv(fulldata)
rating_col = "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)"
overall["Rating"] = overall[rating_col].fillna("")
overall["Rating"] = overall["Rating"].apply(lambda e: "OP" if "Op" in str(e) else str(e))
overall = overall[overall["Rating"] != ""]
overall = overall[overall["Rating"].str.contains("None", na=False) == False]

patients_methylated = methylationinfo.loc[methylationinfo["MGMT qualitative"] == "methylated", "Patient"].values
patients_unmethylated = methylationinfo.loc[methylationinfo["MGMT qualitative"] == "not methylated", "Patient"].values

df_methylated = overall.loc[overall["Patient"].isin(patients_methylated)]
df_unmethylated = overall.loc[overall["Patient"].isin(patients_unmethylated)]

statelist = ['OP', 'CR', 'PR', 'SD', 'PD', 'Death']
deathobj = pd.Series({"Rating": "Death"})

transitions_methylated = pd.DataFrame(data=np.zeros((6, 6)), columns=statelist, index=statelist)
transitions_unmethylated = pd.DataFrame(data=np.zeros((6, 6)), columns=statelist, index=statelist)

# methylated
for patient in patients_methylated:
    currdf = df_methylated.loc[df_methylated["Patient"] == patient, "Rating"]
    currdf = pd.concat((currdf, deathobj), ignore_index=True)

    for idx in np.arange(1, currdf.shape[0]):
        transitions_methylated.at[currdf.values[idx - 1], currdf.values[idx]] += 1
        
print("Methylated Done")

transitions_methylated_norm = transitions_methylated.div(transitions_methylated.sum(axis=1), axis=0)

# not methylated
for patient in patients_unmethylated:
    currdf = df_unmethylated.loc[df_unmethylated["Patient"] == patient, "Rating"]
    currdf = pd.concat((currdf, deathobj), ignore_index=True)

    for idx in np.arange(1, currdf.shape[0]):
        transitions_unmethylated.at[currdf.values[idx - 1], currdf.values[idx]] += 1
        
print("Unmethylated Done")

transitions_unmethylated_norm = transitions_unmethylated.div(transitions_unmethylated.sum(axis=1), axis=0)
transitions_unmethylated_norm.to_csv("/mnt/c/Users/jessi/Desktop/MAT_Project/statespace/unmethylated_train.csv")
transitions_methylated_norm.to_csv("/mnt/c/Users/jessi/Desktop/MAT_Project/statespace/methylated_train.csv")