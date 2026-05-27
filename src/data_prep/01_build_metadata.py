import pandas as pd
import pathlib
PATH = "../../data/metadata/stimmdb_full.csv"
NEW_PATH = "../../data/metadata/svd_cleaned_metadata.csv"
'''

'''
if __name__ == "__main__":
    full_data = pd.read_csv(PATH)
    results = []
    for row in full_data.iterrows():
        id = row[1]["Speaker ID"]
        file_path = f"{id}_{id}-a_n.wav"
        if not pathlib.Path("../../data/raw/"+file_path).exists():
            continue
        status = 0 if row[1]["Recording Type"]=="So-called „Normal” Voices" else 1
        d = {"Id" : id, "Status" : status, "FilePath" : file_path}
        results.append(d)
    metadata = pd.DataFrame(results)
    metadata.to_csv(NEW_PATH, index=False)

