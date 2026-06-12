import pandas as pd
import pathlib
PATH = "../../data/metadata/recordings_binary_clean.csv"
NEW_PATH = "../../data/metadata/svd_cleaned_metadata.csv"
VOVEL_VARIANTS = ['a_n','a_l','a_h','a_lhl','i_n','i_l','i_h','i_lhl','u_n','u_l','u_h','u_lhl',]

if __name__ == "__main__":
    full_data = pd.read_csv(PATH)
    results = []
    for row in full_data.iterrows():
        recording_id = row[1]["AufnahmeID"]
        speaker_id = row[1]["SprecherID"]
        status = row[1]["label"]
        for variant in VOVEL_VARIANTS:
            if not status:
                file_path = pathlib.Path(f"../../data/wav/healthy/{recording_id}/{recording_id}-{variant}.wav")
                if not file_path.exists():
                    continue
                d = {"Id": speaker_id, "Status": status, "FilePath" : f"healthy/{recording_id}/{recording_id}-{variant}.wav","RecordingID": recording_id}
                results.append(d)
            else:
                file_path = pathlib.Path(f"../../data/wav/pathological/{recording_id}/{recording_id}-{variant}.wav")
                if not file_path.exists():
                    continue
                d = {"Id": speaker_id, "Status": status, "FilePath": f"pathological/{recording_id}/{recording_id}-{variant}.wav","RecordingID": recording_id}
                results.append(d)
    metadata = pd.DataFrame(results)
    metadata.to_csv(NEW_PATH, index=False)

