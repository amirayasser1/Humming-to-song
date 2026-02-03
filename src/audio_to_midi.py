import os
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH


INPUT_FOLDER = "mp3songs"
OUTPUT_FOLDER = "midi_output"


def convert_folder():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    audio_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".mp3", ".wav"))
    ]

    if not audio_files:
        print(" No audio files found in mp3songs/")
        return

    print(f"\n Found {len(audio_files)} songs\n")

    for i, file in enumerate(audio_files, 1):

        audio_path = os.path.join(INPUT_FOLDER, file)
        midi_name = os.path.splitext(file)[0] + ".mid"
        midi_path = os.path.join(OUTPUT_FOLDER, midi_name)

        
        if os.path.exists(midi_path):
            print(f"⏭Skipping ({i}/{len(audio_files)}): {file}")
            continue

        try:
            print(f"Converting ({i}/{len(audio_files)}): {file}")

            predict_and_save(
                audio_path_list=[audio_path],
                output_directory=OUTPUT_FOLDER,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False
            )

            print(f" Done: {file}\n")

        except Exception as e:
            print(f" Failed: {file}")
            print(e)


if __name__ == "__main__":
    convert_folder()
