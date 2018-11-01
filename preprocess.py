import glob
import os


def preprocess():
    """
    This is a mock example how you could batch process dataset.
    """

    # First, let's find our files we want to preprocess.
    # Valohai stores all specified input files at '/valohai/inputs/<INPUT_NAME>'
    # but you are also free to download datasets on your own.
    inputs_dir = os.path.realpath(os.getenv('VH_INPUTS_DIR', './inputs'))
    dataset_dir = os.path.realpath(os.path.join(inputs_dir, 'my-dataset'))
    file_paths = glob.glob(f'{dataset_dir}/*')
    if not file_paths:
        print(f'Could not find dataset files in {dataset_dir}')
        return

    # Find out where to store the processed files.
    # Valohai platform will maintain anything written to '/valohai/outputs'.
    outputs_dir = os.path.realpath(os.getenv('VH_OUTPUTS_DIR', './outputs'))

    for file_path in file_paths:
        size_in_bytes = os.path.getsize(file_path)
        size_in_megabytes = round(size_in_bytes / 1000000, 1)
        file_name = os.path.basename(file_path)
        print(f'Processing {file_name} ({size_in_megabytes}MB)...')

        # Here you could actually process the files and save the results
        # to the output directory to chain further executions.
        # This example just writes a bunch of zeroes to the file.
        output_path = os.path.join(outputs_dir, f'{file_name}.processed')
        with open(output_path, 'wb') as f:
            f.seek(1234567 - 1)  # 1.234567MB
            f.write(b'\0')
            f.close()

    print('Done!')


if __name__ == '__main__':
    preprocess()
