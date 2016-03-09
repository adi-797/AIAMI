import os
import json


def write_json(path, file_name, values):
    """
    This function writes Json files for the calculated mean descriptors.
    """

    start = os.getcwd()  # the working directory
    os.chdir(os.getcwd() + '/' + path)
    keys = values.keys()

    output_name = '.'.join(file_name.split('.')[:-1]) + '.json'  # same name as the old file
    with open(output_name, 'w') as outfile:
        json.dump([keys, values], outfile, sort_keys=True, indent=4,
                  ensure_ascii=False)  # This line actually writes the files
    outfile.close()
    os.chdir(start)

    return
