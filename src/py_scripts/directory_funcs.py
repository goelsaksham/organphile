import os
import glob


def exist_directory(directory_path: str):
    return os.path.isdir(directory_path)


def exist_file(file_path: str):
    return os.path.isfile(file_path)


def check_output_directory(directory_path: str):
    if not exist_directory(directory_path):
        try:
            os.makedirs(directory_path)
            return True
        except FileNotFoundError:
            return False
        except PermissionError:
            return False
    else:
        return True


def get_directory_contents(directory_path: str, pattern: str):
    if exist_directory(directory_path):
        # Get the current working directory
        cwd = os.getcwd()
        # Change the directory into the target directory path
        os.chdir(directory_path)
        # Get the list of directory contents
        directory_contents = glob.glob(pattern)
        # Change back to the original process working directory
        os.chdir(cwd)
        # Return back the directory contents
        return directory_contents
    else:
        return []


def get_subdirectory_names(directory_path: str):
    return list(map(lambda subdirectory_name: subdirectory_name.replace('\\', '').replace('//', ''),
                    get_directory_contents(directory_path, '*//')))


def get_file_names(directory_path: str, file_extension: str):
    return get_directory_contents(directory_path, f'*{file_extension}')


def construct_path(directory_path: str, directory_content_name: str):
    return os.path.join(directory_path, directory_content_name)