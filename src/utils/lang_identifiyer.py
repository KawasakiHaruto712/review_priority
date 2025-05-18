import os

def identify_lang_from_file(file_path: str):
    extension = os.path.splitext(file_path)[1][1:]
    lang_map = {
        "ts": "JavaScript",
        "js": "JavaScript",
        "py": "Python",
        "java": "Java",
        "c": "CPP",
        "php": "PHP",
        "dart": "Dart",
        "r": "R",
    }

    if extension not in lang_map:
        raise ValueError(f"Unsupported file extension: {extension}")

    return lang_map[extension]