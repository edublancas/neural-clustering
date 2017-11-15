import subprocess


def get_commit_hash():
    out = subprocess.check_output('git show --oneline -s', shell=True)
    return out.decode('utf-8') .replace('\n', '')
