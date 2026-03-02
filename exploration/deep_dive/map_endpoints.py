from pathlib import Path
import re


OPS_FILE = Path('.venv/lib/python3.13/site-packages/azure/ai/projects/operations/_operations.py')


def main() -> None:
    text = OPS_FILE.read_text(encoding='utf-8')
    paths = sorted(set(re.findall(r'_url\s*=\s*"(/[^\"]+)"', text)))
    print('Foundry SDK generated endpoint paths:')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
