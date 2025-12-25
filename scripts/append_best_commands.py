#!/usr/bin/env python3
from pathlib import Path

def find_best_command_files(root: Path):
    files = []
    for d in sorted(root.glob('jobs-feature-noise-*')):
        for p in sorted(d.rglob('best_command.txt')):
            files.append(p)
    return files

def extract_python_line(path: Path):
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if s and not s.startswith('#') and s.startswith('python'):
                return s
    except Exception:
        return None
    return None

def main():
    repo_root = Path(__file__).resolve().parents[1]
    lr_root = repo_root
    out_path = lr_root / 'main-batch.sh'

    files = find_best_command_files(lr_root)
    commands = []
    seen = set()
    for f in files:
        cmd = extract_python_line(f)
        if cmd and cmd not in seen:
            seen.add(cmd)
            commands.append(cmd)

    header = '# Auto-generated list of best commands from jobs-feature-noise-*\n'

    with out_path.open('w', encoding='utf-8') as out:
        out.write(header)
        out.write('\n')
        for c in commands:
            out.write(c + '\n')

    print(f"Wrote {len(commands)} commands to {out_path}")

if __name__ == '__main__':
    main()
