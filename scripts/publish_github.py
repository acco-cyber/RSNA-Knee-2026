"""Publish code-only files through GitHub's Git Data API without rewriting history.

Usage: GITHUB_TOKEN=... python scripts/publish_github.py [--dry-run]
Credentials are read from the environment and never persisted in repository files.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'acco-cyber/RSNA-Knee-2026'
EXTENSIONS = {'.ipynb', '.py', '.md', '.json', '.csv', '.txt', '.yml', '.yaml'}
SECRET_PATTERNS = [
    re.compile(rb'ghp_[A-Za-z0-9]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(rb'KGAT_[A-Za-z0-9]{20,}'),
    re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
]


def api(method, path, token, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request('https://api.github.com/repos/' + REPOSITORY + path,
        method=method, data=data, headers={'Authorization': 'Bearer ' + token,
        'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json', 'User-Agent': 'RSNA-Knee-2026-publisher'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'GitHub {method} {path}: HTTP {error.code}; ' + error.read(400).decode(errors='replace')) from None
        except urllib.error.URLError as error:
            if attempt == 2:
                raise RuntimeError(f'GitHub connection failed for {method} {path}: {error.reason}') from None
            print(json.dumps({'connection_retry': attempt+1, 'operation': method, 'path': path}), flush=True)
            time.sleep(2 ** (attempt+1))


def file_selection():
    selected = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.as_posix() != '.gitignore' and any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
            continue
        if path.suffix not in EXTENSIONS and relative.as_posix() != '.gitignore':
            continue
        if any(part in {'private_data', 'report_batches', 'report_annotations', 'l2_dataset'} for part in relative.parts):
            raise ValueError(f'Private data must not be published: {relative}')
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise ValueError(f'Credential pattern found in {relative}; publication stopped')
        selected.append((relative.as_posix(), content))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--inline-trees', action='store_true', help='Upload UTF-8 files through sequential, bounded tree-content batches')
    parser.add_argument('--message', default='Recover RSNA history and add audited L2 training and submission pipeline')
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError('GITHUB_TOKEN is required')
    repository = api('GET', '', token)
    branch = repository['default_branch']
    reference = api('GET', '/git/ref/heads/' + branch, token)
    previous = reference['object']['sha']
    commit = api('GET', '/git/commits/' + previous, token)
    tree = api('GET', '/git/trees/' + commit['tree']['sha'] + '?recursive=1', token)
    known = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    changed = []
    for relative, content in file_selection():
        git_sha = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if known.get(relative) != git_sha:
            changed.append((relative, content, git_sha))
    print(json.dumps({'branch': branch, 'previous_commit': previous, 'files_changed': len(changed),
                      'notebooks_changed': sum(relative.endswith('.ipynb') for relative, _, _ in changed),
                      'bytes_changed': sum(len(content) for _, content, _ in changed), 'dry_run': args.dry_run}), flush=True)
    if args.dry_run or not changed:
        return
    entries = []
    if args.inline_trees:
        current_tree = commit['tree']['sha']
        batch = []
        batch_bytes = 0
        uploaded_files = 0
        for relative, content, expected in changed:
            entry = {'path': relative, 'mode': '100644', 'type': 'blob', 'content': content.decode('utf-8')}
            size = len(json.dumps(entry).encode())
            if batch and batch_bytes + size > 2 * 1024 * 1024:
                current_tree = api('POST', '/git/trees', token, {'base_tree': current_tree, 'tree': batch})['sha']
                uploaded_files += len(batch)
                print(json.dumps({'verified_tree_files': uploaded_files, 'total': len(changed)}), flush=True)
                batch, batch_bytes = [], 0
            batch.append(entry)
            batch_bytes += size
            entries.append({'path': relative, 'mode': '100644', 'type': 'blob', 'sha': expected})
        if batch:
            current_tree = api('POST', '/git/trees', token, {'base_tree': current_tree, 'tree': batch})['sha']
            print(json.dumps({'verified_tree_files': uploaded_files + len(batch), 'total': len(changed)}), flush=True)
        verified_tree = api('GET', '/git/trees/' + current_tree + '?recursive=1', token)
        actual = {item['path']: item['sha'] for item in verified_tree['tree'] if item['type'] == 'blob'}
        for entry in entries:
            if actual.get(entry['path']) != entry['sha']:
                raise RuntimeError('Uploaded tree integrity mismatch: ' + entry['path'])
        new_commit = api('POST', '/git/commits', token, {'message': args.message, 'tree': current_tree, 'parents': [previous]})
        api('PATCH', '/git/refs/heads/' + branch, token, {'sha': new_commit['sha'], 'force': False})
        current = api('GET', '/git/ref/heads/' + branch, token)
        if current['object']['sha'] != new_commit['sha']:
            raise RuntimeError('Branch verification did not find the published commit')
        print(json.dumps({'published_commit': new_commit['sha'], 'url': new_commit['html_url'], 'files_written': len(entries)}), flush=True)
        return
    uploaded = set(known.values())
    last_write = 0.0
    for relative, content, expected in changed:
        if expected not in uploaded:
            delay = 1.3 - (time.monotonic() - last_write)
            if delay > 0:
                time.sleep(delay)
            blob = api('POST', '/git/blobs', token, {'encoding': 'base64', 'content': base64.b64encode(content).decode()})
            last_write = time.monotonic()
            if blob['sha'] != expected:
                raise RuntimeError('GitHub blob integrity mismatch: ' + relative)
            uploaded.add(expected)
        entries.append({'path': relative, 'mode': '100644', 'type': 'blob', 'sha': expected})
        if len(entries) % 25 == 0:
            print(json.dumps({'verified_files': len(entries), 'total': len(changed)}), flush=True)
    new_tree = api('POST', '/git/trees', token, {'base_tree': commit['tree']['sha'], 'tree': entries})
    new_commit = api('POST', '/git/commits', token, {'message': args.message, 'tree': new_tree['sha'], 'parents': [previous]})
    # Never force: a concurrent branch update must fail instead of losing work.
    api('PATCH', '/git/refs/heads/' + branch, token, {'sha': new_commit['sha'], 'force': False})
    current = api('GET', '/git/ref/heads/' + branch, token)
    if current['object']['sha'] != new_commit['sha']:
        raise RuntimeError('Branch verification did not find the published commit')
    print(json.dumps({'published_commit': new_commit['sha'], 'url': new_commit['html_url'], 'files_written': len(entries)}), flush=True)


if __name__ == '__main__':
    main()
