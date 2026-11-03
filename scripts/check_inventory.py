import subprocess
import os
from huggingface_hub import HfApi

git_files = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())

api = HfApi()
hf_files = set(api.list_repo_files('Laxminarayan24/medshield-checkpoints', repo_type='model'))

categories = {
    'on_git': [],
    'on_hf': [],
    'venv_or_cache': [],
    'datasets_mimic': [],
    'other_unbacked': []
}

for root, dirs, files in os.walk('.'):
    if '.git' in root.split(os.sep):
        continue
    for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), '.').replace('\\', '/')
        sz = os.path.getsize(rel_path)
        
        if rel_path in git_files:
            categories['on_git'].append((rel_path, sz))
        elif rel_path in hf_files:
            categories['on_hf'].append((rel_path, sz))
        elif rel_path.startswith('.venv/') or rel_path.startswith('.claude/'):
            categories['venv_or_cache'].append((rel_path, sz))
        elif rel_path.startswith('MIMIC -III (10000 patients)/') or rel_path.startswith('data/') or rel_path.startswith('models/'):
            categories['datasets_mimic'].append((rel_path, sz))
        else:
            categories['other_unbacked'].append((rel_path, sz))

print('=== FILE INVENTORY & CLOUD BACKUP AUDIT ===')
print(f'1. Files Tracked on GitHub:      {len(categories["on_git"]):>5} files ({sum(s for _,s in categories["on_git"])/(1024**2):.2f} MB)')
print(f'2. Files on Hugging Face:        {len(categories["on_hf"]):>5} files ({sum(s for _,s in categories["on_hf"])/(1024**2):.2f} MB)')
print(f'3. Python Virtual Env (.venv):   {len(categories["venv_or_cache"]):>5} files ({sum(s for _,s in categories["venv_or_cache"])/(1024**2):.2f} MB)')
print(f'4. Datasets & Raw Cache:         {len(categories["datasets_mimic"]):>5} files ({sum(s for _,s in categories["datasets_mimic"])/(1024**2):.2f} MB)')
print(f'5. Other Unbacked Files:         {len(categories["other_unbacked"]):>5} files ({sum(s for _,s in categories["other_unbacked"])/(1024**2):.2f} MB)')

print('\nDetailed check of Category 5 (Other Unbacked Files):')
for f, s in categories['other_unbacked']:
    print(f'  {f:60s} ({s / 1024:.1f} KB)')
