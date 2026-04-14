import urllib.request
import json
import os
import ssl
import certifi

context = ssl.create_default_context(cafile=certifi.where())

plugins = {
    "dataview": "blacksmithgu/obsidian-dataview",
    "templater-obsidian": "SilentVoid13/Templater",
    "graph-analysis": "SkepticMystic/graph-analysis"
}

base_dir = "wiki/.obsidian/plugins"

for p_dir, repo in plugins.items():
    print(f"Fetching {repo}...")
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=context) as response:
            data = json.loads(response.read().decode())
            assets = data.get('assets', [])
            
            out_dir = os.path.join(base_dir, p_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            for asset in assets:
                name = asset['name']
                if name in ['main.js', 'styles.css', 'manifest.json']:
                    dl_url = asset['browser_download_url']
                    print(f"  Downloading {name}...")
                    req_dl = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_dl, context=context) as response_dl:
                        with open(os.path.join(out_dir, name), 'wb') as f:
                            f.write(response_dl.read())
    except Exception as e:
        print(f"Error fetching {repo}: {e}")
