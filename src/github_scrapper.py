import os
import json
import urllib.request
from pathlib import Path

GITHUB_USERNAME = "b-qt"
DATA_DIR = "./data"

def fetch_github_profile_and_repos():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch User Profile
    user_url = f"https://api.github.com/users/{GITHUB_USERNAME}"
    req = urllib.request.Request(user_url, headers={"User-Agent": "Python-Script"})
    
    summary_lines = ["--- GITHUB PROFILE SUMMARY ---\n"]
    try:
        with urllib.request.urlopen(req) as response:
            user_data = json.loads(response.read().decode("utf-8"))
            summary_lines.append(
                f"Username: {GITHUB_USERNAME}\n"
                f"Name: {user_data.get('name', 'Brenda Tebid')}\n"
                f"Bio: {user_data.get('bio', 'Data Engineer & Analytics Specialist')}\n"
                f"Public Repositories Count: {user_data.get('public_repos', 0)}\n"
                f"Profile URL: https://github.com/{GITHUB_USERNAME}\n\n"
            )
    except Exception as e:
        print(f"Error fetching profile: {e}")

    # 2. Fetch Public Repositories
    summary_lines.append("--- PUBLIC REPOSITORIES AND SIDE PROJECTS ---\n")
    repos_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos?per_page=100&sort=updated"
    req_repos = urllib.request.Request(repos_url, headers={"User-Agent": "Python-Script"})
    
    try:
        with urllib.request.urlopen(req_repos) as response:
            repos = json.loads(response.read().decode("utf-8"))
            for repo in repos:
                name = repo.get("name")
                desc = repo.get("description") or "No description provided."
                lang = repo.get("language") or "Python / SQL"
                url = repo.get("html_url")
                
                repo_block = (
                    f"Project Name: {name}\n"
                    f"Primary Language: {lang}\n"
                    f"Repository URL: {url}\n"
                    f"Project Description: {desc}\n"
                )
                
                # Fetch README content
                readme_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{name}/readme"
                req_readme = urllib.request.Request(
                    readme_url, 
                    headers={"Accept": "application/vnd.github.v3.raw", "User-Agent": "Python-Script"}
                )
                try:
                    with urllib.request.urlopen(req_readme) as readme_resp:
                        readme_text = readme_resp.read().decode("utf-8", errors="ignore")
                        # Keep first 1500 characters of README for dense semantic context
                        repo_block += f"README Summary:\n{readme_text[:1500]}\n"
                except Exception:
                    repo_block += "README Summary: Standard repository setup.\n"
                
                repo_block += "-" * 40 + "\n"
                summary_lines.append(repo_block)
                
        output_path = os.path.join(DATA_DIR, "github_summary_snapshot.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(summary_lines)
            
        print(f"✅ Generated crisp GitHub snapshot at {output_path}")
        
    except Exception as e:
        print(f"Error fetching repositories: {e}")

if __name__ == "__main__":
    fetch_github_profile_and_repos()