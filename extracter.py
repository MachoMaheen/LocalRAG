import requests
import base64
import json

# Function to get branches from the repository
def get_branches(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/branches"
    headers = {'Authorization': f'token {token}'}
    # response = requests.get(url, headers=headers)
    response = requests.get(url)

    if response.status_code == 200:
        return [branch['name'] for branch in response.json()]
    else:
        print(f"Error fetching branches: {response.status_code} - {response.text}")
        return []

# Function to get SHA hashes of files in a selected branch
def get_file_shas(owner, repo, branch_name, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch_name}?recursive=1"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return {item['path']: item['sha'] for item in response.json()['tree'] if item['type'] == 'blob'}
    else:
        print(f"Error fetching file SHAs: {response.status_code} - {response.text}")
        return {}

# Function to get file content using SHA
def get_file_content(owner, repo, sha, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs/{sha}"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content_data = response.json()
        # Check if content is base64 encoded
        if 'content' in content_data:
            try:
                decoded_content = base64.b64decode(content_data['content']).decode('utf-8')
                return decoded_content
            except UnicodeDecodeError:
                print(f"Warning: File with SHA {sha} is not UTF-8 encoded or is binary.")
                return None
    else:
        print(f"Error fetching file content: {response.status_code} - {response.text}")
        return None

# Main function to automate the process
def automate_github_data_collection(git_url, token):
    parts = git_url.split('/')
    owner = parts[-2]
    repo = parts[-1].replace('.git', '')

    # Step 1: Get branches
    branches = get_branches(owner, repo, token)
    print("Available branches:", branches)

    selected_branch = branches[0] if branches else None

    error_log = []

    if selected_branch:
        file_shas = get_file_shas(owner, repo, selected_branch, token)
        
        results = []
        for path, sha in file_shas.items():
            content = get_file_content(owner, repo, sha, token)
            if content is not None:
                results.append({"path": path, "content": content})
            else:
                error_log.append(path)

        with open('output.json', 'w') as f:
            json.dump(results, f, indent=4)

        with open('error_log.txt', 'w') as error_file:
            for error_path in error_log:
                error_file.write(f"{error_path}\n")

        print("Data collection complete. Results saved to output.json.")
        print("Errors logged to error_log.txt.")
    else:
        print("No branches found.")

# Example usage
git_url = "https://github.com/MachoMaheen/windowsGPT.git"
personal_access_token = "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"  # Replace with your actual token
automate_github_data_collection(git_url, personal_access_token)
