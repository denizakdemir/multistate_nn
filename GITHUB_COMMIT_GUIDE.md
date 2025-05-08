# GitHub Commit Guide

Follow these steps to commit the project to GitHub for the first time:

1. Initialize the git repository (if not already done):
```bash
cd /Users/denizakdemir/Dropbox/dakdemirGithub/GitHubProjects/multistate_nn
git init
```

2. Add all files:
```bash
git add .
```

3. Make the first commit:
```bash
git commit -m "Initial commit of MultiStateNN: Neural network models for multistate processes"
```

4. Add the remote repository:
```bash
git remote add origin git@github.com:denizakdemir/multistate_nn.git
```

5. Push to the remote repository:
```bash
git push -u origin main
```

Note: If your default branch is 'master' instead of 'main', replace 'main' with 'master' in the command above, or run this command first to rename your branch:
```bash
git branch -M main
```

Your project is now ready on GitHub\!
