Run the following git commands in sequence, then commit with a thoughtful message:

1. Run `git status` and `git diff HEAD` to see what changed
2. Run `git log --oneline -5` to match this repo's commit style
3. Run `git add -u` to stage all modified tracked files
4. Craft a commit message with:
   - Subject line under 72 chars using an imperative verb (Add, Fix, Remove, Refactor…)
   - Focus on *why*, not just *what*
   - Short body paragraph if multiple unrelated changes are bundled
5. Run the commit using a HEREDOC, appending this trailer:
   `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
6. Run `git status` to confirm success

Do not push. Do not amend.
