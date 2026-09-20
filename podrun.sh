#!/bin/zsh
# Run commands on the RunPod pod.  The proxy only gives an interactive shell,
# so commands go in over stdin rather than as an ssh argument.
POD=84by4q4dkc5ozy-64410f27@ssh.runpod.io
{ cat; echo; echo exit; } | ssh -tt -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=25 -o ServerAliveInterval=10 -i ~/.ssh/id_ed25519 "$POD" 2>&1 \
  | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r//g'
