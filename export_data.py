import wandb
import pandas as pd

# Log in (optional if already logged in)
wandb.login()

# Replace with your entity/project and run ID
api = wandb.Api()
run = api.run("jeffzc-umich/Our_DOPE/y4gh36m9")

# Get all logged history (metrics like accuracy, loss, etc.)
history = run.history(keys=None)

# Save to CSV
history.to_csv("original_cookies.csv", index=False)

print("Exported")
