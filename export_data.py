import wandb
import pandas as pd

# Log in (optional if already logged in)
wandb.login()

# Replace with your entity/project and run ID
api = wandb.Api()
run = api.run("jeffzc-umich/Our_DOPE/evdk5ncs") # run ID: in the URL of the run page

# Get all logged history (metrics like accuracy, loss, etc.)
history = run.history(keys=None)

# Save to CSV
history.to_csv("vit_cookies_4.csv", index=False)

print("Exported")
