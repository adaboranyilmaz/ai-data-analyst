# The demo on Azure Container Apps

The service in replay mode (recorded runs only, no key, no database) as one container app, in one resource group, so deleting the group deletes everything that costs money. The budget alert is created first, and every other resource depends on it.

The app scales to zero when idle, one replica runs at half a vCPU and 1 GiB, and the log workspace is capped at a tenth of a gigabyte a day; the cost the portal reports after a run goes into the check's result. The budget (five dollars a month by default) warns at 50 and 80 percent actual and at 100 percent forecast. It warns; it does not stop spending, so destroy the stack the same day.

## Before

- Terraform 1.9 or newer and the Azure CLI, signed in (`az login`).
- The image published: merge to main runs the publish workflow, which pushes `ghcr.io/<owner>/<repository>/api`. A public repository gives a public package after you set its visibility to public once (package settings). If the repository is still private, push the image to an Azure Container Registry (Basic tier) and set the `registry_*` variables instead.

## Run it (PowerShell, from this folder)

```
Copy-Item terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` (subscription, alert email, and the first day of the month you deploy in).

```
terraform init
terraform plan -out tfplan
terraform apply tfplan
```

Then check it from outside (from the repository root; it writes `results/metrics/cloud_deploy.json`):

```
uv run python scripts/98_cloud_check.py --url (terraform -chdir=infra output -raw url) --region westeurope
```

For the cold start, leave the app idle for about ten minutes before the check: the first request to a scaled-to-zero app starts it, and the check times that request on its own.

Destroy it the same day:

```
terraform destroy
```

`terraform.tfvars`, the state and the plan file are ignored by git: the state names the subscription.
