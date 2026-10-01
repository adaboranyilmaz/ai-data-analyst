# The demo in replay mode on Azure Container Apps: recorded runs only, no key, no database. One
# resource group, so `terraform destroy` removes everything that costs money. The budget is
# created first and everything else depends on it.

resource "azurerm_resource_group" "demo" {
  name     = "rg-${var.name}"
  location = var.location
  tags     = local.tags
}

locals {
  tags = {
    project = "ai-data-analyst"
    purpose = "demo"
  }
}

# A budget alert on the resource group, before anything that costs money exists. It warns; it does
# not stop spending, so the stack is also destroyed the same day.
resource "azurerm_consumption_budget_resource_group" "cap" {
  name              = "budget-${var.name}"
  resource_group_id = azurerm_resource_group.demo.id
  amount            = var.budget_usd
  time_grain        = "Monthly"

  time_period {
    start_date = var.budget_start
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    contact_emails = [var.alert_email]
  }
}

resource "azurerm_log_analytics_workspace" "demo" {
  name                = "log-${var.name}"
  location            = azurerm_resource_group.demo.location
  resource_group_name = azurerm_resource_group.demo.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  daily_quota_gb      = 0.1
  tags                = local.tags

  depends_on = [azurerm_consumption_budget_resource_group.cap]
}

resource "azurerm_container_app_environment" "demo" {
  name                       = "cae-${var.name}"
  location                   = azurerm_resource_group.demo.location
  resource_group_name        = azurerm_resource_group.demo.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.demo.id
  tags                       = local.tags
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${var.name}"
  container_app_environment_id = azurerm_container_app_environment.demo.id
  resource_group_name          = azurerm_resource_group.demo.name
  revision_mode                = "Single"
  tags                         = local.tags

  dynamic "registry" {
    for_each = var.registry_server == null ? [] : [1]
    content {
      server               = var.registry_server
      username             = var.registry_username
      password_secret_name = "registry-password"
    }
  }

  dynamic "secret" {
    for_each = var.registry_server == null ? [] : [1]
    content {
      name  = "registry-password"
      value = var.registry_password
    }
  }

  template {
    min_replicas = 0
    max_replicas = var.max_replicas

    container {
      name   = "api"
      image  = var.image
      cpu    = 0.5
      memory = "1Gi"

      # Replay mode: recorded runs only, no key, no database. The service holds no secret.
      env {
        name  = "ANALYST_SERVING_MODE"
        value = "replay"
      }

      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000
      }

      readiness_probe {
        transport = "HTTP"
        path      = "/ready"
        port      = 8000
      }
    }

    http_scale_rule {
      name                = "http"
      concurrent_requests = "50"
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}
