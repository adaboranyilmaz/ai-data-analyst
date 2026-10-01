output "url" {
  description = "The demo's address."
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "resource_group" {
  description = "Everything is in this group: deleting it deletes all of it."
  value       = azurerm_resource_group.demo.name
}
