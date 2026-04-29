targetScope = 'resourceGroup'

@description('Location for the user-assigned managed identity.')
param location string = resourceGroup().location

@description('Name of the user-assigned managed identity to create for the Foundry IQ flow.')
param userAssignedIdentityName string

@description('Name of the existing Azure AI Search service that should be associated with the UAMI.')
param searchServiceName string

@description('Resource group containing the existing Azure OpenAI / Foundry account.')
param openAiAccountResourceGroup string

@description('Name of the existing Azure OpenAI / Foundry account.')
param openAiAccountName string

@description('Whether to grant the UAMI Cognitive Services OpenAI User on the OpenAI / Foundry account.')
param assignCognitiveServicesOpenAiUser bool = true

@description('Whether to grant the UAMI Azure AI User on the OpenAI / Foundry account.')
param assignAzureAiUser bool = true

@description('Whether to grant the UAMI Cognitive Services User on the OpenAI / Foundry account.')
param assignCognitiveServicesUser bool = true
 
var roleIds = {
  cognitiveServicesOpenAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  azureAiUser: '53ca6127-db72-4b80-b1b0-d745d6d5456d'
  cognitiveServicesUser: 'a97b65f3-24c7-4388-baec-2e87135dc908'
}

resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: userAssignedIdentityName
  location: location
}

module openAiRoleAssignments '../ai-search-existing-project/modules/openai-role-assignments.bicep' = if (assignCognitiveServicesOpenAiUser || assignAzureAiUser || assignCognitiveServicesUser) {
  name: 'fiqUamiOpenAiRoleAssignments-${uniqueString(openAiAccountName, userAssignedIdentity.id)}'
  scope: resourceGroup(openAiAccountResourceGroup)
  params: {
    openAiAccountName: openAiAccountName
    principalId: userAssignedIdentity.properties.principalId
    searchServiceName: userAssignedIdentityName
    cognitiveServicesOpenAiUserRoleId: roleIds.cognitiveServicesOpenAiUser
    azureAiUserRoleId: roleIds.azureAiUser
    cognitiveServicesUserRoleId: roleIds.cognitiveServicesUser
  }
}

output userAssignedIdentityName string = userAssignedIdentity.name
output userAssignedIdentityResourceId string = userAssignedIdentity.id
output userAssignedIdentityPrincipalId string = userAssignedIdentity.properties.principalId
output userAssignedIdentityClientId string = userAssignedIdentity.properties.clientId
output searchServiceResourceId string = resourceId('Microsoft.Search/searchServices', searchServiceName)
output roleAssignmentIds object = {
  uamiCognitiveServicesOpenAiUser: assignCognitiveServicesOpenAiUser ? openAiRoleAssignments!.outputs.roleAssignmentIds.searchServiceOpenAiUser : ''
  uamiAzureAiUser: assignAzureAiUser ? openAiRoleAssignments!.outputs.roleAssignmentIds.searchServiceAzureAiUser : ''
  uamiCognitiveServicesUser: assignCognitiveServicesUser ? openAiRoleAssignments!.outputs.roleAssignmentIds.searchServiceCognitiveServicesUser : ''
}
