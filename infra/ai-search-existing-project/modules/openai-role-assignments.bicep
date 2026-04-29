targetScope = 'resourceGroup'

@description('Name of the existing Azure OpenAI / Foundry account.')
param openAiAccountName string

@description('Principal id of the Azure AI Search managed identity.')
param principalId string

@description('Search service name, used only to stabilize role assignment names.')
param searchServiceName string

@description('Role definition id for Cognitive Services OpenAI User.')
param cognitiveServicesOpenAiUserRoleId string

@description('Role definition id for Azure AI User.')
param azureAiUserRoleId string

@description('Role definition id for Cognitive Services User.')
param cognitiveServicesUserRoleId string

resource openAiAccount 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' existing = {
  name: openAiAccountName
}

resource searchServiceOpenAiUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, openAiAccount.id, searchServiceName, cognitiveServicesOpenAiUserRoleId)
  scope: openAiAccount
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAiUserRoleId)
  }
}

resource searchServiceAzureAiUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, openAiAccount.id, searchServiceName, azureAiUserRoleId)
  scope: openAiAccount
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', azureAiUserRoleId)
  }
}

resource searchServiceCognitiveServicesUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, openAiAccount.id, searchServiceName, cognitiveServicesUserRoleId)
  scope: openAiAccount
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
  }
}

output roleAssignmentIds object = {
  searchServiceOpenAiUser: searchServiceOpenAiUserRoleAssignment.id
  searchServiceAzureAiUser: searchServiceAzureAiUserRoleAssignment.id
  searchServiceCognitiveServicesUser: searchServiceCognitiveServicesUserRoleAssignment.id
}
