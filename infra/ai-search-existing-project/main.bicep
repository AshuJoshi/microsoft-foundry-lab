targetScope = 'resourceGroup'

@description('Location for the Azure AI Search service.')
param location string = resourceGroup().location

@description('Name of the Azure AI Search service to create.')
param searchServiceName string

@description('Azure AI Search service SKU.')
@allowed([
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
  'storage_optimized_l2'
])
param searchServiceSku string = 'standard'

@description('Azure AD object ID of the user who should manage and populate the search service.')
param userObjectId string

@description('Managed identity principal ID of the existing Foundry project.')
param foundryProjectPrincipalId string

@description('Name of the existing Azure OpenAI / Foundry account that backs the vectorizer.')
param openAiAccountName string

@description('Resource group containing the existing Azure OpenAI / Foundry account.')
param openAiAccountResourceGroup string

@description('Whether to grant the current user Search Service Contributor on the new search service.')
param assignUserSearchContributor bool = true

@description('Whether to grant the current user Search Index Data Reader on the new search service.')
param assignUserSearchIndexReader bool = true

@description('Whether to grant the current user Search Index Data Contributor on the new search service.')
param assignUserSearchIndexContributor bool = true

@description('Whether to grant the Foundry project managed identity Search Index Data Reader on the new search service.')
param assignFoundryProjectSearchIndexReader bool = true

@description('Whether to grant the Foundry project managed identity Search Index Data Contributor on the new search service.')
param assignFoundryProjectSearchIndexContributor bool = true

@description('Whether to grant the Foundry project managed identity Search Service Contributor on the new search service.')
param assignFoundryProjectSearchServiceContributor bool = true

@description('Whether to grant the Azure AI Search managed identity the workshop-style Azure OpenAI roles on the existing OpenAI account.')
param assignSearchManagedIdentityOpenAiRoles bool = true

var roleIds = {
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
  searchIndexDataReader: '1407120a-92aa-4202-b7e9-c0e197c71c8f'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  cognitiveServicesOpenAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  azureAiUser: '53ca6127-db72-4b80-b1b0-d745d6d5456d'
  cognitiveServicesUser: 'a97b65f3-24c7-4388-baec-2e87135dc908'
}

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: {
    name: searchServiceSku
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    networkRuleSet: {
      ipRules: []
    }
    encryptionWithCmk: {
      enforcement: 'Unspecified'
    }
    disableLocalAuth: false
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
    semanticSearch: 'standard'
  }
}

resource openAiAccount 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' existing = {
  scope: resourceGroup(openAiAccountResourceGroup)
  name: openAiAccountName
}

resource userSearchContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignUserSearchContributor) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, userObjectId, roleIds.searchServiceContributor)
  scope: searchService
  properties: {
    principalId: userObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchServiceContributor)
  }
}

resource userSearchIndexReaderRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignUserSearchIndexReader) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, userObjectId, roleIds.searchIndexDataReader)
  scope: searchService
  properties: {
    principalId: userObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataReader)
  }
}

resource userSearchIndexContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignUserSearchIndexContributor) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, userObjectId, roleIds.searchIndexDataContributor)
  scope: searchService
  properties: {
    principalId: userObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataContributor)
  }
}

resource foundryProjectSearchIndexReaderRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignFoundryProjectSearchIndexReader) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, foundryProjectPrincipalId, roleIds.searchIndexDataReader)
  scope: searchService
  properties: {
    principalId: foundryProjectPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataReader)
  }
}

resource foundryProjectSearchIndexContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignFoundryProjectSearchIndexContributor) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, foundryProjectPrincipalId, roleIds.searchIndexDataContributor)
  scope: searchService
  properties: {
    principalId: foundryProjectPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataContributor)
  }
}

resource foundryProjectSearchServiceContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignFoundryProjectSearchServiceContributor) {
  name: guid(subscription().id, resourceGroup().id, searchService.name, foundryProjectPrincipalId, roleIds.searchServiceContributor)
  scope: searchService
  properties: {
    principalId: foundryProjectPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchServiceContributor)
  }
}

module openAiRoleAssignments 'modules/openai-role-assignments.bicep' = if (assignSearchManagedIdentityOpenAiRoles) {
  name: 'openAiRoleAssignments-${uniqueString(openAiAccount.id, searchService.name)}'
  scope: resourceGroup(openAiAccountResourceGroup)
  params: {
    openAiAccountName: openAiAccountName
    principalId: searchService.identity.principalId
    searchServiceName: searchService.name
    cognitiveServicesOpenAiUserRoleId: roleIds.cognitiveServicesOpenAiUser
    azureAiUserRoleId: roleIds.azureAiUser
    cognitiveServicesUserRoleId: roleIds.cognitiveServicesUser
  }
}

output searchServiceName string = searchService.name
output searchServiceEndpoint string = 'https://${searchService.name}.search.windows.net'
output searchServicePrincipalId string = searchService.identity.principalId
output searchResourceId string = searchService.id

output openAiAccountResourceId string = openAiAccount.id

output roleAssignmentIds object = {
  userSearchContributor: assignUserSearchContributor ? userSearchContributorRoleAssignment.id : ''
  userSearchIndexReader: assignUserSearchIndexReader ? userSearchIndexReaderRoleAssignment.id : ''
  userSearchIndexContributor: assignUserSearchIndexContributor ? userSearchIndexContributorRoleAssignment.id : ''
  foundryProjectSearchIndexReader: assignFoundryProjectSearchIndexReader ? foundryProjectSearchIndexReaderRoleAssignment.id : ''
  foundryProjectSearchIndexContributor: assignFoundryProjectSearchIndexContributor ? foundryProjectSearchIndexContributorRoleAssignment.id : ''
  foundryProjectSearchServiceContributor: assignFoundryProjectSearchServiceContributor ? foundryProjectSearchServiceContributorRoleAssignment.id : ''
  searchServiceOpenAiUser: assignSearchManagedIdentityOpenAiRoles ? openAiRoleAssignments.outputs.roleAssignmentIds.searchServiceOpenAiUser : ''
  searchServiceAzureAiUser: assignSearchManagedIdentityOpenAiRoles ? openAiRoleAssignments.outputs.roleAssignmentIds.searchServiceAzureAiUser : ''
  searchServiceCognitiveServicesUser: assignSearchManagedIdentityOpenAiRoles ? openAiRoleAssignments.outputs.roleAssignmentIds.searchServiceCognitiveServicesUser : ''
}
