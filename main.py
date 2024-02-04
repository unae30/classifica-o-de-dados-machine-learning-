from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) #Utilização da função "fetch..."para baixar o dataset de ID 94
  
# data (as pandas dataframes) 
X = spambase.data.features # Objetos que são retornados contentando os dados 
y = spambase.data.targets 
  
# metadata 
#print(spambase.metadata) 
  
# variable information 
#print(spambase.variables) 
#-----------------------------------------------------------------------------------------------------------
