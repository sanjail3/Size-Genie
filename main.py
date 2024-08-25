import pandas as pd
from agent.agent import AIAgent
from clustering.clustering_technique import CustomClustering



data = {
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
    'Chest': [39, 41, 43, 45, 47],
    'Brand Size': ['S', 'M', 'L', 'XL', 'XXL'],
    'Shoulder': [17.5, 18, 18.5, 19, 19.5],
    'Length': [27, 28, 29, 30, 30.5],
    'Sleeve Length': [25, 25.5, 26, 26.5, 27]
}

size_chart = pd.DataFrame(data)
database_schema = ""
db_url = ""


ai_agent = AIAgent(db_url=db_url)
df = ai_agent.generate_query(size_chart=size_chart, database_schema=database_schema)


custom_clustering = CustomClustering(df)
custom_clustering.preprocess_data(columns_to_drop=["Purchased Size", "Brand Size"])
optimal_k = custom_clustering.determine_optimal_clusters(max_k=10)
clustered_df = custom_clustering.apply_clustering(optimal_k)


updated_size_chart = custom_clustering.update_size_chart(size_chart)

print("Updated Size Chart:")
print(updated_size_chart)
