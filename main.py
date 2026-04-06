import pandas as pd

df = pd.read_csv("Anonymised Permissions.csv")

#print(df.head())

# Step 1: Feature Engineering

# Count permissions per app
perm_counts = df.groupby('ClientDisplayName')['Permission'].count()

# Unique permissions
unique_perms = df.groupby('ClientDisplayName')['Permission'].nunique()

# Application vs Delegated counts
type_counts = df.pivot_table(
    index='ClientDisplayName', # Each row in the result = one app
    columns='PermissionType', # Split counts into columns: Application, Delegated
    values='Permission', # This is what we are counting
    aggfunc='count', # Count how many permissions exist in each category
    fill_value=0
)

# Combine features
features = pd.concat([perm_counts, unique_perms, type_counts], axis=1)
features.columns = ['total_perms', 'unique_perms', 'application_perms', 'delegated_perms']

features.fillna(0, inplace=True)
#print(features.head())


# Step 2: Add Risk Sensitive Features
#'''
high_risk_keywords = ['Write', 'ReadWrite', 'All']

def count_high_risk(perms):
    return sum(any(k in p for k in high_risk_keywords) for p in perms)

high_risk = df.groupby('ClientDisplayName')['Permission'].apply(list).apply(count_high_risk)

features['high_risk_perms'] = high_risk
#'''
#print(features.head())

# Step 3: Normalize Features
#'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(features)
#'''

# Step 4: Model/Algo 1 - Isolation Forest
'''
Normal apps are similar to each other → hard to separate
Weird apps are different → easy to isolate
The model:
- randomly splits the data
- tries to isolate each point (app)
- anomalies get isolated very quickly
'''
#'''
from sklearn.ensemble import IsolationForest
# contamination=0.1 Means about 10% of apps are suspicious. The model needs to know how many anomalies to expect.
# random_state=42 Ensures same results every time you run the code.
iso = IsolationForest(contamination=0.1, random_state=42)
features['isoforest'] = iso.fit_predict(X)
features['isoforest'] = iso.fit_predict(X)
#features['isoforest_real'] = iso.decision_function(X)
#'''
#print(features.head())

# Step 5: Model/Algo 2 - OneClassSVM
#'''
from sklearn.svm import OneClassSVM
# Draw a shape around normal apps — anything outside = suspicious
# kernel="rbf" RBF = non-linear boundary
# gamma="scale"  Controls how sensitive the boundary is: high gamma → very tight boundary (more anomalies); low gamma → smoother boundary
svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale") 
features['svm'] = svm.fit_predict(X)
#'''
#print(features.head())

#Step 6: Model/Algo 3 - DBSCAN
#'''
from sklearn.cluster import DBSCAN
# Groups apps into clusters based on similarity: dense groups → normal behavior; isolated points → anomalies
# eps=1.5 Maximum distance between points to be considered “neighbors”. small eps → more anomalies. large eps → fewer anomalies
# min_samples=3 Minimum number of nearby points to form a cluster
#If a point: has < 3 neighbors → considered noise (anomaly)
db = DBSCAN(eps=1.5, min_samples=3)
features['cluster'] = db.fit_predict(X)
#'''
#print(features.head())

# Step 7: Identifying Suspicious Apps
'''
anomalies = features[
    (features['isoforest'] == -1) |
    (features['svm'] == -1) |
    (features['cluster'] == -1)
]

#print(anomalies.sort_values(by='high_risk_perms', ascending=False).head(10))
'''

# Step 8: Combine results
#'''
features['final_anomaly'] = (
    (features['isoforest'] == -1).astype(int) +
    (features['svm'] == -1).astype(int) +
    (features['cluster'] == -1).astype(int)
)
#print(features.sort_values(by='final_anomaly', ascending=False).head(15))
#'''

# Step 9: Visualization
# #'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for legend patches

algos = ['isoforest', 'svm', 'cluster', 'final_anomaly']

for idx, algo in enumerate(algos, start=1):
    plt.figure(figsize=(10,6))
    
    # Scatter plot colored by the current algorithm/model
    scatter = plt.scatter(
        features['total_perms'], 
        features['high_risk_perms'],
        c=features[algo],
        cmap='tab10',
        s=100
    )
    
    plt.xlabel("Total Permissions")
    plt.ylabel("High Risk Permissions")
    plt.title(f"Anomaly Detection of Apps - {algo}")
    
    # Annotate each point with app name
    for i, app_name in enumerate(features.index):
        plt.text(
            features['total_perms'].iloc[i] + 0.1,
            features['high_risk_perms'].iloc[i] + 0.1,
            app_name,
            fontsize=8
        )
    
    # Create legend below the plot
    unique_values = sorted(features[algo].unique())
    colors = [scatter.cmap(scatter.norm(val)) for val in unique_values]
    patches = [mpatches.Patch(color=colors[i], label=str(unique_values[i])) for i in range(len(unique_values))]
    
    plt.legend(handles=patches, title=f"{algo} value", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(unique_values))
    
    # Save the figure
    plt.savefig(f"Figure{idx}_{algo}.png", dpi=300, bbox_inches='tight')
    plt.close()
#'''
